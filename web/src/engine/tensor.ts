// A tiny 2-D tensor with reverse-mode automatic differentiation — the same design as
// the Mac app's Swift engine. Every op records its inputs and a closure that pushes the
// output gradient back into them; backward() walks the graph in reverse order.

/** Small, fast, seedable RNG (mulberry32 + Box–Muller) so runs are reproducible. */
export class Rng {
  private s: number
  constructor(seed: number) { this.s = (seed >>> 0) || 1 }
  uniform(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0
    let t = this.s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  normal(): number {
    const u1 = Math.max(this.uniform(), 1e-12), u2 = this.uniform()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }
  int(n: number): number { return Math.floor(this.uniform() * n) % n }
}

export class Tensor {
  data: Float64Array
  grad: Float64Array
  parents: Tensor[] = []
  backwardFn: ((self: Tensor) => void) | null = null

  constructor(public rows: number, public cols: number, data?: ArrayLike<number>, public label = '') {
    this.data = data ? Float64Array.from(data) : new Float64Array(rows * cols)
    if (this.data.length !== rows * cols) throw new Error('shape mismatch')
    this.grad = new Float64Array(rows * cols)
  }

  static randn(rows: number, cols: number, std: number, rng: Rng, label = '') {
    const t = new Tensor(rows, cols, undefined, label)
    for (let i = 0; i < t.data.length; i++) t.data[i] = rng.normal() * std
    return t
  }
  static filled(rows: number, cols: number, v: number, label = '') {
    const t = new Tensor(rows, cols, undefined, label)
    t.data.fill(v)
    return t
  }

  get count() { return this.rows * this.cols }
  get item() { return this.data[0] }
  row(r: number): number[] { return Array.from(this.data.subarray(r * this.cols, (r + 1) * this.cols)) }
  zeroGrad() { this.grad.fill(0) }
  copy() { return new Tensor(this.rows, this.cols, this.data, this.label) }

  /** Back-propagate from this scalar through the whole graph. */
  backward() {
    const order: Tensor[] = []
    const seen = new Set<Tensor>()
    const stack: [Tensor, boolean][] = [[this, false]]
    while (stack.length) {
      const [t, done] = stack.pop()!
      if (done) { order.push(t); continue }
      if (seen.has(t)) continue
      seen.add(t)
      stack.push([t, true])
      for (const p of t.parents) if (!seen.has(p)) stack.push([p, false])
    }
    this.grad.fill(1)
    for (let i = order.length - 1; i >= 0; i--) order[i].backwardFn?.(order[i])
  }
}

export const Ops = {
  /** (R×K)·(K×C) → R×C */
  matmul(a: Tensor, b: Tensor): Tensor {
    if (a.cols !== b.rows) throw new Error(`matmul ${a.rows}x${a.cols} · ${b.rows}x${b.cols}`)
    const R = a.rows, K = a.cols, C = b.cols
    const out = new Tensor(R, C)
    const A = a.data, B = b.data, O = out.data
    for (let i = 0; i < R; i++) for (let k = 0; k < K; k++) {
      const av = A[i * K + k]
      if (av === 0) continue
      const bo = k * C, oo = i * C
      for (let j = 0; j < C; j++) O[oo + j] += av * B[bo + j]
    }
    out.parents = [a, b]
    out.backwardFn = (o) => {
      const G = o.grad, dA = a.grad, dB = b.grad
      for (let i = 0; i < R; i++) for (let k = 0; k < K; k++) {
        let s = 0
        const go = i * C, bo = k * C
        for (let j = 0; j < C; j++) s += G[go + j] * B[bo + j]
        dA[i * K + k] += s
        const av = A[i * K + k]
        if (av !== 0) for (let j = 0; j < C; j++) dB[bo + j] += av * G[go + j]
      }
    }
    return out
  },

  transpose(a: Tensor): Tensor {
    const out = new Tensor(a.cols, a.rows)
    for (let i = 0; i < a.rows; i++) for (let j = 0; j < a.cols; j++) out.data[j * a.rows + i] = a.data[i * a.cols + j]
    out.parents = [a]
    out.backwardFn = (o) => {
      for (let i = 0; i < a.rows; i++) for (let j = 0; j < a.cols; j++) a.grad[i * a.cols + j] += o.grad[j * a.rows + i]
    }
    return out
  },

  add(a: Tensor, b: Tensor): Tensor {
    const out = new Tensor(a.rows, a.cols)
    for (let i = 0; i < a.count; i++) out.data[i] = a.data[i] + b.data[i]
    out.parents = [a, b]
    out.backwardFn = (o) => { for (let i = 0; i < o.count; i++) { a.grad[i] += o.grad[i]; b.grad[i] += o.grad[i] } }
    return out
  },

  /** Add a 1×C row vector to every row. */
  addRow(a: Tensor, bias: Tensor): Tensor {
    const out = new Tensor(a.rows, a.cols), C = a.cols
    for (let i = 0; i < a.rows; i++) for (let j = 0; j < C; j++) out.data[i * C + j] = a.data[i * C + j] + bias.data[j]
    out.parents = [a, bias]
    out.backwardFn = (o) => {
      for (let i = 0; i < a.rows; i++) for (let j = 0; j < C; j++) { const g = o.grad[i * C + j]; a.grad[i * C + j] += g; bias.grad[j] += g }
    }
    return out
  },

  scale(a: Tensor, s: number): Tensor {
    const out = new Tensor(a.rows, a.cols)
    for (let i = 0; i < a.count; i++) out.data[i] = a.data[i] * s
    out.parents = [a]
    out.backwardFn = (o) => { for (let i = 0; i < a.count; i++) a.grad[i] += o.grad[i] * s }
    return out
  },

  /** GELU, tanh approximation (as in GPT-2). */
  gelu(a: Tensor): Tensor {
    const k = Math.sqrt(2 / Math.PI)
    const out = new Tensor(a.rows, a.cols)
    const th = new Float64Array(a.count)
    for (let i = 0; i < a.count; i++) {
      const x = a.data[i], t = Math.tanh(k * (x + 0.044715 * x * x * x))
      th[i] = t; out.data[i] = 0.5 * x * (1 + t)
    }
    out.parents = [a]
    out.backwardFn = (o) => {
      for (let i = 0; i < a.count; i++) {
        const x = a.data[i], t = th[i]
        a.grad[i] += o.grad[i] * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * 0.044715 * x * x))
      }
    }
    return out
  },

  /** Row-wise softmax where row i may only see columns 0…i (the causal mask). */
  causalSoftmax(a: Tensor): Tensor {
    const n = a.rows, out = new Tensor(n, n)
    for (let i = 0; i < n; i++) {
      let m = -Infinity
      for (let j = 0; j <= i; j++) m = Math.max(m, a.data[i * n + j])
      let s = 0
      for (let j = 0; j <= i; j++) { const e = Math.exp(a.data[i * n + j] - m); out.data[i * n + j] = e; s += e }
      for (let j = 0; j <= i; j++) out.data[i * n + j] /= s
    }
    out.parents = [a]
    out.backwardFn = (o) => {
      for (let i = 0; i < n; i++) {
        let dot = 0
        for (let j = 0; j <= i; j++) dot += o.data[i * n + j] * o.grad[i * n + j]
        for (let j = 0; j <= i; j++) a.grad[i * n + j] += o.data[i * n + j] * (o.grad[i * n + j] - dot)
      }
    }
    return out
  },

  layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps = 1e-5): Tensor {
    const R = x.rows, C = x.cols, out = new Tensor(R, C)
    const xhat = new Float64Array(R * C), rstd = new Float64Array(R)
    for (let i = 0; i < R; i++) {
      let mean = 0
      for (let j = 0; j < C; j++) mean += x.data[i * C + j]
      mean /= C
      let v = 0
      for (let j = 0; j < C; j++) { const d = x.data[i * C + j] - mean; v += d * d }
      const r = 1 / Math.sqrt(v / C + eps)
      rstd[i] = r
      for (let j = 0; j < C; j++) {
        const h = (x.data[i * C + j] - mean) * r
        xhat[i * C + j] = h
        out.data[i * C + j] = h * gamma.data[j] + beta.data[j]
      }
    }
    out.parents = [x, gamma, beta]
    out.backwardFn = (o) => {
      for (let i = 0; i < R; i++) {
        let mD = 0, mDX = 0
        for (let j = 0; j < C; j++) {
          const g = o.grad[i * C + j]
          gamma.grad[j] += g * xhat[i * C + j]
          beta.grad[j] += g
          const d = g * gamma.data[j]
          mD += d; mDX += d * xhat[i * C + j]
        }
        mD /= C; mDX /= C
        for (let j = 0; j < C; j++) {
          const d = o.grad[i * C + j] * gamma.data[j]
          x.grad[i * C + j] += rstd[i] * (d - mD - xhat[i * C + j] * mDX)
        }
      }
    }
    return out
  },

  /** One row of the table per token id. */
  embedding(table: Tensor, ids: number[]): Tensor {
    const C = table.cols, out = new Tensor(ids.length, C)
    ids.forEach((id, i) => { for (let j = 0; j < C; j++) out.data[i * C + j] = table.data[id * C + j] })
    out.parents = [table]
    out.backwardFn = (o) => { ids.forEach((id, i) => { for (let j = 0; j < C; j++) table.grad[id * C + j] += o.grad[i * C + j] }) }
    return out
  },

  firstRows(a: Tensor, n: number) { return Ops.embedding(a, Array.from({ length: n }, (_, i) => i)) },

  sliceCols(a: Tensor, start: number, count: number): Tensor {
    const out = new Tensor(a.rows, count)
    for (let i = 0; i < a.rows; i++) for (let j = 0; j < count; j++) out.data[i * count + j] = a.data[i * a.cols + start + j]
    out.parents = [a]
    out.backwardFn = (o) => { for (let i = 0; i < a.rows; i++) for (let j = 0; j < count; j++) a.grad[i * a.cols + start + j] += o.grad[i * count + j] }
    return out
  },

  concatCols(parts: Tensor[]): Tensor {
    const R = parts[0].rows, C = parts.reduce((s, p) => s + p.cols, 0), out = new Tensor(R, C)
    const offs: number[] = []
    let off = 0
    for (const p of parts) {
      offs.push(off)
      for (let i = 0; i < R; i++) for (let j = 0; j < p.cols; j++) out.data[i * C + off + j] = p.data[i * p.cols + j]
      off += p.cols
    }
    out.parents = parts
    out.backwardFn = (o) => {
      parts.forEach((p, k) => { for (let i = 0; i < R; i++) for (let j = 0; j < p.cols; j++) p.grad[i * p.cols + j] += o.grad[i * C + offs[k] + j] })
    }
    return out
  },

  /** Mean cross-entropy of row-wise softmax(logits) against integer targets; `weights` is the loss mask. */
  crossEntropy(logits: Tensor, targets: number[], weights?: number[]): Tensor {
    const R = logits.rows, V = logits.cols
    const w = weights ?? new Array(R).fill(1)
    const total = Math.max(w.reduce((a, b) => a + b, 0), 1e-12)
    const probs = new Float64Array(R * V)
    let loss = 0
    for (let i = 0; i < R; i++) {
      let m = -Infinity
      for (let j = 0; j < V; j++) m = Math.max(m, logits.data[i * V + j])
      let s = 0
      for (let j = 0; j < V; j++) { const e = Math.exp(logits.data[i * V + j] - m); probs[i * V + j] = e; s += e }
      for (let j = 0; j < V; j++) probs[i * V + j] /= s
      loss -= w[i] * Math.log(Math.max(probs[i * V + targets[i]], 1e-300))
    }
    const out = new Tensor(1, 1, [loss / total])
    out.parents = [logits]
    out.backwardFn = (o) => {
      const g = o.grad[0] / total
      for (let i = 0; i < R; i++) {
        if (w[i] === 0) continue
        for (let j = 0; j < V; j++) logits.grad[i * V + j] += g * w[i] * (probs[i * V + j] - (j === targets[i] ? 1 : 0))
      }
    }
    return out
  },

  mean(xs: Tensor[]): Tensor {
    const n = xs.length
    const out = new Tensor(1, 1, [xs.reduce((s, x) => s + x.data[0], 0) / n])
    out.parents = xs
    out.backwardFn = (o) => { for (const x of xs) x.grad[0] += o.grad[0] / n }
    return out
  },
}

export const Num = {
  softmax(xs: number[], temperature = 1): number[] {
    const t = Math.max(temperature, 1e-6)
    const m = Math.max(...xs)
    const e = xs.map((x) => Math.exp((x - m) / t))
    const s = e.reduce((a, b) => a + b, 0)
    return e.map((x) => x / s)
  },
  sample(probs: number[], rng: Rng): number {
    const r = rng.uniform()
    let acc = 0
    for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (r < acc) return i }
    return probs.length - 1
  },
  sigmoid: (x: number) => 1 / (1 + Math.exp(-x)),
  argmax: (xs: number[]) => xs.reduce((bi, x, i) => (x > xs[bi] ? i : bi), 0),
  sum: (xs: number[]) => xs.reduce((a, b) => a + b, 0),
}
