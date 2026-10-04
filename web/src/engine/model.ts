import { Ops, Rng, Tensor, Num } from './tensor'

// ---------- Corpus: the "internet" the tiny model is pretrained on ----------

export const Corpus = {
  subjects: ['the cat', 'a dog', 'the old man', 'my sister', 'a small bird', 'the farmer', 'the baker', 'a child', 'the queen', 'my friend', 'the fox', 'a girl'],
  verbs: ['sees', 'likes', 'finds', 'eats', 'wants', 'holds', 'chases', 'brings', 'loves', 'makes'],
  objects: ['the ball', 'a red apple', 'the fish', 'some bread', 'a big hat', 'the moon', 'a warm cake', 'the river', 'a green leaf', 'the old boat'],
  endings: ['', '', '', ' today', ' in the garden', ' at night', ' near the house', ' again'],

  sentence(rng: Rng): string {
    const p = <T,>(xs: T[]) => xs[rng.int(xs.length)]
    return `${p(Corpus.subjects)} ${p(Corpus.verbs)} ${p(Corpus.objects)}${p(Corpus.endings)}.`
  },
  generate(sentences: number, seed: number): string {
    const rng = new Rng(seed)
    let out = ''
    for (let i = 0; i < sentences; i++) out += Corpus.sentence(rng) + (i % 4 === 3 ? '\n' : ' ')
    return out
  },
}
export const TRAINING_TEXT = Corpus.generate(3000, 7)
export const VALIDATION_TEXT = Corpus.generate(300, 99)

// ---------- Tokenizers ----------

/** One token per character. This is what the tiny model uses. */
export class CharTokenizer {
  readonly vocab: string[]
  private index = new Map<string, number>()
  constructor(text: string) {
    this.vocab = Array.from(new Set(text)).sort()
    this.vocab.forEach((c, i) => this.index.set(c, i))
  }
  get size() { return this.vocab.length }
  encode(s: string) { return Array.from(s).map((c) => this.index.get(c)).filter((x): x is number => x !== undefined) }
  decode(ids: number[]) { return ids.map((i) => this.vocab[i]).join('') }
  clean(s: string) { return Array.from(s.toLowerCase()).filter((c) => this.index.has(c)).join('') }
  display(id: number) { return CharTokenizer.show(this.vocab[id]) }
  static show(c: string) { return c === ' ' ? '␣' : c === '\n' ? '↵' : c }
}
export const tokenizer = new CharTokenizer(TRAINING_TEXT + VALIDATION_TEXT)

export interface Merge { left: string; right: string; joined: string; count: number }

/** Byte-pair encoding, trained for real on the corpus. */
export class BPE {
  merges: Merge[] = []

  static pretokenize(text: string): string[] {
    const out: string[] = []
    let cur = ''
    for (const c of text) {
      if (c === ' ' || c === '\n') {
        if (cur) out.push(cur)
        cur = c === ' ' ? '␣' : ''
        if (c === '\n') out.push('↵')
      } else cur += c
    }
    if (cur) out.push(cur)
    return out
  }

  static apply(m: { left: string; right: string }, w: string[]): string[] {
    const out: string[] = []
    for (let i = 0; i < w.length; i++) {
      if (i < w.length - 1 && w[i] === m.left && w[i + 1] === m.right) { out.push(m.left + m.right); i++ } else out.push(w[i])
    }
    return out
  }

  static train(text: string, numMerges: number): BPE {
    let words = new Map<string, { parts: string[]; n: number }>()
    for (const w of BPE.pretokenize(text)) {
      const e = words.get(w)
      if (e) e.n++; else words.set(w, { parts: Array.from(w), n: 1 })
    }
    const bpe = new BPE()
    for (let k = 0; k < numMerges; k++) {
      const pairs = new Map<string, number>()
      for (const { parts, n } of words.values()) {
        for (let i = 0; i < parts.length - 1; i++) {
          const key = parts[i] + '\u0000' + parts[i + 1]
          pairs.set(key, (pairs.get(key) ?? 0) + n)
        }
      }
      let best = '', bestN = 0
      for (const [key, n] of pairs) {
        const joined = key.replace('\u0000', '')
        if (n > bestN || (n === bestN && joined < best.replace('\u0000', ''))) { best = key; bestN = n }
      }
      if (!bestN) break
      const [left, right] = best.split('\u0000')
      bpe.merges.push({ left, right, joined: left + right, count: bestN })
      const next = new Map<string, { parts: string[]; n: number }>()
      for (const [w, e] of words) next.set(w, { parts: BPE.apply({ left, right }, e.parts), n: e.n })
      words = next
    }
    return bpe
  }

  encode(text: string, upTo = this.merges.length): string[] {
    const use = this.merges.slice(0, upTo)
    const out: string[] = []
    for (const word of BPE.pretokenize(text)) {
      let w = word === '↵' ? ['↵'] : Array.from(word)
      for (const m of use) w = BPE.apply(m, w)
      out.push(...w)
    }
    return out
  }
}

// ---------- The tiny GPT ----------

export interface GPTConfig { vocab: number; context: number; dModel: number; nHeads: number; nLayers: number }
export const defaultConfig = (): GPTConfig => ({ vocab: tokenizer.size, context: 32, dModel: 32, nHeads: 2, nLayers: 2 })

export interface ForwardTrace {
  tokens: number[]
  embeddings: number[][]
  attention: number[][][][] // [layer][head][query][key]
  scores: number[][][][]
  residualRMS: number[][]
  logits: number[][]
}

class Block {
  ln1g: Tensor; ln1b: Tensor; wq: Tensor; wk: Tensor; wv: Tensor; wo: Tensor
  ln2g: Tensor; ln2b: Tensor; w1: Tensor; b1: Tensor; w2: Tensor; b2: Tensor
  constructor(c: GPTConfig, l: number, rng: Rng) {
    const d = c.dModel, s = 1 / Math.sqrt(d), resid = s / Math.sqrt(2 * c.nLayers)
    this.ln1g = Tensor.filled(1, d, 1, `L${l}.ln1.gain`); this.ln1b = Tensor.filled(1, d, 0, `L${l}.ln1.bias`)
    this.wq = Tensor.randn(d, d, s, rng, `L${l}.attn.Wq`); this.wk = Tensor.randn(d, d, s, rng, `L${l}.attn.Wk`)
    this.wv = Tensor.randn(d, d, s, rng, `L${l}.attn.Wv`); this.wo = Tensor.randn(d, d, resid, rng, `L${l}.attn.Wo`)
    this.ln2g = Tensor.filled(1, d, 1, `L${l}.ln2.gain`); this.ln2b = Tensor.filled(1, d, 0, `L${l}.ln2.bias`)
    this.w1 = Tensor.randn(d, 4 * d, s, rng, `L${l}.mlp.W1`); this.b1 = Tensor.filled(1, 4 * d, 0, `L${l}.mlp.b1`)
    this.w2 = Tensor.randn(4 * d, d, resid / 2, rng, `L${l}.mlp.W2`); this.b2 = Tensor.filled(1, d, 0, `L${l}.mlp.b2`)
  }
  get parameters() { return [this.ln1g, this.ln1b, this.wq, this.wk, this.wv, this.wo, this.ln2g, this.ln2b, this.w1, this.b1, this.w2, this.b2] }
}

/** A real (tiny) GPT: token + position embeddings, pre-norm blocks with multi-head causal attention and a GELU MLP. */
export class TinyGPT {
  wte: Tensor; wpe: Tensor; blocks: Block[]; lnfg: Tensor; lnfb: Tensor; head: Tensor; headB: Tensor

  constructor(public config: GPTConfig, seed = 1) {
    const rng = new Rng(seed), d = config.dModel
    this.wte = Tensor.randn(config.vocab, d, 0.3, rng, 'token embeddings')
    this.wpe = Tensor.randn(config.context, d, 0.1, rng, 'position embeddings')
    this.blocks = Array.from({ length: config.nLayers }, (_, l) => new Block(config, l, rng))
    this.lnfg = Tensor.filled(1, d, 1, 'final.ln.gain'); this.lnfb = Tensor.filled(1, d, 0, 'final.ln.bias')
    this.head = Tensor.randn(d, config.vocab, 0.02, rng, 'output head')
    this.headB = Tensor.filled(1, config.vocab, 0, 'output bias')
  }

  get parameters(): Tensor[] {
    return [this.wte, this.wpe, ...this.blocks.flatMap((b) => b.parameters), this.lnfg, this.lnfb, this.head, this.headB]
  }
  get parameterCount() { return this.parameters.reduce((s, p) => s + p.count, 0) }

  copy(): TinyGPT {
    const m = new TinyGPT(this.config)
    m.parameters.forEach((p, i) => p.data.set(this.parameters[i].data))
    return m
  }

  forward(ids: number[], trace?: ForwardTrace): Tensor {
    const T = ids.length, c = this.config, hd = c.dModel / c.nHeads, scale = 1 / Math.sqrt(hd)
    let x = Ops.add(Ops.embedding(this.wte, ids), Ops.firstRows(this.wpe, T))
    if (trace) { trace.tokens = ids; trace.embeddings = Array.from({ length: T }, (_, r) => x.row(r)) }
    for (const b of this.blocks) {
      const h = Ops.layerNorm(x, b.ln1g, b.ln1b)
      const q = Ops.matmul(h, b.wq), k = Ops.matmul(h, b.wk), v = Ops.matmul(h, b.wv)
      const heads: Tensor[] = [], la: number[][][] = [], ls: number[][][] = []
      for (let hh = 0; hh < c.nHeads; hh++) {
        const qh = Ops.sliceCols(q, hh * hd, hd), kh = Ops.sliceCols(k, hh * hd, hd), vh = Ops.sliceCols(v, hh * hd, hd)
        const s = Ops.scale(Ops.matmul(qh, Ops.transpose(kh)), scale)
        const a = Ops.causalSoftmax(s)
        heads.push(Ops.matmul(a, vh))
        if (trace) { la.push(Array.from({ length: T }, (_, r) => a.row(r))); ls.push(Array.from({ length: T }, (_, r) => s.row(r))) }
      }
      x = Ops.add(x, Ops.matmul(Ops.concatCols(heads), b.wo))
      const h2 = Ops.layerNorm(x, b.ln2g, b.ln2b)
      x = Ops.add(x, Ops.addRow(Ops.matmul(Ops.gelu(Ops.addRow(Ops.matmul(h2, b.w1), b.b1)), b.w2), b.b2))
      if (trace) {
        trace.attention.push(la); trace.scores.push(ls)
        trace.residualRMS.push(Array.from({ length: T }, (_, r) => { const row = x.row(r); return Math.sqrt(row.reduce((s, v) => s + v * v, 0) / row.length) }))
      }
    }
    const logits = Ops.addRow(Ops.matmul(Ops.layerNorm(x, this.lnfg, this.lnfb), this.head), this.headB)
    if (trace) trace.logits = Array.from({ length: T }, (_, r) => logits.row(r))
    return logits
  }

  trace(ids: number[]): ForwardTrace {
    const t: ForwardTrace = { tokens: [], embeddings: [], attention: [], scores: [], residualRMS: [], logits: [] }
    this.forward(ids.slice(-this.config.context), t)
    return t
  }

  loss(batch: [number[], number[]][]): Tensor {
    return Ops.mean(batch.map(([x, y]) => Ops.crossEntropy(this.forward(x), y)))
  }

  nextTokenProbs(context: number[], temperature = 1): number[] {
    const ids = context.slice(-this.config.context)
    if (!ids.length) return new Array(this.config.vocab).fill(1 / this.config.vocab)
    return Num.softmax(this.forward(ids).row(ids.length - 1), temperature)
  }

  generate(prompt: number[], count: number, temperature: number, rng: Rng): number[] {
    const ids = [...prompt]
    for (let i = 0; i < count; i++) ids.push(Num.sample(this.nextTokenProbs(ids, temperature), rng))
    return ids.slice(prompt.length)
  }
}

// ---------- Optimizers ----------

export type OptimizerKind = 'SGD' | 'Momentum' | 'AdamW'
export const optimizerKinds: OptimizerKind[] = ['SGD', 'Momentum', 'AdamW']

export class Optimizer {
  beta1 = 0.9; beta2 = 0.95; eps = 1e-8; weightDecay = 0.01
  private m = new Map<Tensor, Float64Array>()
  private v = new Map<Tensor, Float64Array>()
  t = 0
  constructor(public kind: OptimizerKind) {}

  step(params: Tensor[], lr: number) {
    this.t++
    for (const p of params) {
      if (this.kind === 'SGD') { for (let i = 0; i < p.count; i++) p.data[i] -= lr * p.grad[i]; continue }
      let m = this.m.get(p); if (!m) { m = new Float64Array(p.count); this.m.set(p, m) }
      if (this.kind === 'Momentum') {
        for (let i = 0; i < p.count; i++) { m[i] = this.beta1 * m[i] + p.grad[i]; p.data[i] -= lr * m[i] }
        continue
      }
      let v = this.v.get(p); if (!v) { v = new Float64Array(p.count); this.v.set(p, v) }
      const bc1 = 1 - Math.pow(this.beta1, this.t), bc2 = 1 - Math.pow(this.beta2, this.t)
      const decay = p.rows > 1 ? this.weightDecay : 0
      for (let i = 0; i < p.count; i++) {
        const g = p.grad[i]
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g
        p.data[i] -= lr * ((m[i] / bc1) / (Math.sqrt(v[i] / bc2) + this.eps) + decay * p.data[i])
      }
    }
  }
}

export function clipGradients(params: Tensor[], maxNorm: number): number {
  let sq = 0
  for (const p of params) for (const g of p.grad) sq += g * g
  const norm = Math.sqrt(sq)
  if (norm > maxNorm) { const s = maxNorm / norm; for (const p of params) for (let i = 0; i < p.count; i++) p.grad[i] *= s }
  return norm
}

/** Linear warm-up, then cosine decay to 10% of the peak. */
export function scheduledLR(peak: number, warmup: number, total: number, step: number, floor = 0.1): number {
  if (step < warmup) return (peak * (step + 1)) / Math.max(warmup, 1)
  const progress = Math.min(1, (step - warmup) / Math.max(total - warmup, 1))
  return peak * (floor + (1 - floor) * 0.5 * (1 + Math.cos(Math.PI * progress)))
}
