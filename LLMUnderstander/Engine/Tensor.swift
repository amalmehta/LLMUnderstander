import Foundation
import Accelerate

/// A tiny 2-D tensor with reverse-mode automatic differentiation.
///
/// Every operation records its inputs (`parents`) and a closure that pushes the
/// output gradient back into those inputs. Calling `backward()` on a scalar walks
/// the graph in reverse topological order — exactly what PyTorch's autograd does,
/// just small enough to read in one sitting.
final class Tensor {
    let rows: Int
    let cols: Int
    var data: [Double]
    var grad: [Double]
    var parents: [Tensor] = []
    var backwardFn: ((Tensor) -> Void)?
    var label: String

    init(rows: Int, cols: Int, data: [Double]? = nil, label: String = "") {
        self.rows = rows
        self.cols = cols
        self.data = data ?? Array(repeating: 0, count: rows * cols)
        self.grad = Array(repeating: 0, count: rows * cols)
        self.label = label
        precondition(self.data.count == rows * cols, "shape mismatch")
    }

    static func randn(_ rows: Int, _ cols: Int, std: Double, rng: inout SplitMix64, label: String = "") -> Tensor {
        let t = Tensor(rows: rows, cols: cols, label: label)
        for i in 0..<t.data.count { t.data[i] = rng.normal() * std }
        return t
    }

    static func filled(_ rows: Int, _ cols: Int, _ value: Double, label: String = "") -> Tensor {
        Tensor(rows: rows, cols: cols, data: Array(repeating: value, count: rows * cols), label: label)
    }

    var count: Int { rows * cols }
    var item: Double { data[0] }

    subscript(r: Int, c: Int) -> Double {
        get { data[r * cols + c] }
        set { data[r * cols + c] = newValue }
    }

    func row(_ r: Int) -> [Double] { Array(data[(r * cols)..<((r + 1) * cols)]) }

    func zeroGrad() {
        for i in 0..<grad.count { grad[i] = 0 }
    }

    func copy() -> Tensor {
        Tensor(rows: rows, cols: cols, data: data, label: label)
    }

    /// Back-propagate from this (scalar) tensor through the whole graph.
    func backward() {
        var order: [Tensor] = []
        var seen = Set<ObjectIdentifier>()
        func visit(_ t: Tensor) {
            let id = ObjectIdentifier(t)
            if seen.contains(id) { return }
            seen.insert(id)
            for p in t.parents { visit(p) }
            order.append(t)
        }
        visit(self)
        for i in 0..<grad.count { grad[i] = 1 }
        for t in order.reversed() { t.backwardFn?(t) }
    }
}

// MARK: - Operations

enum Ops {
    /// (R×K) · (K×C) → R×C, using Accelerate's BLAS for speed.
    static func matmul(_ a: Tensor, _ b: Tensor) -> Tensor {
        precondition(a.cols == b.rows, "matmul shape mismatch \(a.rows)x\(a.cols) · \(b.rows)x\(b.cols)")
        let R = a.rows, K = a.cols, C = b.cols
        let out = Tensor(rows: R, cols: C)
        BLAS.gemm(false, false, R, C, K, a.data, b.data, &out.data, beta: 0)
        out.parents = [a, b]
        out.backwardFn = { out in
            BLAS.gemm(false, true, R, K, C, out.grad, b.data, &a.grad, beta: 1)   // dA += dOut · Bᵀ
            BLAS.gemm(true, false, K, C, R, a.data, out.grad, &b.grad, beta: 1)   // dB += Aᵀ · dOut
        }
        return out
    }

    static func transpose(_ a: Tensor) -> Tensor {
        let out = Tensor(rows: a.cols, cols: a.rows)
        for i in 0..<a.rows { for j in 0..<a.cols { out.data[j * a.rows + i] = a.data[i * a.cols + j] } }
        out.parents = [a]
        out.backwardFn = { out in
            for i in 0..<a.rows { for j in 0..<a.cols { a.grad[i * a.cols + j] += out.grad[j * a.rows + i] } }
        }
        return out
    }

    static func add(_ a: Tensor, _ b: Tensor) -> Tensor {
        precondition(a.rows == b.rows && a.cols == b.cols)
        let out = Tensor(rows: a.rows, cols: a.cols)
        for i in 0..<a.count { out.data[i] = a.data[i] + b.data[i] }
        out.parents = [a, b]
        out.backwardFn = { out in
            for i in 0..<out.count {
                a.grad[i] += out.grad[i]
                b.grad[i] += out.grad[i]
            }
        }
        return out
    }

    /// Adds a 1×C row vector to every row of a.
    static func addRow(_ a: Tensor, _ bias: Tensor) -> Tensor {
        precondition(bias.rows == 1 && bias.cols == a.cols)
        let out = Tensor(rows: a.rows, cols: a.cols)
        for i in 0..<a.rows { for j in 0..<a.cols { out.data[i * a.cols + j] = a.data[i * a.cols + j] + bias.data[j] } }
        out.parents = [a, bias]
        out.backwardFn = { out in
            for i in 0..<a.rows {
                for j in 0..<a.cols {
                    let g = out.grad[i * a.cols + j]
                    a.grad[i * a.cols + j] += g
                    bias.grad[j] += g
                }
            }
        }
        return out
    }

    static func scale(_ a: Tensor, _ s: Double) -> Tensor {
        let out = Tensor(rows: a.rows, cols: a.cols, data: a.data.map { $0 * s })
        out.parents = [a]
        out.backwardFn = { out in
            for i in 0..<a.count { a.grad[i] += out.grad[i] * s }
        }
        return out
    }

    /// GELU, tanh approximation (as used in GPT-2).
    static func gelu(_ a: Tensor) -> Tensor {
        let k = (2.0 / Double.pi).squareRoot()
        let out = Tensor(rows: a.rows, cols: a.cols)
        var tanhs = [Double](repeating: 0, count: a.count)
        for i in 0..<a.count {
            let x = a.data[i]
            let t = tanh(k * (x + 0.044715 * x * x * x))
            tanhs[i] = t
            out.data[i] = 0.5 * x * (1 + t)
        }
        out.parents = [a]
        out.backwardFn = { out in
            for i in 0..<a.count {
                let x = a.data[i], t = tanhs[i]
                let d = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * 0.044715 * x * x)
                a.grad[i] += out.grad[i] * d
            }
        }
        return out
    }

    /// Row-wise softmax where row i may only look at columns 0...i (the causal mask).
    static func causalSoftmax(_ a: Tensor) -> Tensor {
        precondition(a.rows == a.cols)
        let n = a.rows
        let out = Tensor(rows: n, cols: n)
        for i in 0..<n {
            var m = -Double.infinity
            for j in 0...i { m = max(m, a.data[i * n + j]) }
            var s = 0.0
            for j in 0...i { let e = exp(a.data[i * n + j] - m); out.data[i * n + j] = e; s += e }
            for j in 0...i { out.data[i * n + j] /= s }
        }
        out.parents = [a]
        out.backwardFn = { out in
            for i in 0..<n {
                var dot = 0.0
                for j in 0...i { dot += out.data[i * n + j] * out.grad[i * n + j] }
                for j in 0...i {
                    let y = out.data[i * n + j]
                    a.grad[i * n + j] += y * (out.grad[i * n + j] - dot)
                }
            }
        }
        return out
    }

    /// Layer normalisation over each row, with learnable gain and bias (1×C each).
    static func layerNorm(_ x: Tensor, _ gamma: Tensor, _ beta: Tensor, eps: Double = 1e-5) -> Tensor {
        let R = x.rows, C = x.cols
        let out = Tensor(rows: R, cols: C)
        var xhat = [Double](repeating: 0, count: R * C)
        var rstd = [Double](repeating: 0, count: R)
        for i in 0..<R {
            var mean = 0.0
            for j in 0..<C { mean += x.data[i * C + j] }
            mean /= Double(C)
            var v = 0.0
            for j in 0..<C { let d = x.data[i * C + j] - mean; v += d * d }
            v /= Double(C)
            let r = 1 / (v + eps).squareRoot()
            rstd[i] = r
            for j in 0..<C {
                let h = (x.data[i * C + j] - mean) * r
                xhat[i * C + j] = h
                out.data[i * C + j] = h * gamma.data[j] + beta.data[j]
            }
        }
        out.parents = [x, gamma, beta]
        out.backwardFn = { out in
            for i in 0..<R {
                var meanD = 0.0, meanDX = 0.0
                for j in 0..<C {
                    let g = out.grad[i * C + j]
                    gamma.grad[j] += g * xhat[i * C + j]
                    beta.grad[j] += g
                    let dxh = g * gamma.data[j]
                    meanD += dxh
                    meanDX += dxh * xhat[i * C + j]
                }
                meanD /= Double(C); meanDX /= Double(C)
                for j in 0..<C {
                    let dxh = out.grad[i * C + j] * gamma.data[j]
                    x.grad[i * C + j] += rstd[i] * (dxh - meanD - xhat[i * C + j] * meanDX)
                }
            }
        }
        return out
    }

    /// Look up one row of `table` per token id.
    static func embedding(_ table: Tensor, _ ids: [Int]) -> Tensor {
        let C = table.cols
        let out = Tensor(rows: ids.count, cols: C)
        for (i, id) in ids.enumerated() { for j in 0..<C { out.data[i * C + j] = table.data[id * C + j] } }
        out.parents = [table]
        out.backwardFn = { out in
            for (i, id) in ids.enumerated() { for j in 0..<C { table.grad[id * C + j] += out.grad[i * C + j] } }
        }
        return out
    }

    /// First `n` rows of a tensor (used for position embeddings).
    static func firstRows(_ a: Tensor, _ n: Int) -> Tensor {
        embedding(a, Array(0..<n))
    }

    static func sliceCols(_ a: Tensor, _ start: Int, _ count: Int) -> Tensor {
        let out = Tensor(rows: a.rows, cols: count)
        for i in 0..<a.rows { for j in 0..<count { out.data[i * count + j] = a.data[i * a.cols + start + j] } }
        out.parents = [a]
        out.backwardFn = { out in
            for i in 0..<a.rows { for j in 0..<count { a.grad[i * a.cols + start + j] += out.grad[i * count + j] } }
        }
        return out
    }

    static func concatCols(_ parts: [Tensor]) -> Tensor {
        let R = parts[0].rows
        let C = parts.reduce(0) { $0 + $1.cols }
        let out = Tensor(rows: R, cols: C)
        var off = 0
        var offsets: [Int] = []
        for p in parts {
            offsets.append(off)
            for i in 0..<R { for j in 0..<p.cols { out.data[i * C + off + j] = p.data[i * p.cols + j] } }
            off += p.cols
        }
        out.parents = parts
        out.backwardFn = { out in
            for (k, p) in parts.enumerated() {
                let o = offsets[k]
                for i in 0..<R { for j in 0..<p.cols { p.grad[i * p.cols + j] += out.grad[i * C + o + j] } }
            }
        }
        return out
    }

    /// Mean cross-entropy between row-wise softmax(logits) and integer targets.
    /// `weights` (optional) lets some positions count and others not — the SFT loss mask.
    static func crossEntropy(_ logits: Tensor, _ targets: [Int], weights: [Double]? = nil) -> Tensor {
        let R = logits.rows, V = logits.cols
        precondition(targets.count == R)
        let w = weights ?? Array(repeating: 1, count: R)
        let total = max(w.reduce(0, +), 1e-12)
        var probs = [Double](repeating: 0, count: R * V)
        var loss = 0.0
        for i in 0..<R {
            var m = -Double.infinity
            for j in 0..<V { m = max(m, logits.data[i * V + j]) }
            var s = 0.0
            for j in 0..<V { let e = exp(logits.data[i * V + j] - m); probs[i * V + j] = e; s += e }
            for j in 0..<V { probs[i * V + j] /= s }
            loss -= w[i] * log(max(probs[i * V + targets[i]], 1e-300))
        }
        let out = Tensor(rows: 1, cols: 1, data: [loss / total])
        out.parents = [logits]
        out.backwardFn = { out in
            let g = out.grad[0] / total
            for i in 0..<R where w[i] != 0 {
                for j in 0..<V {
                    let y = j == targets[i] ? 1.0 : 0.0
                    logits.grad[i * V + j] += g * w[i] * (probs[i * V + j] - y)
                }
            }
        }
        return out
    }

    /// Average of several 1×1 tensors.
    static func mean(_ xs: [Tensor]) -> Tensor {
        let n = Double(xs.count)
        let out = Tensor(rows: 1, cols: 1, data: [xs.reduce(0) { $0 + $1.data[0] } / n])
        out.parents = xs
        out.backwardFn = { out in
            for x in xs { x.grad[0] += out.grad[0] / n }
        }
        return out
    }
}

// MARK: - Random numbers

/// Small, fast, seedable RNG so every run of the app is reproducible.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func uniform() -> Double { Double(next() >> 11) / Double(1 << 53) }
    mutating func normal() -> Double {
        let u1 = max(uniform(), 1e-12), u2 = uniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
    }
    mutating func int(_ n: Int) -> Int { Int(next() % UInt64(n)) }
}

// MARK: - Small numeric helpers

enum Num {
    static func softmax(_ xs: [Double], temperature: Double = 1) -> [Double] {
        let t = max(temperature, 1e-6)
        let m = xs.max() ?? 0
        let e = xs.map { exp(($0 - m) / t) }
        let s = e.reduce(0, +)
        return e.map { $0 / s }
    }

    static func sample(_ probs: [Double], rng: inout SplitMix64) -> Int {
        let r = rng.uniform()
        var acc = 0.0
        for (i, p) in probs.enumerated() {
            acc += p
            if r < acc { return i }
        }
        return probs.count - 1
    }

    static func sigmoid(_ x: Double) -> Double { 1 / (1 + exp(-x)) }
}

/// Thin wrapper over cblas_dgemm: C = op(A)·op(B) + beta·C (row-major).
enum BLAS {
    static func gemm(_ transA: Bool, _ transB: Bool, _ m: Int, _ n: Int, _ k: Int,
                     _ a: [Double], _ b: [Double], _ c: inout [Double], beta: Double) {
        let lda = transA ? m : k
        let ldb = transB ? k : n
        cblas_dgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
                    Int32(m), Int32(n), Int32(k), 1, a, Int32(lda), b, Int32(ldb), beta, &c, Int32(n))
    }
}
