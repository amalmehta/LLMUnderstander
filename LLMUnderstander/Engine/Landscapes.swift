import Foundation

/// 2-D loss surfaces for racing optimizers. A real model has millions of
/// dimensions, but the behaviours (zig-zagging, overshooting, momentum) look the same.
enum Landscape: String, CaseIterable, Identifiable {
    case ravine = "Narrow ravine"
    case rosenbrock = "Banana valley"
    case bumpy = "Bumpy bowl"
    var id: String { rawValue }

    var range: (x: ClosedRange<Double>, y: ClosedRange<Double>) {
        switch self {
        case .ravine: (-3...3, -2...2)
        case .rosenbrock: (-2...2, -1...3)
        case .bumpy: (-3...3, -3...3)
        }
    }

    var start: (Double, Double) {
        switch self {
        case .ravine: (-2.6, 1.5)
        case .rosenbrock: (-1.6, 2.6)
        case .bumpy: (-2.5, 2.4)
        }
    }

    func value(_ x: Double, _ y: Double) -> Double {
        switch self {
        case .ravine: return 0.02 * x * x + 2.5 * y * y
        case .rosenbrock: return 0.05 * ((1 - x) * (1 - x) + 10 * (y - x * x) * (y - x * x))
        case .bumpy: return 0.15 * (x * x + y * y) + 0.6 * sin(2 * x) * sin(2 * y) + 0.6
        }
    }

    func gradient(_ x: Double, _ y: Double) -> (Double, Double) {
        let h = 1e-5
        return ((value(x + h, y) - value(x - h, y)) / (2 * h), (value(x, y + h) - value(x, y - h)) / (2 * h))
    }

    /// Path of `steps` optimizer steps from the start point.
    func path(_ kind: OptimizerKind, lr: Double, steps: Int) -> [(Double, Double)] {
        let p = Tensor(rows: 1, cols: 2, data: [start.0, start.1])
        let opt = Optimizer(kind: kind)
        opt.weightDecay = 0
        opt.beta2 = 0.999
        var out = [(p.data[0], p.data[1])]
        // Adam's step size is ~lr regardless of gradient scale, so it gets its own scale.
        let eff = kind == .adam ? lr * 0.15 : lr
        for _ in 0..<steps {
            let g = gradient(p.data[0], p.data[1])
            p.grad = [g.0, g.1]
            opt.step([p], lr: eff)
            let x = min(max(p.data[0], -50), 50), y = min(max(p.data[1], -50), 50)
            p.data = [x, y]
            out.append((x, y))
        }
        return out
    }
}

/// Chinchilla-style scaling law: loss(N params, D tokens) = E + A/N^α + B/D^β,
/// with training compute C ≈ 6·N·D FLOPs. Constants from Besiroglu et al. (2024),
/// the Epoch AI replication of Hoffmann et al. (2022) "Chinchilla".
enum ScalingLaw {
    static let E = 1.82, A = 482.01, B = 2085.43, alpha = 0.3478, beta = 0.3658

    static func loss(params n: Double, tokens d: Double) -> Double {
        E + A / pow(n, alpha) + B / pow(d, beta)
    }

    static func tokens(forCompute c: Double, params n: Double) -> Double { c / (6 * n) }

    /// Sweep model sizes for a fixed compute budget; returns (log10 N, loss).
    static func sweep(compute c: Double) -> [(Double, Double)] {
        stride(from: 6.0, through: 13.0, by: 0.05).map { lg in
            let n = pow(10, lg)
            return (lg, loss(params: n, tokens: tokens(forCompute: c, params: n)))
        }
    }

    static func optimum(compute c: Double) -> (params: Double, tokens: Double, loss: Double) {
        let best = sweep(compute: c).min { $0.1 < $1.1 }!
        let n = pow(10, best.0)
        return (n, tokens(forCompute: c, params: n), best.1)
    }
}

/// Rough parameter count of a GPT-style decoder: 12·L·d² for the blocks + V·d embeddings.
enum ParamCount {
    static func estimate(layers: Int, dModel: Int, vocab: Int) -> Double {
        12 * Double(layers) * Double(dModel * dModel) + Double(vocab * dModel)
    }
}

enum Format {
    static func big(_ x: Double) -> String {
        let a = abs(x)
        switch a {
        case 1e12...: return String(format: "%.1fT", x / 1e12)
        case 1e9...: return String(format: "%.1fB", x / 1e9)
        case 1e6...: return String(format: "%.1fM", x / 1e6)
        case 1e3...: return String(format: "%.1fK", x / 1e3)
        default: return String(format: "%.0f", x)
        }
    }

    static func sci(_ x: Double) -> String {
        let e = Int(floor(log10(x)))
        let m = x / pow(10, Double(e))
        return String(format: "%.1f×10^%d", m, e)
    }
}
