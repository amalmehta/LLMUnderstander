import Foundation

enum OptimizerKind: String, CaseIterable, Identifiable, Codable {
    case sgd = "SGD"
    case momentum = "Momentum"
    case adam = "AdamW"
    var id: String { rawValue }
}

/// Updates parameters from their gradients. Same three algorithms used in the
/// optimizer chapter's 2-D race, here applied to every weight of the model.
final class Optimizer {
    let kind: OptimizerKind
    var beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weightDecay = 0.01
    private var m: [ObjectIdentifier: [Double]] = [:]
    private var v: [ObjectIdentifier: [Double]] = [:]
    private(set) var t = 0

    init(kind: OptimizerKind) { self.kind = kind }

    /// Momentum and second-moment estimates, in the same order as `params`, so a saved run resumes exactly.
    struct State: Codable, Equatable {
        var t: Int
        var m: [[Double]?]
        var v: [[Double]?]
    }

    func state(for params: [Tensor]) -> State {
        State(t: t, m: params.map { m[ObjectIdentifier($0)] }, v: params.map { v[ObjectIdentifier($0)] })
    }

    /// Returns false (and changes nothing) if the saved state doesn't fit these parameters.
    @discardableResult
    func restore(_ s: State, for params: [Tensor]) -> Bool {
        guard s.m.count == params.count, s.v.count == params.count,
              zip(s.m, params).allSatisfy({ $0.0 == nil || $0.0!.count == $0.1.count }),
              zip(s.v, params).allSatisfy({ $0.0 == nil || $0.0!.count == $0.1.count }) else { return false }
        t = s.t
        m = [:]; v = [:]
        for (i, p) in params.enumerated() {
            if let mv = s.m[i] { m[ObjectIdentifier(p)] = mv }
            if let vv = s.v[i] { v[ObjectIdentifier(p)] = vv }
        }
        return true
    }

    func step(_ params: [Tensor], lr: Double) {
        t += 1
        for p in params {
            let id = ObjectIdentifier(p)
            switch kind {
            case .sgd:
                for i in 0..<p.count { p.data[i] -= lr * p.grad[i] }
            case .momentum:
                var mv = m[id] ?? Array(repeating: 0, count: p.count)
                for i in 0..<p.count {
                    mv[i] = beta1 * mv[i] + p.grad[i]
                    p.data[i] -= lr * mv[i]
                }
                m[id] = mv
            case .adam:
                var mv = m[id] ?? Array(repeating: 0, count: p.count)
                var vv = v[id] ?? Array(repeating: 0, count: p.count)
                let bc1 = 1 - pow(beta1, Double(t)), bc2 = 1 - pow(beta2, Double(t))
                // decay matrices only, not gains/biases (standard practice)
                let decay = p.rows > 1 ? weightDecay : 0
                for i in 0..<p.count {
                    let g = p.grad[i]
                    mv[i] = beta1 * mv[i] + (1 - beta1) * g
                    vv[i] = beta2 * vv[i] + (1 - beta2) * g * g
                    let mh = mv[i] / bc1, vh = vv[i] / bc2
                    p.data[i] -= lr * (mh / (vh.squareRoot() + eps) + decay * p.data[i])
                }
                m[id] = mv; v[id] = vv
            }
        }
    }
}

/// Global gradient norm, then scale all gradients down if it exceeds `maxNorm`.
@discardableResult
func clipGradients(_ params: [Tensor], maxNorm: Double) -> Double {
    var sq = 0.0
    for p in params { for g in p.grad { sq += g * g } }
    let norm = sq.squareRoot()
    if norm > maxNorm {
        let s = maxNorm / norm
        for p in params { for i in 0..<p.count { p.grad[i] *= s } }
    }
    return norm
}

/// Linear warm-up, then cosine decay to 10% of the peak — the schedule most LLMs use.
struct LRSchedule {
    var peak: Double
    var warmup: Int
    var total: Int
    var floorFraction = 0.1

    func lr(at step: Int) -> Double {
        if step < warmup { return peak * Double(step + 1) / Double(max(warmup, 1)) }
        let progress = min(1, Double(step - warmup) / Double(max(total - warmup, 1)))
        let cosine = 0.5 * (1 + cos(.pi * progress))
        return peak * (floorFraction + (1 - floorFraction) * cosine)
    }
}
