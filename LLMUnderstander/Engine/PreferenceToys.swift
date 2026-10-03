import Foundation

/// A toy "language model" for post-training: for one prompt it can give one of
/// eight whole answers, and its policy is a softmax over one logit per answer.
/// Every update below is the real math (SFT cross-entropy, Bradley–Terry reward
/// model, KL-regularised policy optimisation, DPO) — just on 8 options instead of
/// every possible string.
enum PostTraining {
    struct Answer: Identifiable {
        let id: Int
        let short: String
        let text: String
        /// What the reward model can "see": helpfulness, politeness, length (0…1).
        let features: [Double]
        /// Extra penalty humans apply that features can't capture (e.g. unsafe advice).
        let hiddenPenalty: Double
    }

    static let prompt = "how do i boil an egg?"

    static let answers: [Answer] = [
        Answer(id: 0, short: "Keeps listing questions",
               text: "how do i fry an egg? how do i poach an egg? how do i…",
               features: [0.0, 0.0, 0.35], hiddenPenalty: 0),
        Answer(id: 1, short: "Short & correct",
               text: "Put the egg in boiling water for 9 minutes, then cool it in cold water.",
               features: [1.0, 0.3, 0.30], hiddenPenalty: 0),
        Answer(id: 2, short: "Friendly & correct",
               text: "Happy to help! Lower the egg into boiling water, cook 9 minutes for hard-boiled (6 for soft), then cool it in cold water.",
               features: [0.95, 1.0, 0.45], hiddenPenalty: 0),
        Answer(id: 3, short: "Off-topic fact",
               text: "Eggs are laid by hens.",
               features: [0.1, 0.2, 0.15], hiddenPenalty: 0),
        Answer(id: 4, short: "Needless refusal",
               text: "Sorry, I can't help with cooking.",
               features: [0.0, 0.6, 0.15], hiddenPenalty: 0),
        Answer(id: 5, short: "Long rambling essay",
               text: "What a wonderful question! Eggs have fascinated cooks for millennia… (400 more words on the history of eggs) …and that is why, eventually, you boil it.",
               features: [0.65, 1.0, 1.0], hiddenPenalty: 0),
        Answer(id: 6, short: "Wrong & unsafe",
               text: "Microwave it in its shell for 5 minutes.",
               features: [0.3, 0.3, 0.20], hiddenPenalty: 2.5),
        Answer(id: 7, short: "Curt",
               text: "Boil it.",
               features: [0.4, 0.0, 0.05], hiddenPenalty: 0),
    ]

    /// How good each answer really is to a careful person (hidden from the model).
    static func trueQuality(_ a: Answer) -> Double {
        let h = a.features[0], p = a.features[1], l = a.features[2]
        return 3 * h + 0.8 * p + 2 * min(l, 0.45) - 6 * max(l - 0.55, 0) - a.hiddenPenalty
    }
    static let quality = answers.map(trueQuality)

    /// Pretrained (base) model: it just continues text, so "keep listing questions" is likely.
    static let baseLogits: [Double] = [1.6, -0.6, -0.9, 1.0, -0.2, -1.4, 0.3, 0.2]
    /// Human-written demonstrations used for SFT.
    static let demonstrations = [1, 2, 2, 1, 2]
    /// The SFT model that preference tuning starts from (and is held close to by the KL term).
    static let sftLogits: [Double] = [0.5, 1.2, 1.0, 0.0, 0.3, -0.5, 0.0, 0.3]

    // MARK: SFT

    /// One gradient step of cross-entropy on the demonstrations. d loss / d logit_j = p_j − 1[j = demo].
    static func sftStep(_ logits: [Double], lr: Double) -> [Double] {
        let p = Num.softmax(logits)
        var g = [Double](repeating: 0, count: logits.count)
        for d in demonstrations { for j in 0..<g.count { g[j] += (p[j] - (j == d ? 1 : 0)) / Double(demonstrations.count) } }
        return zip(logits, g).map { $0 - lr * $1 }
    }

    static func sftLoss(_ logits: [Double]) -> Double {
        let p = Num.softmax(logits)
        return demonstrations.reduce(0) { $0 - log(p[$1]) } / Double(demonstrations.count)
    }

    // MARK: Human preferences + reward model

    struct Comparison: Identifiable {
        let id: Int
        let chosen: Int
        let rejected: Int
    }

    /// Labelers compare pairs of answers. Like real labelers they slightly favour longer
    /// answers — a bias the reward model will faithfully learn. Answer 5 never shows up
    /// in the labelled data.
    static func collectComparisons(count: Int = 240, seed: UInt64 = 3) -> [Comparison] {
        var rng = SplitMix64(seed: seed)
        let seen = [0, 1, 2, 3, 4, 6, 7]
        return (0..<count).map { i in
            let a = seen[rng.int(seen.count)]
            var b = seen[rng.int(seen.count)]
            while b == a { b = seen[rng.int(seen.count)] }
            let la = quality[a] + 3 * answers[a].features[2]
            let lb = quality[b] + 3 * answers[b].features[2]
            let pa = Num.sigmoid(la - lb)
            return rng.uniform() < pa ? Comparison(id: i, chosen: a, rejected: b) : Comparison(id: i, chosen: b, rejected: a)
        }
    }

    /// Linear reward model r(answer) = w · features, trained with the Bradley–Terry loss
    /// −log σ(r(chosen) − r(rejected)).
    struct RewardModel {
        var w: [Double] = [0, 0, 0]
        func reward(_ a: Answer) -> Double { zip(w, a.features).reduce(0) { $0 + $1.0 * $1.1 } }
        var rewards: [Double] { answers.map(reward) }

        mutating func step(_ data: [Comparison], lr: Double) {
            var g = [0.0, 0.0, 0.0]
            for c in data {
                let d = zip(answers[c.chosen].features, answers[c.rejected].features).map { $0 - $1 }
                let s = Num.sigmoid(zip(w, d).reduce(0) { $0 + $1.0 * $1.1 })
                for k in 0..<3 { g[k] += (1 - s) * d[k] }
            }
            for k in 0..<3 { w[k] += lr * g[k] / Double(data.count) }
        }

        func loss(_ data: [Comparison]) -> Double {
            data.reduce(0) { acc, c in acc - log(Num.sigmoid(reward(answers[c.chosen]) - reward(answers[c.rejected]))) } / Double(data.count)
        }

        func accuracy(_ data: [Comparison]) -> Double {
            Double(data.filter { reward(answers[$0.chosen]) > reward(answers[$0.rejected]) }.count) / Double(data.count)
        }
    }

    static func trainedRewardModel(steps: Int = 400) -> RewardModel {
        let data = collectComparisons()
        var rm = RewardModel()
        for _ in 0..<steps { rm.step(data, lr: 0.5) }
        return rm
    }

    // MARK: RLHF (KL-regularised policy optimisation)

    /// Objective: E_π[r] − β·KL(π ‖ π_sft). One natural-policy-gradient step on the logits.
    /// Its fixed point is π ∝ π_sft · exp(r / β), the known optimum of this objective.
    static func rlhfStep(_ logits: [Double], rewards: [Double], beta: Double, lr: Double = 0.3) -> [Double] {
        (0..<logits.count).map { i in logits[i] + lr * (rewards[i] - beta * (logits[i] - sftLogits[i])) }
    }

    static func kl(_ p: [Double], _ q: [Double]) -> Double {
        zip(p, q).reduce(0) { $0 + ($1.0 > 0 ? $1.0 * log($1.0 / $1.1) : 0) }
    }

    static func expected(_ p: [Double], _ values: [Double]) -> Double {
        zip(p, values).reduce(0) { $0 + $1.0 * $1.1 }
    }

    // MARK: DPO

    /// Direct Preference Optimisation: no reward model, no sampling — a classification
    /// loss on the comparisons themselves:
    /// −log σ(β[(log π(y_w) − log π_sft(y_w)) − (log π(y_l) − log π_sft(y_l))]).
    static func dpoStep(_ logits: [Double], data: [Comparison], beta: Double, lr: Double) -> [Double] {
        let lp = logProbs(logits), lref = logProbs(sftLogits)
        var g = [Double](repeating: 0, count: logits.count)
        for c in data {
            let z = beta * ((lp[c.chosen] - lref[c.chosen]) - (lp[c.rejected] - lref[c.rejected]))
            let coef = 1 - Num.sigmoid(z)
            // ∂z/∂logit_j = β(1[j=w] − 1[j=l]); the softmax normaliser cancels.
            g[c.chosen] += coef * beta
            g[c.rejected] -= coef * beta
        }
        return (0..<logits.count).map { logits[$0] + lr * g[$0] / Double(data.count) }
    }

    static func dpoLoss(_ logits: [Double], data: [Comparison], beta: Double) -> Double {
        let lp = logProbs(logits), lref = logProbs(sftLogits)
        return data.reduce(0) { acc, c in
            acc - log(Num.sigmoid(beta * ((lp[c.chosen] - lref[c.chosen]) - (lp[c.rejected] - lref[c.rejected]))))
        } / Double(data.count)
    }

    static func logProbs(_ logits: [Double]) -> [Double] { Num.softmax(logits).map { log($0) } }
}
