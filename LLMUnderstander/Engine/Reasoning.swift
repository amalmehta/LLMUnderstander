import Foundation

/// A toy reasoning model trained with GRPO (group relative policy optimisation).
///
/// Each problem is a sum of 2–6 one-digit numbers, so it needs 1–5 additions.
/// Before answering, the policy chooses how many "thinking steps" to write out:
/// each step does one addition correctly. With fewer steps than additions, the
/// rest is done "in its head" and each such addition is right only 35% of the time.
/// The checker is verifiable: it compares the final number with the true sum.
///
/// The policy has one softmax over thinking lengths 0…6 per difficulty, and starts
/// out preferring to answer straight away, like a model that was never rewarded for
/// working things out. GRPO samples a group of answers to one problem, scores each
/// against the group's average, and pushes up the lengths that did better than
/// their group.
struct ReasoningToy {
    static let maxThinking = 6
    static let difficulties = 1...5          // additions needed
    static let headAccuracy = 0.35           // chance an un-written-out addition is right
    static let stepAccuracy = 0.98           // chance a written-out addition is right

    /// logits[difficulty - 1][length]
    var logits: [[Double]]
    var lengthCost = 0.0                     // reward subtracted per thinking step
    var groupSize = 8
    var learningRate = 0.5

    init() {
        // Prefers answering at once: a base model has rarely seen worked-out sums.
        let start = (0...Self.maxThinking).map { -0.6 * Double($0) }
        logits = Array(repeating: start, count: Self.difficulties.count)
    }

    func policy(difficulty d: Int) -> [Double] { Num.softmax(logits[d - 1]) }

    /// Probability that thinking for `k` steps gets a problem needing `d` additions right.
    static func pCorrect(thinking k: Int, difficulty d: Int) -> Double {
        let written = min(k, d), inHead = d - written
        return pow(stepAccuracy, Double(written)) * pow(headAccuracy, Double(inHead))
    }

    /// Accuracy and average thinking length over all difficulties, under the current policy.
    var expectedAccuracy: Double {
        Self.difficulties.map { d in
            zip(policy(difficulty: d), 0...Self.maxThinking).map { $0 * Self.pCorrect(thinking: $1, difficulty: d) }.reduce(0, +)
        }.reduce(0, +) / Double(Self.difficulties.count)
    }

    var expectedLength: Double {
        Self.difficulties.map { d in
            zip(policy(difficulty: d), 0...Self.maxThinking).map { $0 * Double($1) }.reduce(0, +)
        }.reduce(0, +) / Double(Self.difficulties.count)
    }

    struct Answer: Hashable {
        let thinking: Int
        let correct: Bool
        let reward: Double
        let advantage: Double
    }

    struct Group {
        let problem: [Int]
        let answers: [Answer]
        /// True when every answer got the same reward: no learning signal at all.
        var silent: Bool { Set(answers.map(\.reward)).count == 1 }
    }

    static func makeProblem(_ rng: inout SplitMix64) -> [Int] {
        let n = 2 + rng.int(difficulties.count)            // 2…6 numbers
        return (0..<n).map { _ in 1 + rng.int(9) }
    }

    /// One GRPO update on one problem. Returns the group so the page can show it.
    @discardableResult
    mutating func step(_ rng: inout SplitMix64) -> Group {
        let problem = Self.makeProblem(&rng)
        let d = problem.count - 1
        let pi = policy(difficulty: d)
        var lengths: [Int] = [], rewards: [Double] = [], correct: [Bool] = []
        for _ in 0..<groupSize {
            let k = Num.sample(pi, rng: &rng)
            let ok = rng.uniform() < Self.pCorrect(thinking: k, difficulty: d)
            lengths.append(k); correct.append(ok)
            rewards.append((ok ? 1 : 0) - lengthCost * Double(k))
        }
        // Group-relative advantage: (reward − group mean) ÷ group standard deviation.
        let mean = rewards.reduce(0, +) / Double(groupSize)
        let sd = (rewards.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(groupSize)).squareRoot()
        let adv = rewards.map { sd < 1e-9 ? 0 : ($0 - mean) / sd }
        // Policy gradient: ∇ log π(k) for a softmax is (one-hot(k) − π).
        var grad = Array(repeating: 0.0, count: pi.count)
        for (k, a) in zip(lengths, adv) {
            for j in pi.indices { grad[j] += a * ((j == k ? 1 : 0) - pi[j]) }
        }
        for j in pi.indices { logits[d - 1][j] += learningRate * grad[j] / Double(groupSize) }
        let answers = (0..<groupSize).map { Answer(thinking: lengths[$0], correct: correct[$0], reward: rewards[$0], advantage: adv[$0]) }
        return Group(problem: problem, answers: answers)
    }

    /// The written-out working for a problem with `k` thinking steps (for display).
    static func working(_ problem: [Int], thinking k: Int, rng: inout SplitMix64) -> (lines: [String], answer: Int, correct: Bool) {
        var total = problem[0], lines: [String] = []
        let d = problem.count - 1
        for i in 1...d {
            if i <= k {
                let next = total + problem[i]
                lines.append("\(total) + \(problem[i]) = \(next)")
                total = next
            } else {
                // Done in its head: right with probability headAccuracy, otherwise off by a little.
                total += problem[i]
                if rng.uniform() >= headAccuracy { total += [-2, -1, 1, 2][rng.int(4)] }
            }
        }
        let truth = problem.reduce(0, +)
        return (lines, total, total == truth)
    }
}
