import Foundation

/// Multi-armed bandits: pick one of K options again and again, each with an unknown
/// success rate, and try to succeed as often as possible. The simplest form of the
/// explore-vs-exploit problem that runs through RL and agent design.
enum Bandit {
    struct Arm: Identifiable {
        let id: Int
        let name: String
        let rate: Double
    }

    /// Five ways an agent could tackle a task, each with a hidden success rate.
    static let arms: [Arm] = [
        Arm(id: 0, name: "Prompt A", rate: 0.30),
        Arm(id: 1, name: "Prompt B", rate: 0.45),
        Arm(id: 2, name: "Prompt C", rate: 0.62),
        Arm(id: 3, name: "Prompt D", rate: 0.55),
        Arm(id: 4, name: "Prompt E", rate: 0.50),
    ]
    static var bestRate: Double { arms.map(\.rate).max()! }

    enum Strategy: String, CaseIterable, Identifiable {
        case greedy = "Greedy"
        case epsilon = "ε-greedy (10%)"
        case ucb = "UCB1"
        case thompson = "Thompson sampling"
        var id: String { rawValue }

        var explanation: String {
            switch self {
            case .greedy: "Try each option once, then always pick the best average so far. Never explores, so an unlucky start locks it onto a worse option forever."
            case .epsilon: "Pick the best average 90% of the time and a random option 10% of the time. Simple and robust, but it keeps wasting 10% on options it already knows are bad."
            case .ucb: "Pick the option with the highest *optimistic* estimate: average + a bonus that shrinks as an option gets tried more. Uncertainty is treated as opportunity."
            case .thompson: "Keep a belief (a Beta distribution) about each option's success rate, draw one plausible rate from each belief, and pick the highest draw. Explores exactly as much as its uncertainty warrants."
            }
        }
    }

    struct State {
        var pulls: [Int]
        var wins: [Int]
        var regret = 0.0
        var regretHistory: [Point] = []
        var successes = 0
        var t = 0

        init(k: Int) { pulls = Array(repeating: 0, count: k); wins = Array(repeating: 0, count: k) }

        func mean(_ i: Int) -> Double { pulls[i] == 0 ? 0 : Double(wins[i]) / Double(pulls[i]) }
    }

    static func choose(_ s: Strategy, _ st: State, rng: inout SplitMix64) -> Int {
        let k = st.pulls.count
        if let untried = st.pulls.firstIndex(of: 0), s != .thompson { return untried }
        switch s {
        case .greedy:
            return (0..<k).max { st.mean($0) < st.mean($1) }!
        case .epsilon:
            if rng.uniform() < 0.1 { return rng.int(k) }
            return (0..<k).max { st.mean($0) < st.mean($1) }!
        case .ucb:
            let total = Double(st.t)
            return (0..<k).max { a, b in
                st.mean(a) + (2 * log(total) / Double(st.pulls[a])).squareRoot()
                    < st.mean(b) + (2 * log(total) / Double(st.pulls[b])).squareRoot()
            }!
        case .thompson:
            let draws = (0..<k).map { betaSample(Double(st.wins[$0] + 1), Double(st.pulls[$0] - st.wins[$0] + 1), rng: &rng) }
            return draws.indices.max { draws[$0] < draws[$1] }!
        }
    }

    /// One pull: choose an arm, see success or failure, update counts and regret.
    static func step(_ s: Strategy, _ st: inout State, rng: inout SplitMix64) {
        let a = choose(s, st, rng: &rng)
        let win = rng.uniform() < arms[a].rate
        st.pulls[a] += 1
        if win { st.wins[a] += 1; st.successes += 1 }
        st.t += 1
        st.regret += bestRate - arms[a].rate
        if st.t % 5 == 0 { st.regretHistory.append(Point(x: Double(st.t), y: st.regret)) }
    }

    static func run(_ s: Strategy, steps: Int, seed: UInt64) -> State {
        var st = State(k: arms.count)
        var rng = SplitMix64(seed: seed)
        for _ in 0..<steps { step(s, &st, rng: &rng) }
        return st
    }

    // MARK: Beta sampling (Marsaglia–Tsang gamma)

    static func gammaSample(_ shape: Double, rng: inout SplitMix64) -> Double {
        if shape < 1 {
            return gammaSample(shape + 1, rng: &rng) * pow(max(rng.uniform(), 1e-12), 1 / shape)
        }
        let d = shape - 1.0 / 3, c = 1 / (9 * d).squareRoot()
        while true {
            var x = 0.0, v = 0.0
            repeat { x = rng.normal(); v = 1 + c * x } while v <= 0
            v = v * v * v
            let u = rng.uniform()
            if u < 1 - 0.0331 * x * x * x * x || log(max(u, 1e-300)) < 0.5 * x * x + d * (1 - v + log(v)) { return d * v }
        }
    }

    static func betaSample(_ a: Double, _ b: Double, rng: inout SplitMix64) -> Double {
        let x = gammaSample(a, rng: &rng), y = gammaSample(b, rng: &rng)
        return x / (x + y)
    }
}

/// Exact maths for "run the model several times" workflow patterns.
enum Sampling {
    static func choose(_ n: Int, _ k: Int) -> Double {
        var r = 1.0
        for i in 0..<k { r *= Double(n - i) / Double(i + 1) }
        return r
    }

    /// Majority vote over n independent attempts (n odd), each right with probability p,
    /// on a question where wrong answers disagree with each other.
    static func majority(p: Double, n: Int) -> Double {
        (n / 2 + 1...n).reduce(0) { $0 + choose(n, $1) * pow(p, Double($1)) * pow(1 - p, Double(n - $1)) }
    }

    /// Best-of-n with a perfect checker (e.g. run the tests): succeed if any attempt is right.
    static func anyCorrect(p: Double, n: Int) -> Double { 1 - pow(1 - p, Double(n)) }
}
