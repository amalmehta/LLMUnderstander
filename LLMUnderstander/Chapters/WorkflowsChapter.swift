import SwiftUI
import Charts

struct WorkflowsChapter: View {
    @State private var pattern: Pattern = .chaining
    @State private var p = 0.6
    @State private var n = 5.0

    // bandit race
    @State private var states: [Bandit.Strategy: Bandit.State] = WorkflowsChapter.freshStates()
    @State private var rngs: [Bandit.Strategy: SplitMix64] = WorkflowsChapter.freshRngs(seed: 1)
    @State private var seed: UInt64 = 1
    @State private var remaining = 0
    @State private var focus: Bandit.Strategy = .thompson
    private let total = 600

    static let colors: [Bandit.Strategy: Color] = [.greedy: .red, .epsilon: .orange, .ucb: .blue, .thompson: .green]

    enum Pattern: String, CaseIterable, Identifiable {
        case chaining = "Chaining"
        case routing = "Routing"
        case parallel = "Parallel & voting"
        case orchestrator = "Orchestrator–workers"
        case evaluator = "Evaluator–optimizer"
        case agent = "Autonomous agent"
        var id: String { rawValue }

        var steps: [(String, String)] {
            switch self {
            case .chaining: [("Input", "a topic"), ("LLM call 1", "write an outline"), ("Check", "code: outline has 5 parts?"), ("LLM call 2", "write each part"), ("LLM call 3", "translate"), ("Output", "")]
            case .routing: [("Input", "support ticket"), ("Router LLM", "classify: billing / bug / question"), ("Specialist", "a prompt + tools for that type"), ("Output", "")]
            case .parallel: [("Input", "one question"), ("N calls at once", "same prompt (voting) or\ndifferent sub-parts (sectioning)"), ("Combine", "majority vote, or merge"), ("Output", "")]
            case .orchestrator: [("Input", "a big task"), ("Orchestrator LLM", "plans and splits the work"), ("Workers", "sub-agents, each with\nits own clean context"), ("Orchestrator", "merges results"), ("Output", "")]
            case .evaluator: [("Input", "a draft request"), ("Generator LLM", "writes a draft"), ("Evaluator", "LLM or tests give feedback"), ("Loop", "revise until it passes"), ("Output", "")]
            case .agent: [("Input", "an open-ended goal"), ("Model", "decides the next step"), ("Tools", "act, observe the result"), ("Repeat", "the model decides when it's done"), ("Output", "")]
            }
        }

        var when: LocalizedStringKey {
            switch self {
            case .chaining: "**Use when** a task splits cleanly into fixed steps. Each call is easier than the whole, and code can check the output between steps. **Cost:** more calls in sequence, so slower."
            case .routing: "**Use when** inputs fall into distinct kinds that need different handling. Easy questions can even go to a smaller, cheaper model. **Cost:** a misrouted input gets the wrong specialist."
            case .parallel: "**Use when** sub-tasks are independent (sectioning), or when you want more reliability by asking several times and taking a vote. See the calculator below. **Cost:** N× the tokens."
            case .orchestrator: "**Use when** you can't know the sub-tasks in advance (\"change every file that uses this API\"). The plan is decided by the model at run time. **Cost:** more tokens and more coordination to get right."
            case .evaluator: "**Use when** there are clear criteria and feedback measurably helps, like code that must pass tests or a translation that must keep its meaning. **Cost:** loops can run long; cap the rounds."
            case .agent: "**Use when** the path really can't be predicted and you trust the model to steer (Chapters 14–17). The most flexible pattern and the hardest to make reliable. Start with the simplest pattern that works."
            }
        }
    }

    var body: some View {
        ChapterPage(chapter: .workflows) {
            Prose("Not every useful system is a free-running agent. Often the best design is a **workflow**: model calls wired together in a fixed pattern by ordinary code. And whenever a system can choose between options — prompts, models, tools, strategies — it faces the oldest problem in RL: **explore or exploit?**")

            Card(title: "Six common patterns", subtitle: "From most predictable to most autonomous") {
                Picker("", selection: $pattern) { ForEach(Pattern.allCases) { Text($0.rawValue).tag($0) } }
                    .pickerStyle(.segmented).labelsHidden()
                HStack(spacing: 6) {
                    ForEach(Array(pattern.steps.enumerated()), id: \.offset) { i, s in
                        VStack(spacing: 3) {
                            Text(s.0).font(.system(size: 13, weight: .semibold)).multilineTextAlignment(.center)
                            if !s.1.isEmpty { Text(s.1).font(.system(size: 10.5)).foregroundStyle(.secondary).multilineTextAlignment(.center) }
                        }
                        .padding(8)
                        .frame(maxWidth: .infinity, minHeight: 70)
                        .background(color(i).opacity(0.13), in: RoundedRectangle(cornerRadius: 10))
                        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(color(i).opacity(0.45)))
                        if i < pattern.steps.count - 1 {
                            Image(systemName: pattern == .evaluator && i == 3 || pattern == .agent && i == 3 ? "arrow.uturn.left" : "arrow.right")
                                .foregroundStyle(.secondary).font(.caption)
                        }
                    }
                }
                Text(pattern.when).font(.system(size: 15)).fixedSize(horizontal: false, vertical: true)
            }

            votingCard
            banditCard

            KeyIdea("Use the simplest pattern that does the job: one call, then a fixed workflow, then an agent. Whenever the system must choose between options it can't fully evaluate in advance, give it a principled way to explore — like UCB or Thompson sampling — rather than greedily trusting early results.", color: .green)

            GoDeeper(title: "where bandits show up in LLM systems") {
                Prose("**Choosing prompts, models or tools online:** each user request is a pull; success (a thumbs-up, a passing test) is the reward. A bandit shifts traffic toward what works while still checking the alternatives, which is smarter than a fixed 50/50 A/B test.")
                Prose("**Exploration in RL training:** in Chapter 17 the policy only discovered the test-deleting shortcut because it sometimes tried it. Sampling at a temperature is how language models explore. Too little and RL never finds better strategies; too much and training is noisy.")
                Prose("**Search at inference time:** tree search over reasoning steps or code edits must decide which branch to expand next. Monte Carlo tree search does this with the UCB rule from this page.")
                Formula("UCB1:      pick argmax_i [ mean_i + √(2 ln t / n_i) ]\nThompson:  draw θ_i ~ Beta(wins_i + 1, losses_i + 1), pick argmax_i θ_i\nRegret:    Σ_t ( best rate − rate of the arm chosen at t )")
            }
        }
        .ticking(every: 0.03, while: remaining > 0) { tick() }
    }

    private func color(_ i: Int) -> Color {
        [Color.gray, .green, .teal, .blue, .purple, .gray][min(i, 5)]
    }

    // MARK: - Voting

    private var votingCard: some View {
        let ns = stride(from: 1, through: 21, by: 2).map { $0 }
        return Card(title: "Why ask more than once?", subtitle: "Each attempt is independently right with probability p. Exact binomial maths.") {
            HStack {
                Text("Single attempt right: \(Int(p * 100))%").monospacedDigit().frame(width: 200, alignment: .leading)
                Slider(value: $p, in: 0.05...0.95)
            }.font(.callout)
            HStack {
                Text("Attempts: \(Int(n))").monospacedDigit().frame(width: 200, alignment: .leading)
                Slider(value: $n, in: 1...21, step: 2)
            }.font(.callout)
            Chart {
                ForEach(ns, id: \.self) { k in
                    LineMark(x: .value("n", k), y: .value("p", Sampling.majority(p: p, n: k)), series: .value("m", "majority vote"))
                        .foregroundStyle(by: .value("m", "majority vote"))
                    LineMark(x: .value("n", k), y: .value("p", Sampling.anyCorrect(p: p, n: k)), series: .value("m", "best-of-n with a checker"))
                        .foregroundStyle(by: .value("m", "best-of-n with a checker"))
                }
                RuleMark(x: .value("n", Int(n))).foregroundStyle(.gray.opacity(0.5))
            }
            .chartForegroundStyleScale(["majority vote": Color.blue, "best-of-n with a checker": Color.green])
            .chartYScale(domain: 0...1)
            .chartXScale(domain: 1...21)
            .chartXAxisLabel("attempts")
            .frame(height: 180)
            HStack(spacing: 10) {
                StatTile(label: "One attempt", value: String(format: "%.0f%%", p * 100), color: .gray)
                StatTile(label: "Majority of \(Int(n))", value: String(format: "%.0f%%", Sampling.majority(p: p, n: Int(n)) * 100), color: .blue)
                StatTile(label: "Any of \(Int(n)), checked", value: String(format: "%.0f%%", Sampling.anyCorrect(p: p, n: Int(n)) * 100), color: .green)
            }
            Text("Voting helps only if a single attempt is right more often than not; below 50%, the majority is *more* likely to be wrong. With a reliable checker (tests, a verifier), any one success is enough. The catch: this assumes independent attempts, and real attempts from the same model tend to make the same mistakes.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    // MARK: - Bandit race

    private static func freshStates() -> [Bandit.Strategy: Bandit.State] {
        Dictionary(uniqueKeysWithValues: Bandit.Strategy.allCases.map { ($0, Bandit.State(k: Bandit.arms.count)) })
    }

    private static func freshRngs(seed: UInt64) -> [Bandit.Strategy: SplitMix64] {
        Dictionary(uniqueKeysWithValues: Bandit.Strategy.allCases.enumerated().map { ($0.element, SplitMix64(seed: seed &* 1000 &+ UInt64($0.offset))) })
    }

    private var banditCard: some View {
        let st = states[focus]!
        return Card(title: "Multi-armed bandit: explore or exploit?", subtitle: "Five prompts with hidden success rates. Four strategies each get \(total) tries. Regret = successes lost by not always using the best prompt.") {
            HStack {
                Button(remaining > 0 ? "Running…" : "Run \(total) tries") {
                    if states.values.first!.t >= total { resetBandit(newSeed: false) }
                    remaining = total
                }
                .buttonStyle(.borderedProminent).tint(.green).disabled(remaining > 0)
                Button("New random run") { resetBandit(newSeed: true); remaining = total }
                Spacer()
                Text("Tries: \(states.values.first!.t)").monospacedDigit().font(.callout)
            }
            HStack(alignment: .top, spacing: 20) {
                VStack(alignment: .leading) {
                    Text("Cumulative regret (lower is better)").font(.subheadline.weight(.semibold))
                    Chart {
                        ForEach(Bandit.Strategy.allCases) { s in
                            ForEach(states[s]!.regretHistory) { pt in
                                LineMark(x: .value("try", pt.x), y: .value("regret", pt.y), series: .value("s", s.rawValue))
                                    .foregroundStyle(by: .value("s", s.rawValue))
                            }
                        }
                    }
                    .chartForegroundStyleScale(domain: Bandit.Strategy.allCases.map(\.rawValue),
                                               range: Bandit.Strategy.allCases.map { Self.colors[$0]! })
                    .chartXScale(domain: 0...total)
                    .frame(height: 230)
                }
                VStack(alignment: .leading, spacing: 8) {
                    Picker("Strategy", selection: $focus) { ForEach(Bandit.Strategy.allCases) { Text($0.rawValue).tag($0) } }
                    Text("How often \(focus.rawValue) tried each prompt").font(.subheadline.weight(.semibold))
                    Chart(Bandit.arms) { a in
                        BarMark(x: .value("tries", st.pulls[a.id]), y: .value("arm", "\(a.name) (\(Int(a.rate * 100))%)"))
                            .foregroundStyle(a.rate == Bandit.bestRate ? Color.green : Color.gray.opacity(0.6))
                            .annotation(position: .trailing) {
                                Text(st.pulls[a.id] > 0 ? String(format: "est. %.0f%%", st.mean(a.id) * 100) : "").font(.caption2)
                            }
                    }
                    .frame(height: 170)
                }
                .frame(width: 340)
            }
            Text(.init("**\(focus.rawValue):** \(focus.explanation)")).font(.callout).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 10) {
                ForEach(Bandit.Strategy.allCases) { s in
                    StatTile(label: "\(s.rawValue) regret", value: String(format: "%.1f", states[s]!.regret), color: Self.colors[s]!)
                }
            }
            Text("Try **New random run** a few times. Greedy is sometimes perfect and sometimes stuck on a bad prompt for good. Thompson sampling is the most reliable here: its regret curve bends flat as it stops wasting tries. UCB1 is the most cautious: with prompts this close in quality it keeps re-checking them, so over 600 tries its regret is often higher, but it never gets permanently stuck the way greedy can.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    private func resetBandit(newSeed: Bool) {
        if newSeed { seed &+= 1 }
        states = Self.freshStates()
        rngs = Self.freshRngs(seed: seed)
        remaining = 0
    }

    private func tick() {
        guard remaining > 0 else { return }
        for _ in 0..<6 where remaining > 0 {
            for s in Bandit.Strategy.allCases {
                var st = states[s]!, r = rngs[s]!
                Bandit.step(s, &st, rng: &r)
                states[s] = st; rngs[s] = r
            }
            remaining -= 1
        }
    }
}
