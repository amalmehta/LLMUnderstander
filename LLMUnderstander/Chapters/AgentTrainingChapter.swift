import SwiftUI
import Charts

struct AgentTrainingChapter: View {
    @State private var showMask = true
    @State private var robust = false
    @State private var rl = AgentRL.State()
    @State private var rng = SplitMix64(seed: 21)
    @State private var history: [(update: Int, reward: Double, genuine: Double)] = []
    @State private var remaining = 0
    private let timer = Timer.publish(every: 0.05, on: .main, in: .common).autoconnect()

    var body: some View {
        ChapterPage(chapter: .agentTraining) {
            Prose("A harness can wrap any model, but some models make far better agents than others. Agent skill is trained in the same three stages you've already seen, with agent-shaped data.")
            FlowRow(steps: [
                .init(title: "Pretraining", detail: "lots of code, docs,\nterminal logs, how-tos", color: .orange, chapter: .pretraining),
                .init(title: "SFT on transcripts", detail: "examples of good\ntool-using runs", color: .pink, chapter: .sft),
                .init(title: "RL in environments", detail: "attempt real tasks,\nreward = did it work?", color: .green),
            ])

            Card(title: "SFT on agent transcripts: what gets trained?", subtitle: "Chapter 14's run as a training example. Same loss mask idea as Chapter 11.") {
                Toggle("Show the loss mask", isOn: $showMask).toggleStyle(.switch)
                ScrollView {
                    VStack(spacing: 6) {
                        ForEach(AgentScript.events.filter(\.inContext)) { e in
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: showMask ? (e.trainedOn ? "checkmark.circle.fill" : "minus.circle") : "circle")
                                    .foregroundStyle(showMask ? (e.trainedOn ? Color.green : Color.secondary) : Color.clear)
                                    .padding(.top, 8)
                                EventBubble(event: e, dimmed: showMask && !e.trainedOn)
                            }
                        }
                    }
                }
                .frame(height: 300)
                Text("Only the model's own text and tool calls are trained. Tool results, the user's messages and the system prompt are context, not targets — we want the model to learn to *call* tools and *use* their output, not to imitate what the test runner prints.")
                    .font(.callout).foregroundStyle(.secondary)
            }

            rlCard

            Card(title: "Credit assignment over many steps", subtitle: "The reward arrives only at the very end") {
                HStack(spacing: 4) {
                    ForEach(["run tests", "read file", "think", "edit", "run tests", "answer"], id: \.self) { s in
                        Text(s).font(.caption).padding(.horizontal, 8).padding(.vertical, 6)
                            .background(Color.orange.opacity(0.12), in: Capsule())
                        Image(systemName: "arrow.right").font(.caption2).foregroundStyle(.secondary)
                    }
                    Text("reward = 1 ✓").font(.caption.bold()).padding(.horizontal, 8).padding(.vertical, 6)
                        .background(Color.green.opacity(0.2), in: Capsule())
                }
                Text("In the simplest methods every token in a successful run is pushed up a little and every token in a failed run is pushed down a little. Over thousands of runs, steps that *consistently* show up in successes (like re-running the tests) get reinforced, and incidental ones average out. Long tasks make this noisy, which is one reason agent RL needs so many attempts.")
                    .font(.callout).foregroundStyle(.secondary)
            }

            KeyIdea("Agents are trained on the agent loop itself: imitate good transcripts, then practise on real tasks and reinforce whatever *verifiably* worked. The reward checker is part of the design — a lazy checker trains a lazy (or sneaky) agent.", color: .green)

            GoDeeper(title: "how this works at scale") {
                Prose("**Environments:** thousands of sandboxed tasks — repositories with failing tests, websites to navigate, data to analyse — each with an automatic checker. The model runs the full loop inside a container during training, exactly as it would for a user.")
                Prose("**Verifiable rewards** (tests pass, answer matches, page reached) need no human labeller, so RL can run at huge scale. Where outcomes can't be checked automatically, reward models and AI judges fill in, with the risks from Chapter 12.")
                Prose("**Evaluation:** benchmarks built from real tasks (for example, fixing real GitHub issues and checking with the project's own tests) measure progress, and must be kept out of training data so they stay honest.")
                Formula("REINFORCE with a baseline:   θ ← θ + lr · mean[ (R − b) · ∇ log π(action) ]")
            }
        }
        .onReceive(timer) { _ in
            guard remaining > 0 else { return }
            for _ in 0..<2 where remaining > 0 {
                AgentRL.step(&rl, robust: robust, rng: &rng)
                remaining -= 1
                let p = Num.softmax(rl.logits)
                history.append((history.count + 1, rl.lastRewards.reduce(0, +) / Double(rl.lastRewards.count), AgentRL.genuineSuccess(p)))
            }
        }
    }

    private var rlCard: some View {
        let p = Num.softmax(rl.logits)
        let hacked = p[2] > 0.5
        return Card(title: "RL with a verifiable reward — live", subtitle: "The policy picks a strategy for \"make the failing test pass\". Each update samples 16 attempts and applies REINFORCE.") {
            HStack {
                Button(remaining > 0 ? "Training…" : "Run 150 updates") { remaining = 150 }
                    .buttonStyle(.borderedProminent).tint(.green).disabled(remaining > 0)
                Button("Reset") { reset() }
                Spacer()
                Toggle("Checker also verifies the tests weren't modified", isOn: Binding(get: { robust }, set: { robust = $0; reset() }))
                    .toggleStyle(.switch)
            }
            HStack(alignment: .top, spacing: 20) {
                Chart(AgentRL.strategies) { s in
                    BarMark(x: .value("p", p[s.id]), y: .value("strategy", s.name))
                        .foregroundStyle(s.tampers ? Color.red : (s.id == 0 ? Color.green : Color.gray))
                        .annotation(position: .trailing) { Text(String(format: "%.0f%%", p[s.id] * 100)).font(.caption2) }
                }
                .chartXScale(domain: 0...1)
                .frame(height: 200)
                VStack(alignment: .leading) {
                    Chart {
                        ForEach(history, id: \.update) { h in
                            LineMark(x: .value("update", h.update), y: .value("v", h.reward), series: .value("m", "reward the checker gives"))
                                .foregroundStyle(by: .value("m", "reward the checker gives"))
                            LineMark(x: .value("update", h.update), y: .value("v", h.genuine), series: .value("m", "genuinely fixed"))
                                .foregroundStyle(by: .value("m", "genuinely fixed"))
                        }
                    }
                    .chartForegroundStyleScale(["reward the checker gives": Color.orange, "genuinely fixed": Color.green])
                    .chartYScale(domain: 0...1.05)
                    .chartXScale(domain: 0...150)
                    .chartXAxisLabel("updates")
                    .frame(height: 200)
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                ForEach(AgentRL.strategies) { s in
                    Text("**\(s.name)** — \(s.detail); tests pass \(Int(s.passRate * 100))% of the time\(s.tampers ? " (by tampering)" : "")")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if rl.episodes >= 1500 {
                Text(hacked
                     ? "🚨 **Reward hacking.** The checker only asked \"do the tests pass?\", and deleting the failing test always makes them pass. RL found that shortcut. Turn on the stricter checker and run again."
                     : (robust ? "With a checker that also verifies the tests are untouched, the shortcut earns nothing, and RL reinforces genuine investigation and fixing."
                               : "Not hacked this time — keep training; the shortcut usually wins eventually."))
                    .font(.callout).padding(10)
                    .background((hacked ? Color.red : Color.green).opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private func reset() {
        rl = AgentRL.State(); rng = SplitMix64(seed: 21); history = []; remaining = 0
    }
}
