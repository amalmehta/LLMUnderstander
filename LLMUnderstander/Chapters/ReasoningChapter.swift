import SwiftUI
import Charts

struct ReasoningChapter: View {
    @State private var toy = ReasoningToy()
    @State private var rng = SplitMix64(seed: 42)
    @State private var updates = 0
    @State private var history: [(x: Int, accuracy: Double, length: Double)] = []
    @State private var lastGroup: ReasoningToy.Group?
    @State private var silentCount = 0
    @State private var running = false
    @State private var example = [7, 5, 9, 3]

    var body: some View {
        ChapterPage(chapter: .reasoning) {
            Prose("Preference tuning rewards answers people *like*. For maths, code and puzzles there's something better: a **checker**. Did the number match? Did the tests pass? Rewarding only answers the checker accepts is called **RL with verifiable rewards**, and it's how today's reasoning models are trained.")
            Prose("Nobody tells the model to show its working. It discovers that writing out intermediate steps before answering (a **chain of thought**) gets more answers right, so RL makes it do that more, and its answers get longer.")

            liveCard
            exampleCard

            KeyIdea("Reward the result, not the method, and let the model find the method. Thinking longer isn't taught; it's what earns reward when problems need several steps.")

            GoDeeper(title: "GRPO and its relatives") {
                Formula("advantage_i = (reward_i − mean of the group) ÷ standard deviation of the group\nupdate ∝ Σ_i advantage_i · ∇ log π(answer_i)   (minus a small KL penalty to the starting model)")
                Prose("**GRPO** (group relative policy optimisation) was introduced with DeepSeekMath (Shao et al., 2024) and used to train DeepSeek-R1 (2025). Older RLHF used PPO, which needs a second network, a *value model*, to estimate how good each situation is. GRPO replaces it with the group: sample several answers to the same problem, and the group's average *is* the baseline. Cheaper, and simple.")
                Prose("**No signal from easy or impossible problems.** If all 8 answers are right (or all wrong), every advantage is 0 and nothing is learned. Training sets are filtered toward problems the model gets right *sometimes*. The counter above shows how often this happens here.")
                Prose("**Outcome vs process rewards.** We reward only the final answer (an *outcome* reward). A *process* reward grades each step, which gives more signal but needs a grader for steps, and models learn to please that grader.")
                Prose("**Overthinking.** More thinking costs time and money. Raise the thinking cost and the policy settles on the shortest working that still gets the answer: here, one step per addition.")
                Prose("**What's simplified.** A real model chooses every token, not one number, and its \"steps\" are free-form text. The update maths is the same: sample a group, score, compare with the group, push up what beat it.")
            }
        }
        .onAppear { if history.isEmpty { record() } }
        .task(id: running) {
            guard running else { return }
            while running && updates < 300 {
                for _ in 0..<3 { step() }
                try? await Task.sleep(for: .milliseconds(40))
            }
            running = false
        }
    }

    // MARK: - Live GRPO

    private var liveCard: some View {
        Card(title: "Live: GRPO on a toy reasoning model", subtitle: "Sums of 2–6 digits. Before answering, the model picks how many additions to write out; the rest it does in its head, where each is right only \(Int(ReasoningToy.headAccuracy * 100))% of the time.") {
            HStack(spacing: 10) {
                Button { running.toggle() } label: {
                    Label(running ? "Pause" : (updates == 0 ? "Train with GRPO" : "Resume"), systemImage: running ? "pause.fill" : "play.fill")
                        .frame(minWidth: 130)
                }
                .buttonStyle(.borderedProminent).tint(.pink).controlSize(.large)
                .disabled(updates >= 300)
                Button("One update") { step() }.disabled(running || updates >= 300)
                Button { reset() } label: { Label("Reset", systemImage: "arrow.counterclockwise") }
                Spacer()
                VStack(alignment: .leading, spacing: 2) {
                    Text("Thinking cost: \(String(format: "%.2f", toy.lengthCost)) per step").font(.caption)
                    Slider(value: $toy.lengthCost, in: 0...0.15).frame(width: 180)
                        .accessibilityLabel("Thinking cost per step")
                }
            }
            HStack(spacing: 10) {
                StatTile(label: "Updates", value: "\(updates)", color: .pink)
                StatTile(label: "Accuracy", value: String(format: "%.0f%%", toy.expectedAccuracy * 100), color: .green)
                StatTile(label: "Average thinking", value: String(format: "%.1f steps", toy.expectedLength), color: .purple)
                StatTile(label: "Groups with no signal", value: updates == 0 ? "—" : "\(silentCount) of \(updates)", color: .gray)
            }
            HStack(alignment: .top, spacing: 18) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Over training").font(.subheadline.weight(.semibold))
                    Chart {
                        ForEach(history, id: \.x) { h in
                            LineMark(x: .value("update", h.x), y: .value("value", h.accuracy), series: .value("s", "Accuracy"))
                                .foregroundStyle(Color.green)
                            LineMark(x: .value("update", h.x), y: .value("value", h.length / Double(ReasoningToy.maxThinking)), series: .value("s", "Thinking"))
                                .foregroundStyle(Color.purple)
                        }
                    }
                    .chartXScale(domain: 0...300)
                    .chartYScale(domain: 0...1)
                    .chartYAxis { AxisMarks(values: [0, 0.5, 1]) { v in AxisGridLine(); AxisValueLabel { Text("\(Int((v.as(Double.self) ?? 0) * 100))%") } } }
                    .chartXAxisLabel("GRPO updates")
                    .frame(height: 170)
                    HStack(spacing: 14) {
                        legend(.green, "accuracy")
                        legend(.purple, "thinking length (share of the 6-step maximum)")
                    }
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("How long it thinks, by difficulty").font(.subheadline.weight(.semibold))
                    policyGrid
                    Text("Darker = more likely. The outlined box is one step per addition, just enough.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .frame(width: 330)
            }
            groupView
        }
    }

    private var policyGrid: some View {
        Grid(horizontalSpacing: 3, verticalSpacing: 3) {
            GridRow {
                Text("").frame(width: 72)
                ForEach(0...ReasoningToy.maxThinking, id: \.self) { k in
                    Text("\(k)").font(.caption2).foregroundStyle(.secondary).frame(width: 30)
                }
            }
            ForEach(Array(ReasoningToy.difficulties), id: \.self) { d in
                let p = toy.policy(difficulty: d)
                GridRow {
                    Text("\(d) addition\(d == 1 ? "" : "s")").font(.caption).frame(width: 72, alignment: .leading)
                    ForEach(0...ReasoningToy.maxThinking, id: \.self) { k in
                        RoundedRectangle(cornerRadius: 3)
                            .fill(Color.purple.opacity(0.08 + 0.9 * p[k]))
                            .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(k == d ? Color.primary.opacity(0.6) : .clear, lineWidth: 1.5))
                            .frame(width: 30, height: 20)
                            .animation(.easeOut(duration: 0.15), value: p[k])
                    }
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(d) additions")
                .accessibilityValue("most likely \(p.indices.max { p[$0] < p[$1] }!) thinking steps, \(Int(p.max()! * 100)) percent")
            }
            GridRow {
                Text("").frame(width: 72)
                Text("thinking steps →").font(.caption2).foregroundStyle(.secondary).gridCellColumns(7)
            }
        }
    }

    @ViewBuilder private var groupView: some View {
        if let g = lastGroup {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Latest group: \(g.problem.map(String.init).joined(separator: " + ")) = \(g.problem.reduce(0, +))")
                        .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    if g.silent {
                        Text("all 8 scored the same → advantage 0 → no update")
                            .font(.caption).padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Color.gray.opacity(0.18), in: Capsule())
                    }
                }
                HStack(spacing: 6) {
                    ForEach(Array(g.answers.enumerated()), id: \.offset) { _, a in
                        VStack(spacing: 3) {
                            Image(systemName: a.correct ? "checkmark.circle.fill" : "xmark.circle")
                                .foregroundStyle(a.correct ? .green : .secondary)
                            Text("\(a.thinking) step\(a.thinking == 1 ? "" : "s")").font(.caption2)
                            Text(String(format: "%+.1f", a.advantage)).font(.caption2.monospacedDigit())
                                .foregroundStyle(a.advantage > 0 ? .green : (a.advantage < 0 ? .pink : .secondary))
                        }
                        .frame(width: 62).padding(.vertical, 6)
                        .background(Color(nsColor: .windowBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                        .accessibilityElement(children: .ignore)
                        .accessibilityLabel("\(a.thinking) steps, \(a.correct ? "right" : "wrong"), advantage \(String(format: "%.1f", a.advantage))")
                    }
                }
                Text("Bottom number = advantage: how much better than its group's average. Positive pushes that thinking length up; negative pushes it down.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .transition(.opacity)
        }
    }

    // MARK: - A worked example

    private var exampleCard: some View {
        let d = example.count - 1
        let p = toy.policy(difficulty: d)
        let k = p.indices.max { p[$0] < p[$1] }!
        var r = SplitMix64(seed: UInt64(updates / 10 + example.reduce(0, +)))
        let w = ReasoningToy.working(example, thinking: k, rng: &r)
        return Card(title: "What its answer looks like right now", subtitle: "The model's most likely thinking length for this problem, written out") {
            HStack(alignment: .top, spacing: 20) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Q: \(example.map(String.init).joined(separator: " + ")) = ?").font(.system(size: 14, design: .monospaced))
                    if w.lines.isEmpty {
                        Text("(answers straight away)").font(.system(size: 13, design: .monospaced)).foregroundStyle(.secondary)
                    }
                    ForEach(w.lines, id: \.self) { l in
                        Text("  " + l).font(.system(size: 13, design: .monospaced)).foregroundStyle(.purple)
                            .transition(.opacity.combined(with: .move(edge: .leading)))
                    }
                    Text("A: \(w.answer)  \(w.correct ? "✓" : "✗ (true answer \(example.reduce(0, +)))")")
                        .font(.system(size: 14, weight: .semibold, design: .monospaced))
                        .foregroundStyle(w.correct ? .green : .pink)
                }
                .animation(.easeOut(duration: 0.2), value: w.lines)
                Spacer()
                Button("New problem") {
                    var g = SplitMix64(seed: UInt64.random(in: 1...UInt64.max))
                    example = ReasoningToy.makeProblem(&g)
                }
            }
        }
    }

    // MARK: - Actions

    private func step() {
        let g = toy.step(&rng)
        if g.silent { silentCount += 1 }
        updates += 1
        withAnimation(.easeOut(duration: 0.15)) { lastGroup = g }
        record()
    }

    private func record() {
        history.append((updates, toy.expectedAccuracy, toy.expectedLength))
    }

    private func reset() {
        running = false
        let cost = toy.lengthCost
        toy = ReasoningToy()
        toy.lengthCost = cost
        rng = SplitMix64(seed: 42)
        updates = 0; silentCount = 0; lastGroup = nil
        history = []
        record()
    }

    private func legend(_ c: Color, _ s: String) -> some View {
        HStack(spacing: 6) { Circle().fill(c).frame(width: 8, height: 8); Text(s).font(.caption).foregroundStyle(.secondary) }
    }
}
