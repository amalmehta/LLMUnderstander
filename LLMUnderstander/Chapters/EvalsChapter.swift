import SwiftUI
import Charts

struct EvalsChapter: View {
    @Environment(AppState.self) private var app
    @State private var n = 40.0
    @State private var p = 0.6
    @State private var gap = 0.05

    var body: some View {
        ChapterPage(chapter: .evals) {
            Prose("Training lowers the loss, but nobody ships a loss. You want to know: *can it do the job?* An **eval** (evaluation) is a fixed set of tasks with a way to score each answer. Benchmarks are evals many people share, so models can be compared.")
            Prose("Two things make evals harder than they look: the **scoring rule** changes the story the numbers tell, and a score from a few dozen questions is **noisier** than it looks.")

            ModelStatusBanner()
            curveCard
            BenchmarkItems()
            noiseCard

            KeyIdea("A score is a measurement, with a rule and an error bar. Before trusting \"it suddenly learned X\" or \"model A beats B\", ask how it was scored and how many questions it rests on.")

            GoDeeper(title: "the traps") {
                Prose("**\"Emergence\" can be the metric.** Schaeffer, Miranda and Koyejo (2023) showed that many abilities that seem to appear suddenly in bigger models turn smooth when scored per token instead of all-or-nothing. The curve above is the same effect in miniature. Some jumps may still be real; the point is to check the metric first.")
                Prose("**Contamination.** If test questions leak into the training data (they're on the web, after all), the model can score well by remembering. Labs search training data for benchmark text, and new benchmarks hide their answers or are written after the model's training cut-off. Our benchmark comes from sentences generated with a different seed than the training text, so it's held out, though the tiny grammar means similar sentences do appear in training.")
                Prose("**Goodhart's law.** \"When a measure becomes a target, it ceases to be a good measure.\" Tune on a benchmark long enough and you'll improve the benchmark more than the skill. Keep a test set you never tune on.")
                Prose("**LLM-as-judge.** For open-ended answers, another model often does the grading. Judges have known biases, such as preferring longer answers, the first of two options, or their own style; good setups swap answer order and check the judge against people.")
                Formula("standard error = √( p · (1 − p) / n )\n95% interval ≈ p ± 1.96 · standard error")
                Prose("**p** = the score as a fraction, **n** = number of independent questions. To compare two models on the same questions, a *paired* test (which questions each got right) is more sensitive than comparing two intervals.")
            }
        }
    }

    // MARK: - Two scores

    private var curveCard: some View {
        let t = app.trainer
        return Card(title: "Two scores, one model", subtitle: "Your model's score on \(Benchmark.items.count) word endings it never trained on, measured every 100 training steps") {
            if t.benchmark.count < 2 {
                Text("Train the model (Chapter 9, or the button above) and the two curves fill in as it learns.")
                    .font(.callout).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 120)
            } else {
                Chart {
                    ForEach(t.benchmarkPerChar) { q in
                        LineMark(x: .value("step", q.x), y: .value("score", q.y), series: .value("s", "per character"))
                            .foregroundStyle(Color.teal).symbol(.circle)
                    }
                    ForEach(t.benchmark) { q in
                        LineMark(x: .value("step", q.x), y: .value("score", q.y), series: .value("s", "exact match"))
                            .foregroundStyle(Color.pink).symbol(.square)
                    }
                }
                .chartXScale(domain: 0...Double(t.settings.totalSteps))
                .chartYScale(domain: 0...1)
                .chartYAxis { AxisMarks(values: [0, 0.25, 0.5, 0.75, 1]) { v in AxisGridLine(); AxisValueLabel { Text("\(Int((v.as(Double.self) ?? 0) * 100))%") } } }
                .chartXAxisLabel("training step")
                .frame(height: 220)
                HStack(spacing: 16) {
                    legend(.teal, "per character: share of letters right, each given the right letters before it")
                    legend(.pink, "exact match: the whole word right, or nothing")
                }
                Text("Same answers, two rules. Per-character accuracy climbs steadily from the first hundred steps; exact match stays near zero, then shoots up once most letters are right, because a word needs every letter. Late in training exact match also wobbles by several points between checks: with \(Benchmark.items.count) items, one word is \(String(format: "%.1f", 100 / Double(Benchmark.items.count)))%.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Noise

    private var noiseCard: some View {
        let se = Benchmark.standardError(p: p, n: Int(n))
        let seDiff = (2 * se * se).squareRoot()
        let z = gap / max(seDiff, 1e-9)
        let curve = stride(from: 10.0, through: 2000, by: 10).map { m in (m, 1.96 * Benchmark.standardError(p: p, n: Int(m))) }
        return Card(title: "How much can you trust a score?", subtitle: "The error bar of a benchmark score, from the number of questions alone") {
            HStack(alignment: .top, spacing: 24) {
                VStack(alignment: .leading, spacing: 10) {
                    HStack { Text("Questions").frame(width: 110, alignment: .leading); Slider(value: $n, in: 10...2000); Text("\(Int(n))").monospacedDigit().frame(width: 44) }
                    HStack { Text("Score").frame(width: 110, alignment: .leading); Slider(value: $p, in: 0.02...0.98); Text("\(Int((p * 100).rounded()))%").monospacedDigit().frame(width: 44) }
                    HStack(spacing: 10) {
                        StatTile(label: "Standard error", value: String(format: "± %.1f points", se * 100), color: .teal)
                        StatTile(label: "95% interval", value: String(format: "%.0f%% – %.0f%%", max(0, p - 1.96 * se) * 100, min(1, p + 1.96 * se) * 100), color: .blue)
                    }
                    Divider().padding(.vertical, 4)
                    HStack { Text("Model B is better by").frame(width: 150, alignment: .leading); Slider(value: $gap, in: 0...0.2); Text("\(Int((gap * 100).rounded())) pts").monospacedDigit().frame(width: 50) }
                    Text(z >= 1.96
                         ? "With \(Int(n)) questions each, a \(Int((gap * 100).rounded()))-point gap is about \(String(format: "%.1f", z)) standard errors: unlikely to be luck."
                         : "With \(Int(n)) questions each, a \(Int((gap * 100).rounded()))-point gap is only \(String(format: "%.1f", z)) standard errors: it could easily be luck.")
                        .font(.callout).foregroundStyle(z >= 1.96 ? .primary : .secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("Treats the two models' scores as independent; a paired comparison on the same questions can be sharper.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .frame(width: 430)
                Chart {
                    ForEach(curve, id: \.0) { m, w in
                        AreaMark(x: .value("questions", m), yStart: .value("low", -w * 100), yEnd: .value("high", w * 100))
                            .foregroundStyle(Color.teal.opacity(0.25))
                    }
                    RuleMark(x: .value("you", n)).foregroundStyle(Color.teal).lineStyle(StrokeStyle(lineWidth: 2))
                }
                .chartXScale(type: .log)
                .chartXAxisLabel("questions (log scale)")
                .chartYAxisLabel("± points (95%)")
                .frame(height: 230)
                .accessibilityLabel("Error bar width against number of questions")
                .accessibilityValue(String(format: "plus or minus %.1f points at %d questions", 1.96 * se * 100, Int(n)))
            }
        }
    }

    private func legend(_ c: Color, _ s: String) -> some View {
        HStack(spacing: 6) { Circle().fill(c).frame(width: 8, height: 8); Text(s).font(.caption).foregroundStyle(.secondary) }
    }
}

/// Kept separate so dragging the sliders above doesn't re-run the model on these items.
private struct BenchmarkItems: View {
    @Environment(AppState.self) private var app

    var body: some View {
        let t = app.trainer
        let model = t.snapshot
        let _ = t.snapshotVersion
        let shown = Array(Benchmark.items.prefix(8))
        return Card(title: "Some of the questions, answered by your model now", subtitle: "The model writes the rest of the word; the grader compares letters") {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                GridRow {
                    Text("Prompt").font(.caption).foregroundStyle(.secondary)
                    Text("Right ending").font(.caption).foregroundStyle(.secondary)
                    Text("Model wrote").font(.caption).foregroundStyle(.secondary)
                    Text("").font(.caption)
                }
                ForEach(shown) { item in
                    let guess = Benchmark.complete(model, item)
                    GridRow {
                        Text(item.prompt + "…").font(.system(size: 12.5, design: .monospaced))
                        Text(item.answer).font(.system(size: 12.5, design: .monospaced)).foregroundStyle(.secondary)
                        Text(guess.isEmpty ? "(nothing)" : guess).font(.system(size: 12.5, design: .monospaced))
                        Image(systemName: guess == item.answer ? "checkmark.circle.fill" : "xmark.circle")
                            .foregroundStyle(guess == item.answer ? .green : .secondary)
                            .accessibilityLabel(guess == item.answer ? "right" : "wrong")
                    }
                }
            }
        }
    }

}
