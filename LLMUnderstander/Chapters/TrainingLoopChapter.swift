import SwiftUI
import Charts

struct TrainingLoopChapter: View {
    @Environment(AppState.self) private var app

    var body: some View {
        @Bindable var t = app.trainer
        ChapterPage(chapter: .trainingLoop) {
            Prose("Everything comes together here. Below is a real transformer (\(t.config.nLayers) blocks, \(t.config.nHeads) heads, width \(t.config.dModel), \(t.parameterCount.formatted()) weights) and a real training loop. It reads a corpus of simple English sentences, **one character at a time**. Press **Start** and watch it go from random noise to writing sentences.")

            Card {
                HStack(spacing: 10) {
                    Button { t.isRunning ? t.pause() : t.start() } label: {
                        Label(t.isRunning ? "Pause" : (t.step == 0 ? "Start training" : "Resume"), systemImage: t.isRunning ? "pause.fill" : "play.fill")
                            .frame(minWidth: 130)
                    }
                    .buttonStyle(.borderedProminent).controlSize(.large).tint(.teal)
                    .disabled(t.finished)
                    Button { t.reset() } label: { Label("Reset", systemImage: "arrow.counterclockwise") }
                        .controlSize(.large)
                    Toggle("Slow motion", isOn: $t.slowMotion).toggleStyle(.switch)
                        .help("Run one step at a time and highlight each phase")
                    Spacer()
                    Picker("Optimizer", selection: $t.settings.optimizer) {
                        ForEach(OptimizerKind.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .frame(width: 170)
                    .disabled(t.step > 0)
                    Picker("Peak LR", selection: $t.settings.peakLR) {
                        ForEach([3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], id: \.self) { Text(String(format: "%g", $0)).tag($0) }
                    }
                    .frame(width: 140)
                    .disabled(t.step > 0)
                }
                if t.step > 0 && !t.isRunning && !t.finished {
                    Text("Settings are locked once training starts. Press Reset to try different ones (e.g. a huge learning rate, or plain SGD).")
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack(spacing: 10) {
                    StatTile(label: "Step", value: "\(t.step) / \(t.settings.totalSteps)", color: .teal)
                    StatTile(label: "Tokens seen", value: Format.big(Double(t.tokensSeen)), color: .blue)
                    StatTile(label: "Train loss", value: t.smoothedTrainLoss.last.map { String(format: "%.3f", $0.y) } ?? "—", color: .orange)
                    StatTile(label: "Validation loss", value: t.valLoss.last.map { String(format: "%.3f", $0.y) } ?? "—", color: .pink)
                    StatTile(label: "Learning rate", value: t.learningRates.last.map { String(format: "%.5f", $0.y) } ?? "—", color: .purple)
                    StatTile(label: "Gradient norm", value: t.gradNorms.last.map { String(format: "%.2f", $0.y) } ?? "—", color: .gray)
                }
            }

            HStack(alignment: .top, spacing: 18) {
                Card(title: "The loop", subtitle: t.slowMotion ? "Following one step at a time" : "Turn on slow motion to follow each phase") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("for step in range(\(t.settings.totalSteps)):").font(.system(size: 13, design: .monospaced))
                        ForEach(LoopPhase.allCases) { p in
                            let on = t.phase == p
                            HStack(spacing: 8) {
                                Text(p.code).font(.system(size: 13, weight: on ? .bold : .regular, design: .monospaced))
                                    .padding(.leading, 26)
                                Spacer()
                            }
                            .padding(.vertical, 5)
                            .background(on ? Color.teal.opacity(0.25) : Color.clear, in: RoundedRectangle(cornerRadius: 5))
                            .overlay(alignment: .trailing) {
                                Text(p.title).font(.caption).foregroundStyle(on ? .primary : .secondary).padding(.trailing, 8)
                            }
                            .animation(.easeInOut(duration: 0.2), value: on)
                        }
                    }
                }
                .frame(width: 560)

                Card(title: "What it writes", subtitle: "Prompt \"the \", sampled every 100 steps") {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(t.samples.reversed()) { s in
                                HStack(alignment: .top) {
                                    Text("step \(s.step)").font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 64, alignment: .leading)
                                    Text(s.text.replacingOccurrences(of: "\n", with: " ↵ "))
                                        .font(.system(size: 12.5, design: .monospaced))
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(height: 190)
                }
            }

            Card(title: "Loss", subtitle: "Lower is better. Dashed line = random guessing (log \(t.config.vocab) = \(String(format: "%.2f", t.uniformLoss))).") {
                Chart {
                    ForEach(t.trainLoss) { p in
                        LineMark(x: .value("step", p.x), y: .value("loss", p.y), series: .value("s", "raw"))
                            .foregroundStyle(Color.orange.opacity(0.18))
                    }
                    ForEach(t.smoothedTrainLoss) { p in
                        LineMark(x: .value("step", p.x), y: .value("loss", p.y), series: .value("s", "train"))
                            .foregroundStyle(Color.orange).lineStyle(StrokeStyle(lineWidth: 2.5))
                    }
                    ForEach(t.valLoss) { p in
                        LineMark(x: .value("step", p.x), y: .value("loss", p.y), series: .value("s", "val"))
                            .foregroundStyle(Color.pink)
                        PointMark(x: .value("step", p.x), y: .value("loss", p.y)).foregroundStyle(Color.pink).symbolSize(30)
                    }
                    RuleMark(y: .value("uniform", t.uniformLoss)).lineStyle(StrokeStyle(dash: [5, 4])).foregroundStyle(.gray)
                }
                .chartXScale(domain: 0...Double(t.settings.totalSteps))
                .chartYScale(domain: 0...4)
                .chartXAxisLabel("training step")
                .frame(height: 240)
                HStack(spacing: 16) {
                    legend(.orange, "training loss (on the batch it just learned from)")
                    legend(.pink, "validation loss (on sentences it never trains on)")
                }
            }

            HStack(spacing: 18) {
                Card(title: "Learning rate", subtitle: "Warm-up, then cosine decay") {
                    Chart(t.learningRates) { LineMark(x: .value("step", $0.x), y: .value("lr", $0.y)).foregroundStyle(.purple) }
                        .chartXScale(domain: 0...Double(t.settings.totalSteps))
                        .frame(height: 130)
                }
                Card(title: "Gradient norm", subtitle: "Size of the whole gradient (clipped at \(String(format: "%g", t.settings.clip)))") {
                    Chart(t.gradNorms) { LineMark(x: .value("step", $0.x), y: .value("norm", $0.y)).foregroundStyle(.gray) }
                        .chartXScale(domain: 0...Double(t.settings.totalSteps))
                        .frame(height: 130)
                }
            }

            KeyIdea("That's the entire algorithm behind every LLM: take a batch, predict, score, backprop, step — repeat. GPT-4-class models run this same loop with ~10¹² times more compute.")

            GoDeeper(title: "reading the curves") {
                Prose("**Fast early drop:** first the model learns which characters are common (spaces, 'e'), then common pairs ('th'), then whole words, then grammar. You can see these phases in the samples.")
                Prose("**Train vs validation:** if training loss keeps falling while validation loss rises, the model is *memorising* rather than generalising (overfitting). LLM pretraining rarely sees the same text twice, so the two curves stay together, as they do here.")
                Prose("**Spikes:** a sudden jump usually means the learning rate is too high for that moment. Try Reset with a peak learning rate of 0.1.")
                Formula("for step in range(total_steps):\n    x, y = get_batch(\"train\")            # (batch, T) token ids, y = x shifted by one\n    logits = model(x)                      # (batch, T, vocab)\n    loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))\n    optimizer.zero_grad()\n    loss.backward()\n    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n    for g in optimizer.param_groups: g[\"lr\"] = schedule(step)\n    optimizer.step()")
            }
        }
    }

    private func legend(_ c: Color, _ s: String) -> some View {
        HStack(spacing: 6) { Circle().fill(c).frame(width: 8, height: 8); Text(s).font(.caption).foregroundStyle(.secondary) }
    }
}
