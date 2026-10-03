import SwiftUI

struct BigPictureChapter: View {
    var body: some View {
        ChapterPage(chapter: .bigPicture) {
            Prose("A large language model is a function that reads some text and outputs **a probability for every possible next piece of text**. That's it. Everything impressive — answering questions, writing code, holding a conversation — comes from doing that one thing extremely well, over and over.")
            Prose("**Training** is how the function gets good: show it text, measure how wrong its guesses are, and nudge its billions of numbers (its *weights*) so it's a little less wrong next time. Repeat trillions of times.")

            Card(title: "The three stages", subtitle: "Click any stage to jump to its chapter") {
                FlowRow(steps: [
                    .init(title: "Raw text", detail: "web pages, books, code", color: .gray, chapter: .pretraining),
                    .init(title: "Tokens", detail: "text → numbers", color: .blue, chapter: .tokens),
                    .init(title: "Pretraining", detail: "predict the next token\non trillions of tokens", color: .orange, chapter: .pretraining),
                    .init(title: "Base model", detail: "a brilliant autocomplete", color: .orange, chapter: .nextToken),
                    .init(title: "SFT", detail: "learn the assistant format\nfrom examples", color: .pink, chapter: .sft),
                    .init(title: "Preference tuning", detail: "RLHF / DPO: learn\nwhat people prefer", color: .pink, chapter: .preferences),
                    .init(title: "Assistant", detail: "helpful, honest,\nharmless", color: .purple, chapter: .pipeline),
                ])
            }

            HStack(alignment: .top, spacing: 20) {
                Card(title: "The one loop inside every stage", subtitle: "Click a step to learn how it works") {
                    LoopCircle()
                        .frame(height: 440)
                        .frame(maxWidth: .infinity)
                }
                Card(title: "What changes between stages?") {
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 10) {
                        GridRow {
                            Text("").font(.caption)
                            Text("Data").font(.caption.weight(.semibold))
                            Text("What counts as 'right'").font(.caption.weight(.semibold))
                        }
                        Divider().gridCellUnsizedAxes(.horizontal)
                        row("Pretraining", "Trillions of tokens of raw text", "The actual next token", .orange)
                        row("SFT", "~10k–1M written conversations", "The demo answer's tokens (only the answer)", .pink)
                        row("RLHF", "Prompts + a reward model", "Whatever scores high reward, without drifting far", .pink)
                        row("DPO", "Pairs: better vs worse answer", "Raise the better one relative to the worse", .pink)
                    }
                    .font(.callout)
                    Text("The loop — forward, loss, backward, update — is identical in all four. Only the data and the definition of 'right' change.")
                        .font(.callout).foregroundStyle(.secondary).padding(.top, 6)
                }
            }

            KeyIdea("Learn the loop once and you understand every stage. Chapters 2–5 build the model, 6–9 show how it learns, 10–12 show how that loop is used at scale and then reshaped into an assistant.")

            Card(title: "There's a real model inside this app") {
                Prose("Most pictures in this app are not drawings — they are computed live by a tiny GPT (a few tens of thousands of weights) that runs on this Mac. It starts out knowing nothing. In **Chapter 9** you can train it and watch it learn English-like sentences in about a minute. Once trained, the tokens, embeddings, attention and predictions in the other chapters all come from it.")
            }
        }
    }

    @ViewBuilder
    private func row(_ stage: String, _ data: String, _ target: String, _ color: Color) -> some View {
        GridRow {
            Text(stage).fontWeight(.semibold).foregroundStyle(color)
            Text(data)
            Text(target)
        }
    }
}

/// Five steps of the training loop on a circle, with a pulse travelling round.
struct LoopCircle: View {
    @Environment(AppState.self) private var app
    private let targets: [LoopPhase: Chapter] = [
        .batch: .tokens, .forward: .attention, .loss: .nextToken, .backward: .backprop, .update: .optimizer,
    ]

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1.1)) { ctx in
            let active = Int(ctx.date.timeIntervalSinceReferenceDate / 1.1) % LoopPhase.allCases.count
            GeometryReader { geo in
                let c = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                let r = min(geo.size.width / 2 - 72, geo.size.height / 2 - 48)
                ZStack {
                    Circle()
                        .stroke(Color.teal.opacity(0.25), style: StrokeStyle(lineWidth: 3, dash: [6, 6]))
                        .frame(width: 2 * r, height: 2 * r)
                        .position(c)
                    VStack(spacing: 2) {
                        Text("repeat").font(.caption).foregroundStyle(.secondary)
                        Text("millions of times").font(.caption.weight(.semibold))
                    }
                    .position(c)
                    ForEach(LoopPhase.allCases) { p in
                        let angle = -Double.pi / 2 + Double(p.rawValue) * 2 * .pi / 5
                        let pt = CGPoint(x: c.x + r * cos(angle), y: c.y + r * sin(angle))
                        let on = p.rawValue == active
                        Button { if let t = targets[p] { app.selected = t } } label: {
                            VStack(spacing: 3) {
                                Text("\(p.rawValue + 1)").font(.caption2.weight(.bold))
                                    .frame(width: 18, height: 18)
                                    .background(Circle().fill(on ? Color.teal : Color.secondary.opacity(0.3)))
                                    .foregroundStyle(.white)
                                Text(p.title).font(.system(size: 11, weight: on ? .semibold : .regular))
                                    .multilineTextAlignment(.center)
                                Text(p.code).font(.system(size: 9.5, design: .monospaced)).foregroundStyle(.secondary)
                            }
                            .padding(6)
                            .frame(width: 138)
                            .background(RoundedRectangle(cornerRadius: 8).fill(on ? Color.teal.opacity(0.18) : Color(nsColor: .windowBackgroundColor)))
                            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(on ? Color.teal : Color.primary.opacity(0.1)))
                            .scaleEffect(on ? 1.06 : 1)
                            .animation(.spring(duration: 0.4), value: on)
                        }
                        .buttonStyle(.plain)
                        .position(pt)
                    }
                }
            }
        }
    }
}
