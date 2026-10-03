import SwiftUI

struct PipelineChapter: View {
    @Environment(AppState.self) private var app

    private struct Stage: Identifiable {
        let id = UUID()
        let name: String
        let color: Color
        let data: String
        let objective: String
        let loss: String
        let learns: String
        let scale: String
        let chapters: [Chapter]
    }

    private let stages: [Stage] = [
        Stage(name: "0 · Tokenizer", color: .blue,
              data: "A sample of the training text",
              objective: "Find frequent chunks (BPE merges)",
              loss: "— (not gradient-based)",
              learns: "How to turn any text into token IDs and back",
              scale: "Minutes on CPUs", chapters: [.tokens]),
        Stage(name: "1 · Pretraining", color: .orange,
              data: "Trillions of tokens of web, code, books, papers",
              objective: "Predict the next token, everywhere",
              loss: "loss = cross_entropy(logits, next_tokens)",
              learns: "Language, facts, skills, reasoning patterns → a base model",
              scale: "The overwhelming majority of total compute: weeks–months on thousands of GPUs",
              chapters: [.embeddings, .attention, .block, .nextToken, .backprop, .optimizer, .trainingLoop, .pretraining]),
        Stage(name: "2 · Supervised fine-tuning", color: .pink,
              data: "Thousands to millions of example conversations",
              objective: "Imitate good assistant answers",
              loss: "loss = cross_entropy(logits, next_tokens, mask = assistant_tokens)",
              learns: "Conversation format, following instructions, when to stop",
              scale: "Small: hours to days", chapters: [.sft]),
        Stage(name: "3 · Preference tuning", color: .pink,
              data: "Comparisons (A better than B), or a reward model, or checkable tasks",
              objective: "Produce answers people (or tests) prefer, without drifting far from SFT",
              loss: "RLHF: −(reward − β·KL)    DPO: −log σ(β·Δ log-ratio)",
              learns: "Helpfulness, honesty, harmlessness, style; with verifiable rewards, better reasoning",
              scale: "Small to growing: days to weeks", chapters: [.preferences]),
        Stage(name: "4 · Assistant", color: .purple,
              data: "Your prompt",
              objective: "— (inference: no learning)",
              loss: "next = sample(softmax(model(tokens) / temperature))",
              learns: "Nothing. The weights are frozen; it predicts one token at a time",
              scale: "Every message you send", chapters: [.nextToken]),
    ]

    var body: some View {
        ChapterPage(chapter: .pipeline) {
            Prose("Here is the whole journey on one page. Read down the **loss** column: the loop never changes. Only the data and the definition of a good answer do.")

            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(stages.enumerated()), id: \.element.id) { i, s in
                    HStack(alignment: .top, spacing: 16) {
                        VStack(spacing: 0) {
                            Circle().fill(s.color).frame(width: 18, height: 18)
                            if i < stages.count - 1 {
                                Rectangle().fill(s.color.opacity(0.4)).frame(width: 3).frame(maxHeight: .infinity)
                            }
                        }
                        .frame(width: 18)
                        VStack(alignment: .leading, spacing: 8) {
                            Text(s.name).font(.title3.weight(.semibold)).foregroundStyle(s.color)
                            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                                row("Data", s.data)
                                row("Goal", s.objective)
                                GridRow {
                                    Text("Loss").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                                    Text(s.loss).font(.system(size: 12.5, design: .monospaced))
                                        .padding(.horizontal, 6).padding(.vertical, 3)
                                        .background(s.color.opacity(0.1), in: RoundedRectangle(cornerRadius: 5))
                                }
                                row("Learns", s.learns)
                                row("Scale", s.scale)
                            }
                            FlowLayout(spacing: 6) {
                                ForEach(s.chapters) { c in
                                    Button("\(c.number). \(c.title)") { app.selected = c }
                                        .buttonStyle(.bordered).controlSize(.small)
                                }
                            }
                        }
                        .padding(.bottom, 22)
                    }
                }
            }

            Card(title: "The whole stack in one sentence each") {
                VStack(alignment: .leading, spacing: 8) {
                    line(.tokens, "Text is chopped into tokens and turned into ID numbers.")
                    line(.embeddings, "Each ID looks up a learned vector; position is added.")
                    line(.attention, "Tokens look back at earlier tokens and borrow information.")
                    line(.block, "Attention + MLP + shortcuts, stacked N times, ending in a score for every possible next token.")
                    line(.nextToken, "Softmax turns scores into probabilities; the loss is −log(probability of the real next token).")
                    line(.backprop, "Backprop finds every weight's gradient in one backward sweep.")
                    line(.optimizer, "AdamW nudges every weight downhill, with warm-up and decay.")
                    line(.trainingLoop, "Repeat: batch → forward → loss → backward → update.")
                    line(.pretraining, "Do it on trillions of tokens across thousands of GPUs → base model.")
                    line(.sft, "Fine-tune on example conversations, grading only the answers → assistant format.")
                    line(.preferences, "Push toward answers people prefer (RLHF or DPO), on a KL leash → aligned assistant.")
                }
            }

            KeyIdea("If you remember one thing: an LLM is next-token prediction, trained by gradient descent, with the data and the definition of 'good' changed at each stage.", color: .purple)

            GoDeeper(title: "what this app leaves out") {
                Prose("**Evaluation:** benchmarks, human ratings and red-teaming run throughout every stage to decide what to change next.")
                Prose("**Mid-training and long context:** many pipelines add stages between pretraining and SFT: high-quality data annealing, extending the context window, adding code or math.")
                Prose("**Inference engineering:** KV caching, batching many users together, quantisation (storing weights in 8 or 4 bits) and speculative decoding make serving fast and cheap.")
                Prose("**Agents:** how a trained model becomes an agent that uses tools inside a harness is its own section — Chapters 14–17.")
            }
        }
    }

    @ViewBuilder
    private func row(_ k: String, _ v: String) -> some View {
        GridRow {
            Text(k).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Text(v).font(.callout).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func line(_ c: Chapter, _ text: String) -> some View {
        Button { app.selected = c } label: {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Image(systemName: c.symbol).foregroundStyle(c.section.color).frame(width: 20)
                Text(c.title).font(.callout.weight(.semibold)).frame(width: 190, alignment: .leading)
                Text(text).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.leading)
                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
