import SwiftUI
import Charts

struct SFTChapter: View {
    @State private var showMask = true
    @State private var logits = PostTraining.baseLogits
    @State private var lossHistory: [Point] = [Point(x: 0, y: PostTraining.sftLoss(PostTraining.baseLogits))]
    @State private var lr = 0.5

    private let template: [(String, Bool, Bool)] = [
        // (text, is special token, counts toward loss)
        ("<|system|>", true, false), ("You", false, false), (" are", false, false), (" a", false, false), (" helpful", false, false), (" assistant.", false, false),
        ("<|user|>", true, false), ("how", false, false), (" do", false, false), (" i", false, false), (" boil", false, false), (" an", false, false), (" egg?", false, false),
        ("<|assistant|>", true, false), ("Put", false, true), (" the", false, true), (" egg", false, true), (" in", false, true), (" boiling", false, true),
        (" water", false, true), (" for", false, true), (" 9", false, true), (" minutes,", false, true), (" then", false, true), (" cool", false, true),
        (" it.", false, true), ("<|end|>", true, true),
    ]

    var body: some View {
        let p = Num.softmax(logits)
        ChapterPage(chapter: .sft) {
            Prose("A base model has read the internet, but it's an autocomplete. Give it *\"how do i boil an egg?\"* and a perfectly reasonable continuation is *\"how do i fry an egg? how do i poach an egg?…\"* — that's what lists of questions on the web look like.")
            Prose("**Supervised fine-tuning (SFT)** fixes this by continuing training on a much smaller set of example conversations, written or curated by people, showing the behaviour we want. Same loop, same loss — with one twist.")

            Card(title: "Chat template and loss mask", subtitle: "Conversations are flattened into one token stream with special role tokens") {
                Toggle("Show which tokens count toward the loss", isOn: $showMask).toggleStyle(.switch)
                FlowLayout(spacing: 3) {
                    ForEach(Array(template.enumerated()), id: \.offset) { _, t in
                        Text(t.0.replacingOccurrences(of: " ", with: "␣"))
                            .font(.system(size: 13, weight: t.1 ? .bold : .regular, design: .monospaced))
                            .padding(.horizontal, 5).padding(.vertical, 3)
                            .background(chipColor(t).opacity(0.22), in: RoundedRectangle(cornerRadius: 5))
                            .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(chipColor(t).opacity(0.5)))
                            .opacity(showMask && !t.2 ? 0.45 : 1)
                    }
                }
                if showMask {
                    HStack(spacing: 16) {
                        HStack(spacing: 6) { RoundedRectangle(cornerRadius: 3).fill(Color.gray.opacity(0.4)).frame(width: 14, height: 10); Text("weight 0: the model reads these but isn't graded on them") }
                        HStack(spacing: 6) { RoundedRectangle(cornerRadius: 3).fill(Color.pink.opacity(0.5)).frame(width: 14, height: 10); Text("weight 1: the model is trained to produce these") }
                    }
                    .font(.caption).foregroundStyle(.secondary)
                }
                Text("The twist: the loss is only counted on the **assistant's** tokens. We don't want the model to learn to write user questions — only how to answer them. The `<|end|>` token teaches it when to stop talking.")
                    .font(.callout).foregroundStyle(.secondary)
            }

            Card(title: "Live: fine-tune a toy model", subtitle: "A simplified model that chooses between 8 whole answers to \"\(PostTraining.prompt)\". The update is the real cross-entropy gradient.") {
                HStack(alignment: .top, spacing: 20) {
                    VStack(alignment: .leading) {
                        Chart(PostTraining.answers) { a in
                            BarMark(x: .value("p", p[a.id]), y: .value("answer", a.short))
                                .foregroundStyle(PostTraining.demonstrations.contains(a.id) ? Color.pink : Color.gray.opacity(0.5))
                                .annotation(position: .trailing) { Text(String(format: "%.0f%%", p[a.id] * 100)).font(.caption2) }
                        }
                        .chartXScale(domain: 0...1)
                        .chartXAxis { AxisMarks(values: [0, 0.5, 1]) { v in AxisGridLine(); AxisValueLabel { Text("\(Int((v.as(Double.self) ?? 0) * 100))%") } } }
                        .frame(height: 260)
                        Text("Pink = answers that appear in the human-written demonstrations").font(.caption).foregroundStyle(.secondary)
                    }
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Button("Train 1 step") { train(1) }
                            Button("Train 25 steps") { train(25) }.buttonStyle(.borderedProminent).tint(.pink)
                            Button("Reset to base model") { logits = PostTraining.baseLogits; lossHistory = [Point(x: 0, y: PostTraining.sftLoss(logits))] }
                        }
                        HStack {
                            Text("Learning rate \(String(format: "%.2f", lr))").frame(width: 130, alignment: .leading)
                            Slider(value: $lr, in: 0.05...2)
                        }.font(.callout)
                        Chart(lossHistory) { LineMark(x: .value("step", $0.x), y: .value("loss", $0.y)).foregroundStyle(.pink) }
                            .chartXAxisLabel("SFT steps").chartYAxisLabel("SFT loss")
                            .frame(height: 130)
                        HStack(spacing: 10) {
                            StatTile(label: "SFT loss", value: String(format: "%.2f", PostTraining.sftLoss(logits)), color: .pink)
                            StatTile(label: "Chance of a good answer", value: String(format: "%.0f%%", (p[1] + p[2]) * 100), color: .green)
                        }
                    }
                }
                Text("Before SFT the toy base model mostly \"keeps listing questions\" or states an off-topic fact. A few dozen steps on the demonstrations move almost all of the probability onto assistant-style answers.")
                    .font(.callout).foregroundStyle(.secondary)
            }

            KeyIdea("SFT doesn't teach much new knowledge — the base model already has it. It teaches **format and behaviour**: you are an assistant, this is a conversation, answer the question, then stop. A surprisingly small dataset (thousands to a few million examples) is enough.")

            GoDeeper(title: "practical details") {
                Formula("loss = − Σ_t mask_t · log p(x_{t+1} | x_≤t)  /  Σ_t mask_t")
                Prose("**Data:** InstructGPT (2022) used about 13,000 human-written demonstrations. Modern pipelines mix human-written, model-generated-then-filtered, and task-specific data: coding, math, tool use, safety refusals and multi-turn chat.")
                Prose("**Learning rate** is much smaller than in pretraining, and training is short (a few epochs), to avoid *catastrophic forgetting* — overwriting what pretraining learned.")
                Prose("**LoRA** and other parameter-efficient methods freeze the big weight matrices and train small low-rank add-ons instead. The results are often almost as good, with a fraction of the memory.")
                Prose("**The limit of SFT:** it can only imitate the demonstrations. It never sees *bad* answers, so it can't learn what to avoid, and the model can never exceed its teachers. That is what preference training is for.")
            }
        }
    }

    private func chipColor(_ t: (String, Bool, Bool)) -> Color {
        if t.1 { return .purple }
        return t.2 ? .pink : .gray
    }

    private func train(_ n: Int) {
        for _ in 0..<n {
            logits = PostTraining.sftStep(logits, lr: lr)
            lossHistory.append(Point(x: Double(lossHistory.count), y: PostTraining.sftLoss(logits)))
        }
    }
}
