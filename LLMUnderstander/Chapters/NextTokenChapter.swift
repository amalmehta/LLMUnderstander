import SwiftUI
import Charts

struct NextTokenChapter: View {
    @Environment(AppState.self) private var app
    @State private var prompt = "the cat "
    @State private var temperature = 1.0
    @State private var actual: Int = CharTokenizer.shared.encode("s").first ?? 0
    @State private var rng = SplitMix64(seed: 11)
    @State private var sentence = "the queen chases the old boat at night."

    private let tok = CharTokenizer.shared

    var body: some View {
        let model = app.trainer.snapshot
        let _ = app.trainer.snapshotVersion
        let ids = tok.encode(tok.clean(prompt))
        let probs = model.nextTokenProbs(ids, temperature: temperature)
        let rawProbs = model.nextTokenProbs(ids)

        ChapterPage(chapter: .nextToken) {
            ModelStatusBanner()
            Prose("At the top of the stack, the model outputs one score (a **logit**) per vocabulary token. **Softmax** turns the scores into probabilities that add up to 100%. That list *is* the model's prediction: not one answer, but a bet spread across every possible next token.")

            Card(title: "Ask the model what comes next", subtitle: "Real probabilities from the tiny model") {
                HStack {
                    TextField("Prompt", text: $prompt).textFieldStyle(.roundedBorder).font(.system(.body, design: .monospaced))
                    Button("Sample next") {
                        let i = Num.sample(probs, rng: &rng)
                        prompt += String(tok.vocab[i])
                    }
                    .buttonStyle(.borderedProminent)
                    Button("Most likely") {
                        if let i = probs.indices.max(by: { probs[$0] < probs[$1] }) { prompt += String(tok.vocab[i]) }
                    }
                    Button("Reset") { prompt = "the cat " }
                }
                let top = probs.enumerated().sorted { $0.element > $1.element }.prefix(14)
                ProbabilityBars(items: top.map { .init(id: $0.offset, label: tok.display($0.offset), p: $0.element) }, color: .teal)
                HStack {
                    Text("Temperature \(String(format: "%.2f", temperature))").monospacedDigit().frame(width: 140, alignment: .leading)
                    Slider(value: $temperature, in: 0.1...2.5)
                }
                .font(.callout)
                Text("Generation is just this, in a loop: predict, pick one token, append it, predict again. **Temperature** divides the scores before softmax — low = safe and repetitive, high = creative and chaotic.")
                    .font(.callout).foregroundStyle(.secondary)
            }

            Card(title: "Scoring a guess: the loss", subtitle: "Training needs one number that says how wrong the prediction was") {
                HStack(alignment: .top, spacing: 24) {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Suppose the real next character was")
                            Picker("", selection: $actual) {
                                ForEach(0..<tok.size, id: \.self) { Text(tok.display($0)).tag($0) }
                            }
                            .labelsHidden().frame(width: 70)
                        }
                        let p = rawProbs[actual]
                        HStack(spacing: 12) {
                            StatTile(label: "Model gave it", value: String(format: "%.1f%%", p * 100), color: .teal)
                            StatTile(label: "Loss = −log(p)", value: String(format: "%.2f", -log(p)), color: .orange)
                        }
                        Text("If the model gave the right answer 100%, the loss is 0. At 50% it's 0.69. At 1% it's 4.6. Confident *and* wrong is punished hardest.")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: 380, alignment: .leading)
                    let curve = stride(from: 0.005, through: 1.0, by: 0.005).map { Point(x: $0, y: -log($0)) }
                    Chart {
                        ForEach(curve) { LineMark(x: .value("p", $0.x), y: .value("loss", $0.y)) }
                        PointMark(x: .value("p", rawProbs[actual]), y: .value("loss", -log(rawProbs[actual])))
                            .foregroundStyle(.orange).symbolSize(150)
                    }
                    .chartXAxisLabel("probability given to the right answer")
                    .chartYAxisLabel("loss")
                    .chartYScale(domain: 0...5.5)
                    .frame(height: 200)
                }
            }

            Card(title: "Loss over a whole sentence", subtitle: "Every position is a prediction. Redder = more surprised.") {
                TextField("Sentence", text: $sentence).textFieldStyle(.roundedBorder).font(.system(.body, design: .monospaced))
                let sids = Array(tok.encode(tok.clean(sentence)).prefix(model.config.context + 1))
                if sids.count >= 2 {
                    let logits = model.forward(Array(sids.dropLast()))
                    let losses = (0..<(sids.count - 1)).map { i in -log(Num.softmax(logits.row(i))[sids[i + 1]]) }
                    FlowLayout(spacing: 2) {
                        TokenChip(text: tok.display(sids[0]), color: .gray, sub: "start")
                        ForEach(Array(losses.enumerated()), id: \.offset) { i, l in
                            TokenChip(text: tok.display(sids[i + 1]), color: Color(hue: max(0, 0.33 - l / 12), saturation: 0.9, brightness: 0.9),
                                      sub: String(format: "%.1f", l))
                        }
                    }
                    HStack(spacing: 12) {
                        StatTile(label: "Average loss (what training reports)", value: String(format: "%.2f", losses.reduce(0, +) / Double(losses.count)), color: .orange)
                        StatTile(label: "Random guessing would be", value: String(format: "%.2f", log(Double(tok.size))), color: .gray)
                    }
                    Text("Notice the pattern once trained: the *first* letter of each word is hard to guess, the rest of the word is easy. A single forward pass scores every position at once — that's why training is so efficient.")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }

            KeyIdea("Pretraining has one objective: make the average −log(probability of the actual next token) as small as possible across all the training text. Everything the model 'knows' is a side effect of getting better at this.")

            GoDeeper(title: "cross-entropy, perplexity and sampling") {
                Formula("p = softmax(logits)          p_i = e^{z_i} / Σ_j e^{z_j}\nloss = −(1/T) Σ_t log p(x_{t+1} | x_1 … x_t)")
                Prose("This is the **cross-entropy** between the true next token and the model's distribution. Its exponential, **perplexity**, reads as \"how many tokens is the model choosing between\": loss 3.33 here = perplexity 28 = pure guessing among 28 characters.")
                Prose("**Teacher forcing:** during training, each position predicts its next token given the *real* previous tokens (not the model's own guesses), so all T predictions are computed in one parallel pass.")
                Prose("**Sampling tricks:** besides temperature, **top-k** keeps only the k most likely tokens and **top-p (nucleus)** keeps the smallest set whose probabilities sum to p. Both cut off the long tail of unlikely tokens.")
            }
        }
    }
}
