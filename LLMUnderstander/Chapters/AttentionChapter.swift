import SwiftUI
import Charts

struct AttentionChapter: View {
    @Environment(AppState.self) private var app
    @State private var text = "the farmer brings a warm cake."
    @State private var layer = 0
    @State private var head = 0
    @State private var query: Int = 10
    @State private var sharpness = 1.0

    private let tok = CharTokenizer.shared

    var body: some View {
        let model = app.trainer.snapshot
        let _ = app.trainer.snapshotVersion
        let ids = Array(tok.encode(tok.clean(text)).prefix(model.config.context))

        ChapterPage(chapter: .attention) {
            ModelStatusBanner()
            Prose("After embedding, each token is a vector that only knows about itself. But to predict what comes after \"the farmer brings a warm …\", the last position needs information from *earlier* tokens. **Attention** is how tokens share information.")
            Prose("Every token makes three vectors from its embedding: a **query** (*what am I looking for?*), a **key** (*what do I contain?*) and a **value** (*what will I hand over if picked?*). Each token compares its query with the keys of every earlier token. Strong matches get a bigger share of those tokens' values.")

            if ids.count < 2 {
                TextField("Text", text: $text).textFieldStyle(.roundedBorder)
                Text("Type at least two characters.").foregroundStyle(.secondary)
            } else {
                let trace = model.trace(ids)
                let q = min(query, ids.count - 1)
                let labels = ids.map { tok.display($0) }

                Card(title: "Attention patterns inside the tiny model", subtitle: "Row = the token doing the looking. Column = the token being looked at. Darker = more attention. Click a row.") {
                    HStack {
                        TextField("Text", text: $text).textFieldStyle(.roundedBorder).font(.system(.body, design: .monospaced))
                        Picker("Layer", selection: $layer) { ForEach(0..<model.config.nLayers, id: \.self) { Text("Layer \($0 + 1)").tag($0) } }
                            .frame(width: 140)
                        Picker("Head", selection: $head) { ForEach(0..<model.config.nHeads, id: \.self) { Text("Head \($0 + 1)").tag($0) } }
                            .frame(width: 130)
                    }
                    ScrollView(.horizontal) {
                        Heatmap(values: trace.attention[layer][head], rowLabels: labels, colLabels: labels,
                                cell: 19, highlightRow: q) { query = $0 }
                            .padding(4)
                    }
                    Text("The empty upper-right triangle is the **causal mask**: a token may only look backwards. Otherwise, during training, it could simply read the answer it is supposed to predict.")
                        .font(.callout).foregroundStyle(.secondary)
                }

                Card(title: "One row, step by step", subtitle: "The token \"\(labels[q])\" at position \(q) decides where to look") {
                    let raw = trace.scores[layer][head][q]
                    let visible = Array(raw.prefix(q + 1)).map { $0 * sharpness }
                    let weights = Num.softmax(visible)
                    HStack(alignment: .top, spacing: 24) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("① Scores: query · key ÷ √\(model.config.headDim)").font(.subheadline.weight(.semibold))
                            Chart(Array(visible.enumerated()), id: \.offset) { j, s in
                                BarMark(x: .value("Token", "\(j):\(labels[j])"), y: .value("Score", s))
                                    .foregroundStyle(s >= 0 ? Color.teal : Color.gray)
                            }
                            .chartXAxis { AxisMarks { _ in AxisValueLabel().font(.system(size: 9, design: .monospaced)) } }
                            .frame(height: 170)
                            Text("Any number, positive or negative. Future tokens (after position \(q)) are masked out.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        VStack(alignment: .leading, spacing: 6) {
                            Text("② Softmax → weights that sum to 1").font(.subheadline.weight(.semibold))
                            Chart(Array(weights.enumerated()), id: \.offset) { j, w in
                                BarMark(x: .value("Token", "\(j):\(labels[j])"), y: .value("Weight", w))
                                    .foregroundStyle(Color.blue)
                            }
                            .chartYScale(domain: 0...1)
                            .chartXAxis { AxisMarks { _ in AxisValueLabel().font(.system(size: 9, design: .monospaced)) } }
                            .frame(height: 170)
                            Text("③ Output = \(weights.enumerated().sorted { $0.1 > $1.1 }.prefix(3).map { String(format: "%.0f%%", $0.1 * 100) + " of \"\(labels[$0.0])\"" }.joined(separator: " + ")) + … (their value vectors)")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    HStack {
                        Text("What if the scores were bigger? ×\(String(format: "%.1f", sharpness))").font(.callout).frame(width: 260, alignment: .leading)
                        Slider(value: $sharpness, in: 0...5)
                        Button("Reset") { sharpness = 1 }
                    }
                    Text("Bigger scores make softmax pick one token decisively; scores near zero spread attention evenly. That's why scores are divided by √(head size) — so they don't get too big just because vectors are long.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack(spacing: 6) {
                        Text("Pick a position:").font(.caption)
                        ForEach(Array(labels.enumerated()), id: \.offset) { j, l in
                            Button(l) { query = j }
                                .buttonStyle(.bordered)
                                .controlSize(.mini)
                                .tint(j == q ? .orange : .gray)
                        }
                    }
                }
            }

            KeyIdea("Attention is a soft, learned lookup: each token asks a question, every earlier token advertises what it has, and information flows to wherever the match is strongest. Different heads learn to ask different questions.")

            GoDeeper(title: "the formula and its cost") {
                Formula("Q = X·Wq    K = X·Wk    V = X·Wv\nAttention(Q, K, V) = softmax( Q·Kᵀ / √d_head  +  mask ) · V")
                Prose("**Multi-head:** the vectors are split into *h* smaller pieces, attention runs on each independently, and the results are concatenated and mixed by one more matrix (W_o). This model has \(app.trainer.config.nHeads) heads per layer; large models have 32–128.")
                Prose("**Cost:** every token compares with every earlier token, so the work grows with the *square* of the context length. This is why long context is expensive, and why tricks like FlashAttention (computing attention in tiles that fit in fast on-chip memory) matter.")
                Prose("**KV cache:** when generating text one token at a time, the keys and values of earlier tokens don't change, so they're stored and reused rather than recomputed.")
            }
        }
    }
}
