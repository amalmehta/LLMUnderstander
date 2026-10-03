import SwiftUI
import Charts

struct EmbeddingsChapter: View {
    @Environment(AppState.self) private var app
    @State private var text = "the cat"
    @State private var picked: Int?

    private let tok = CharTokenizer.shared

    var body: some View {
        let model = app.trainer.snapshot
        let _ = app.trainer.snapshotVersion
        let ids = Array(tok.encode(tok.clean(text)).prefix(model.config.context))
        let d = model.config.dModel

        ChapterPage(chapter: .embeddings) {
            ModelStatusBanner()
            Prose("A token ID like 19 is just a name — the number itself means nothing. So the first layer of the model is a **lookup table**: one row of numbers per token. Token 19 → row 19. That row is the token's **embedding**, a list of \(d) numbers here (thousands in real models).")
            Prose("Nobody writes these numbers. They start random and are **learned** by training, like every other weight. Tokens that behave similarly end up with similar rows.")

            Card(title: "Lookup: token embedding + position embedding", subtitle: "Each row is one token; each column one of the \(d) dimensions. Red = positive, blue = negative.") {
                TextField("Text", text: $text).textFieldStyle(.roundedBorder).font(.system(.body, design: .monospaced)).frame(maxWidth: 300)
                if ids.isEmpty {
                    Text("Type some lowercase letters.").foregroundStyle(.secondary)
                } else {
                    let labels = ids.map { tok.display($0) }
                    let tokE = ids.map { model.wte.row($0) }
                    let posE = (0..<ids.count).map { model.wpe.row($0) }
                    let sum = zip(tokE, posE).map { zip($0, $1).map(+) }
                    ScrollView(.horizontal) {
                        HStack(alignment: .top, spacing: 14) {
                            labelled("what token is it?", Heatmap(values: tokE, rowLabels: labels, diverging: true, cell: 11))
                            Text("+").font(.title).padding(.top, 30)
                            labelled("where is it?", Heatmap(values: posE, rowLabels: (0..<ids.count).map { "\($0)" }, diverging: true, cell: 11))
                            Text("=").font(.title).padding(.top, 30)
                            labelled("what the transformer receives", Heatmap(values: sum, rowLabels: labels, diverging: true, cell: 11))
                        }
                    }
                }
                Text("Attention (next chapter) has no built-in sense of order, so a **position embedding** is added. Without it, \"dog bites man\" and \"man bites dog\" would look identical.")
                    .font(.callout).foregroundStyle(.secondary)
            }

            Card(title: "The embedding map", subtitle: "All \(tok.size) token embeddings squashed from \(d) dimensions to 2 (PCA). Click a token to find its nearest neighbours.") {
                let pts = Self.project(model.wte)
                HStack(alignment: .top, spacing: 20) {
                    Chart(pts) { p in
                        PointMark(x: .value("PC1", p.x), y: .value("PC2", p.y))
                            .foregroundStyle(by: .value("Kind", p.kind))
                            .symbolSize(p.id == picked ? 700 : 380)
                            .annotation(position: .overlay, alignment: .center) {
                                Text(p.label).font(.system(size: 12, weight: .bold, design: .monospaced)).foregroundStyle(.white)
                            }
                    }
                    .chartForegroundStyleScale(["vowel": Color.pink, "consonant": Color.blue, "space/punctuation": Color.gray])
                    .chartXAxis(.hidden).chartYAxis(.hidden)
                    .chartOverlay { proxy in
                        GeometryReader { geo in
                            Rectangle().fill(.clear).contentShape(Rectangle())
                                .onTapGesture { loc in
                                    let frame = geo[proxy.plotFrame!]
                                    let x = loc.x - frame.origin.x, y = loc.y - frame.origin.y
                                    picked = pts.min { a, b in
                                        dist(proxy, a, x, y) < dist(proxy, b, x, y)
                                    }?.id
                                }
                        }
                    }
                    .frame(height: 340)

                    VStack(alignment: .leading, spacing: 8) {
                        if let picked {
                            Text("Closest to \(tok.display(picked))").font(.headline)
                            ForEach(Self.neighbours(of: picked, in: model.wte).prefix(6), id: \.0) { id, sim in
                                HStack {
                                    TokenChip(text: tok.display(id), color: Palette.token(id))
                                    ProgressView(value: max(0, sim)).frame(width: 90)
                                    Text(String(format: "%.2f", sim)).font(.caption).monospacedDigit()
                                }
                            }
                            Text("cosine similarity of the full \(d)-number rows").font(.caption2).foregroundStyle(.secondary)
                        } else {
                            Text("Click a dot").foregroundStyle(.secondary)
                        }
                    }
                    .frame(width: 210, alignment: .leading)
                }
                Text(app.trainer.isTrained
                     ? "After training, letters that play similar roles tend to drift together — vowels often cluster, and space and full stop sit apart."
                     : "Untrained: the dots are randomly scattered. Train the model (Chapter 9) and come back — the map reorganises itself.")
                    .font(.callout).foregroundStyle(.secondary)
            }

            KeyIdea("An embedding turns a meaningless ID into a point in space. Training moves the points so that their positions carry meaning the rest of the network can use.")

            GoDeeper(title: "sizes and modern variants") {
                Prose("The table has **vocabulary × width** numbers. GPT-2 small: 50,257 × 768 ≈ 39M weights — about a third of the whole model. GPT-3: 12,288 dimensions per token.")
                Prose("Many models **tie** the input table to the output layer (the same matrix turns vectors back into token scores at the end).")
                Prose("Modern models mostly replace learned position embeddings with **RoPE** (rotary position embeddings), which rotates the query and key vectors by an angle proportional to position. Relative distance then appears directly in the attention scores.")
                Formula("x₀[t] = TokenEmbedding[id_t] + PositionEmbedding[t]      shape: (T × d)")
            }
        }
    }

    private func labelled(_ title: String, _ v: some View) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            v
        }
    }

    private func dist(_ proxy: ChartProxy, _ p: EmbPoint, _ x: CGFloat, _ y: CGFloat) -> CGFloat {
        guard let px = proxy.position(forX: p.x), let py = proxy.position(forY: p.y) else { return .infinity }
        return hypot(px - x, py - y)
    }

    struct EmbPoint: Identifiable { let id: Int; let x: Double; let y: Double; let label: String; let kind: String }

    /// Principal-component projection of each row to 2-D (power iteration with deflation).
    static func project(_ table: Tensor) -> [EmbPoint] {
        let n = table.rows, d = table.cols
        var rows = (0..<n).map { table.row($0) }
        let mean = (0..<d).map { j in rows.reduce(0) { $0 + $1[j] } / Double(n) }
        rows = rows.map { zip($0, mean).map(-) }
        var cov = [[Double]](repeating: [Double](repeating: 0, count: d), count: d)
        for r in rows { for i in 0..<d { for j in 0..<d { cov[i][j] += r[i] * r[j] } } }
        func top(_ m: [[Double]]) -> [Double] {
            var v = (0..<d).map { Double($0 % 3) - 1 + 0.1 }
            for _ in 0..<200 {
                var w = (0..<d).map { i in (0..<d).reduce(0) { $0 + m[i][$1] * v[$1] } }
                let norm = max(w.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1e-12)
                w = w.map { $0 / norm }
                v = w
            }
            return v
        }
        let v1 = top(cov)
        let lambda = (0..<d).reduce(0.0) { acc, i in acc + v1[i] * (0..<d).reduce(0) { $0 + cov[i][$1] * v1[$1] } }
        let cov2 = (0..<d).map { i in (0..<d).map { j in cov[i][j] - lambda * v1[i] * v1[j] } }
        let v2 = top(cov2)
        let tok = CharTokenizer.shared
        return rows.enumerated().map { i, r in
            let c = tok.vocab[i]
            let kind = "aeiou".contains(c) ? "vowel" : (c.isLetter ? "consonant" : "space/punctuation")
            return EmbPoint(id: i, x: zip(r, v1).reduce(0) { $0 + $1.0 * $1.1 },
                            y: zip(r, v2).reduce(0) { $0 + $1.0 * $1.1 }, label: tok.display(i), kind: kind)
        }
    }

    static func neighbours(of id: Int, in table: Tensor) -> [(Int, Double)] {
        let a = table.row(id)
        func cos(_ x: [Double], _ y: [Double]) -> Double {
            let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
            return dot / max(1e-12, (x.reduce(0) { $0 + $1 * $1 } * y.reduce(0) { $0 + $1 * $1 }).squareRoot())
        }
        return (0..<table.rows).filter { $0 != id }.map { ($0, cos(a, table.row($0))) }.sorted { $0.1 > $1.1 }
    }
}
