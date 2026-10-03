import SwiftUI
import Charts

struct BlockChapter: View {
    @Environment(AppState.self) private var app
    @State private var part: Part = .attention
    @State private var layers = 12.0
    @State private var width = 768.0
    @State private var vocab = 50257.0

    enum Part: String, CaseIterable, Identifiable {
        case input, ln1, attention, add1, ln2, mlp, add2, head
        var id: String { rawValue }
        var title: String {
            switch self {
            case .input: "Input vectors"
            case .ln1: "LayerNorm"
            case .attention: "Multi-head attention"
            case .add1: "Add (residual)"
            case .ln2: "LayerNorm"
            case .mlp: "MLP (feed-forward)"
            case .add2: "Add (residual)"
            case .head: "Final LayerNorm + output head"
            }
        }
        var shape: String {
            switch self {
            case .input: "T × d"
            case .ln1, .ln2: "T × d → T × d"
            case .attention: "T × d → T × d"
            case .add1, .add2: "T × d + T × d"
            case .mlp: "T × d → T × 4d → T × d"
            case .head: "T × d → T × vocab"
            }
        }
        var color: Color {
            switch self {
            case .input: .gray
            case .ln1, .ln2: .secondary
            case .attention: .blue
            case .add1, .add2: .green
            case .mlp: .orange
            case .head: .purple
            }
        }
        var explanation: LocalizedStringKey {
            switch self {
            case .input: "One vector per token (embedding + position). This running vector is called the **residual stream** — every layer reads from it and writes back into it."
            case .ln1, .ln2: "Rescales each token's vector to mean 0 and spread 1 (then applies a learned gain and shift). It keeps numbers in a healthy range so deep stacks train stably."
            case .attention: "Tokens exchange information (Chapter 4). This is the *only* place where one position can see another."
            case .add1, .add2: "The layer's output is **added** to its input instead of replacing it. Information can flow straight through, and during training gradients can flow straight back. Without these shortcuts, deep networks barely train."
            case .mlp: "A small two-layer network applied to each token separately: widen to 4×, apply GELU, shrink back. Attention *moves* information between tokens; the MLP *processes* it. Most of a model's stored knowledge is thought to live here."
            case .head: "After the last block, one matrix turns each vector into a score (a *logit*) for every token in the vocabulary. Softmax turns scores into probabilities (Chapter 6)."
            }
        }
    }

    var body: some View {
        let model = app.trainer.snapshot
        let _ = app.trainer.snapshotVersion

        ChapterPage(chapter: .block) {
            Prose("A transformer is a stack of identical **blocks**. Each block does two things: let tokens talk to each other (**attention**), then let each token think on its own (**MLP**). Around both are **shortcuts** that add the result back onto the input.")

            Card(title: "Inside one block", subtitle: "Click a part") {
                HStack(alignment: .top, spacing: 28) {
                    blockDiagram

                    VStack(alignment: .leading, spacing: 10) {
                        Text(part.title).font(.title3.weight(.semibold))
                        Text("Shape: \(part.shape)").font(.system(.callout, design: .monospaced)).foregroundStyle(.secondary)
                        Text(part.explanation).font(.system(size: 15)).lineSpacing(3).fixedSize(horizontal: false, vertical: true)
                        if part == .add1 || part == .add2 || part == .input {
                            Formula("x = x + Attention(LayerNorm(x))\nx = x + MLP(LayerNorm(x))")
                        }
                        if part == .mlp { Formula("MLP(x) = GELU(x·W₁ + b₁)·W₂ + b₂") }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            Card(title: "The residual stream in our tiny model", subtitle: "Average size of each token's vector after each block, for the sentence \"the cat sees the moon.\"") {
                let ids = CharTokenizer.shared.encode("the cat sees the moon.")
                let trace = model.trace(ids)
                let embRMS = trace.embeddings.map { r in (r.reduce(0) { $0 + $1 * $1 } / Double(r.count)).squareRoot() }
                let stages = [("embedding", embRMS)] + trace.residualRMS.enumerated().map { ("after block \($0.offset + 1)", $0.element) }
                Chart(stages, id: \.0) { s in
                    BarMark(x: .value("Stage", s.0), y: .value("Size", s.1.reduce(0, +) / Double(s.1.count)))
                        .foregroundStyle(Color.green.gradient)
                }
                .frame(height: 160)
                Text("Each block adds its contribution on top of what was already there, so the stream accumulates information as it flows up through the layers.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Card(title: "How many weights?", subtitle: "Standard GPT layout: about 12·d² per block, plus the embedding table") {
                HStack(spacing: 8) {
                    preset("Our tiny model", model.config.nLayers, model.config.dModel, model.config.vocab)
                    preset("GPT-2 small", 12, 768, 50257)
                    preset("GPT-2 XL", 48, 1600, 50257)
                    preset("GPT-3", 96, 12288, 50257)
                    preset("8B-class", 32, 4096, 128256)
                }
                slider("Blocks (N)", $layers, 1...128, "%.0f")
                slider("Width (d)", $width, 16...16384, "%.0f")
                slider("Vocabulary", $vocab, 20...256000, "%.0f")
                let p = ParamCount.estimate(layers: Int(layers), dModel: Int(width), vocab: Int(vocab))
                HStack(spacing: 12) {
                    StatTile(label: "Parameters", value: Format.big(p), color: .orange)
                    StatTile(label: "Memory at 2 bytes each", value: Format.big(p * 2) + "B", color: .blue)
                    StatTile(label: "Share in blocks", value: String(format: "%.0f%%", 100 * 12 * layers * width * width / p), color: .green)
                }
                Text("Our model's exact count is \(model.parameterCount.formatted()) (it also has biases and LayerNorm weights, and a separate output head).")
                    .font(.caption).foregroundStyle(.secondary)
            }

            KeyIdea("A transformer = embed → (attention + MLP, with shortcuts) × N → scores for the next token. Scaling up mostly means a wider stream (d) and more blocks (N); the recipe stays the same.")

            GoDeeper(title: "LayerNorm and modern tweaks") {
                Formula("LayerNorm(x) = γ · (x − mean(x)) / √(var(x) + ε) + β")
                Prose("**Pre-norm** (normalise *before* each sub-layer, as here and in GPT-2 onward) trains far more stably than the original 2017 post-norm design.")
                Prose("Modern models often use **RMSNorm** (skip the mean subtraction), **SwiGLU** (a gated MLP instead of GELU) and **grouped-query attention** (several query heads share one key/value head, to shrink the KV cache). Same skeleton, better parts.")
            }
        }
    }

    // Fixed geometry so the residual shortcuts can be drawn exactly.
    private static let boxW: CGFloat = 240, arrowH: CGFloat = 18, dividerH: CGFloat = 26, gutter: CGFloat = 34
    private static func boxH(_ p: Part) -> CGFloat { p == .attention || p == .mlp ? 46 : 30 }

    /// y of the top of each box, plus the total height.
    private static var layout: (tops: [Part: CGFloat], height: CGFloat) {
        var y: CGFloat = 0
        var tops: [Part: CGFloat] = [:]
        for p in Part.allCases {
            if p == .head { y += dividerH }
            tops[p] = y
            y += boxH(p)
            if p != .head { y += arrowH }
        }
        return (tops, y)
    }

    private var blockDiagram: some View {
        let (tops, height) = Self.layout
        let x0 = Self.gutter
        return ZStack(alignment: .topLeading) {
            // residual shortcuts: branch off the stream, skip the sub-layer, re-enter at the Add
            Path { path in
                for (from, to) in [(Part.input, Part.add1), (Part.add1, Part.add2)] {
                    let startY = tops[from]! + Self.boxH(from) + Self.arrowH / 2
                    let endY = tops[to]! + Self.boxH(to) / 2
                    path.move(to: CGPoint(x: x0 + Self.boxW / 2, y: startY))
                    path.addLine(to: CGPoint(x: 10, y: startY))
                    path.addLine(to: CGPoint(x: 10, y: endY))
                    path.addLine(to: CGPoint(x: x0 - 2, y: endY))
                    path.move(to: CGPoint(x: x0 - 8, y: endY - 5)); path.addLine(to: CGPoint(x: x0 - 2, y: endY)); path.addLine(to: CGPoint(x: x0 - 8, y: endY + 5))
                }
            }
            .stroke(Color.green, style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round, dash: [5, 3]))

            ForEach(Part.allCases) { p in
                Button { part = p } label: {
                    HStack {
                        if p == .add1 || p == .add2 { Image(systemName: "plus.circle.fill") }
                        Text(p.title).font(.system(size: 13, weight: .semibold))
                    }
                    .frame(width: Self.boxW, height: Self.boxH(p))
                    .background(p.color.opacity(part == p ? 0.35 : 0.13), in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(p.color.opacity(part == p ? 1 : 0.3), lineWidth: part == p ? 2 : 1))
                }
                .buttonStyle(.plain)
                .offset(x: x0, y: tops[p]!)
                if p != .head {
                    Image(systemName: "arrow.down").font(.caption2).foregroundStyle(.secondary)
                        .frame(width: Self.boxW, height: Self.arrowH)
                        .offset(x: x0, y: tops[p]! + Self.boxH(p))
                }
            }
            HStack { Rectangle().frame(height: 1); Text("× N blocks, then:").font(.caption2); Rectangle().frame(height: 1) }
                .foregroundStyle(.secondary)
                .frame(width: Self.boxW + x0, height: Self.dividerH)
                .offset(x: 0, y: tops[.head]! - Self.dividerH)
        }
        .frame(width: Self.boxW + x0, height: height, alignment: .topLeading)
    }

    private func preset(_ name: String, _ l: Int, _ d: Int, _ v: Int) -> some View {
        Button(name) { withAnimation { layers = Double(l); width = Double(d); vocab = Double(v) } }
            .buttonStyle(.bordered)
    }

    private func slider(_ label: String, _ v: Binding<Double>, _ r: ClosedRange<Double>, _ f: String) -> some View {
        HStack {
            Text(label).frame(width: 110, alignment: .leading)
            Slider(value: v, in: r)
            Text(String(format: f, v.wrappedValue)).monospacedDigit().frame(width: 70, alignment: .trailing)
        }
        .font(.callout)
    }
}
