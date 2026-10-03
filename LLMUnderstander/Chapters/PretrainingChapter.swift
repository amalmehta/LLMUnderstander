import SwiftUI
import Charts

struct PretrainingChapter: View {
    @Environment(AppState.self) private var app
    @State private var tab = 0
    @State private var logC = 23.0
    @State private var context = 48.0
    @State private var workers = 4
    @State private var dpStage = 0

    var body: some View {
        ChapterPage(chapter: .pretraining) {
            Prose("Pretraining is the loop from Chapter 9, unchanged, scaled up by a factor of about a trillion. The model reads a large slice of the written internet and learns to predict the next token. That's where nearly all of its knowledge, grammar and reasoning ability comes from — and nearly all of the cost.")

            Picker("", selection: $tab) {
                Text("1 · The data").tag(0)
                Text("2 · Packing batches").tag(1)
                Text("3 · Scaling laws").tag(2)
                Text("4 · Thousands of GPUs").tag(3)
                Text("5 · Tiny vs real").tag(4)
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            switch tab {
            case 0: data
            case 1: packing
            case 2: scaling
            case 3: parallel
            default: comparison
            }

            KeyIdea("Pretraining = next-token prediction on trillions of tokens. The result is a **base model**: a superb autocomplete that knows a lot, but doesn't yet know it's supposed to be an assistant. Ask it a question and it may just continue with more questions.")
        }
    }

    // MARK: - Data

    private var data: some View {
        Card(title: "From the raw web to training tokens", subtitle: "A typical pipeline. Widths are illustrative of how much each step keeps.") {
            let stages: [(String, String, Double, Color)] = [
                ("Crawl", "Billions of web pages (e.g. Common Crawl), plus books, code, papers", 1.0, .gray),
                ("Extract text", "Strip HTML, menus, ads, boilerplate", 0.7, .blue),
                ("Language & quality filters", "Keep fluent, informative text; classifiers score 'educational value'", 0.35, .teal),
                ("Deduplicate", "Near-identical pages removed; repeats make models memorise", 0.2, .green),
                ("Safety & privacy", "Remove personal data and toxic content", 0.18, .orange),
                ("Mix", "Blend web + code + math + books + multilingual in chosen proportions", 0.18, .pink),
                ("Tokenize & shuffle", "Trillions of tokens, e.g. ~15T for recent open 8–400B models", 0.16, .purple),
            ]
            VStack(alignment: .leading, spacing: 8) {
                ForEach(stages, id: \.0) { s in
                    HStack(spacing: 12) {
                        Text(s.0).font(.callout.weight(.semibold)).frame(width: 190, alignment: .trailing)
                        GeometryReader { g in
                            RoundedRectangle(cornerRadius: 6).fill(s.3.opacity(0.55))
                                .frame(width: g.size.width * s.2)
                                .frame(maxWidth: .infinity)
                        }
                        .frame(height: 22)
                        Text(s.1).font(.caption).foregroundStyle(.secondary).frame(width: 280, alignment: .leading)
                    }
                }
            }
            Prose("Data quality matters as much as quantity: filtering and deduplication often improve a model more than adding raw tokens. Our tiny model's \"internet\" is \(Corpus.training.count.formatted()) characters of generated sentences.")
        }
    }

    // MARK: - Packing

    private var packing: some View {
        Card(title: "Packing documents into fixed-length rows", subtitle: "GPUs want rectangular batches: (batch size × context length)") {
            HStack {
                Text("Context length \(Int(context)) tokens").frame(width: 200, alignment: .leading)
                Slider(value: $context, in: 16...96, step: 8)
            }.font(.callout)
            let rows = Self.pack(context: Int(context))
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(rows.enumerated()), id: \.offset) { r, row in
                    HStack(spacing: 0) {
                        Text("row \(r)").font(.caption2).foregroundStyle(.secondary).frame(width: 44)
                        ForEach(Array(row.enumerated()), id: \.offset) { _, seg in
                            ZStack {
                                Rectangle().fill(seg.doc < 0 ? Color.black.opacity(0.75) : Palette.token(seg.doc).opacity(0.45))
                                Text(seg.doc < 0 ? "eot" : seg.text).font(.system(size: 9, design: .monospaced)).lineLimit(1)
                                    .foregroundStyle(seg.doc < 0 ? .white : .primary)
                            }
                            .frame(width: CGFloat(seg.length) * 9.5, height: 20)
                        }
                    }
                }
            }
            Prose("Documents are joined end to end with an **end-of-text** token between them, then cut into rows. One row can hold the end of one document and the start of the next. The end-of-text token teaches the model that what follows is unrelated. A batch of 1,024 rows × 8,192 tokens is ~8 million tokens per step.")
        }
    }

    private struct Segment { let doc: Int; let text: String; let length: Int }

    private static func pack(context: Int) -> [[Segment]] {
        var rng = SplitMix64(seed: 5)
        var stream: [(Int, Character)] = []
        for d in 0..<10 {
            let n = 1 + rng.int(3)
            let text = (0..<n).map { _ in Corpus.sentence(&rng) }.joined(separator: " ")
            stream += text.map { (d, $0) }
            stream.append((-1, "·"))
        }
        var rows: [[Segment]] = []
        var i = 0
        while i < stream.count && rows.count < 6 {
            let chunk = stream[i..<min(i + context, stream.count)]
            var segs: [Segment] = []
            var cur: (Int, String)? = nil
            for (d, c) in chunk {
                if let c0 = cur, c0.0 == d { cur!.1.append(c) } else {
                    if let c0 = cur { segs.append(Segment(doc: c0.0, text: c0.1, length: c0.1.count)) }
                    cur = (d, String(c))
                }
            }
            if let c0 = cur { segs.append(Segment(doc: c0.0, text: c0.1, length: c0.1.count)) }
            rows.append(segs)
            i += context
        }
        return rows
    }

    // MARK: - Scaling laws

    private var scaling: some View {
        let c = pow(10, logC)
        let opt = ScalingLaw.optimum(compute: c)
        let curves = [logC - 2, logC, logC + 2].map { lc in (lc, ScalingLaw.sweep(compute: pow(10, lc))) }
        return Card(title: "Scaling laws: how big a model, for how much compute?", subtitle: "Predicted final loss = E + A/N^α + B/D^β (Chinchilla form; constants from the 2024 Epoch AI replication)") {
            HStack {
                Text("Compute budget \(Format.sci(c)) FLOPs").monospacedDigit().frame(width: 260, alignment: .leading)
                Slider(value: $logC, in: 18...26)
            }.font(.callout)
            Chart {
                ForEach(curves, id: \.0) { lc, pts in
                    ForEach(pts.filter { $0.1 < 6 }, id: \.0) { p in
                        LineMark(x: .value("log10 params", p.0), y: .value("loss", p.1), series: .value("C", "\(lc)"))
                            .foregroundStyle(lc == logC ? Color.orange : Color.gray.opacity(0.35))
                            .lineStyle(StrokeStyle(lineWidth: lc == logC ? 3 : 1.5))
                    }
                }
                PointMark(x: .value("log10 params", log10(opt.params)), y: .value("loss", opt.loss))
                    .foregroundStyle(.red).symbolSize(160)
                    .annotation(position: .top) { Text("best").font(.caption.bold()).foregroundStyle(.red) }
            }
            .chartXScale(domain: 6...13)
            .chartYScale(domain: 1.8...4.5)
            .chartXAxis { AxisMarks(values: Array(stride(from: 6.0, through: 13.0, by: 1.0))) { v in
                AxisGridLine(); AxisValueLabel { Text(Format.big(pow(10, v.as(Double.self) ?? 0))) }
            } }
            .chartXAxisLabel("model size (parameters)").chartYAxisLabel("predicted loss")
            .frame(height: 260)
            HStack(spacing: 10) {
                StatTile(label: "Best model size", value: Format.big(opt.params), color: .orange)
                StatTile(label: "Train it on", value: Format.big(opt.tokens) + " tokens", color: .blue)
                StatTile(label: "Tokens per parameter", value: String(format: "%.0f", opt.tokens / opt.params), color: .green)
                StatTile(label: "Predicted loss", value: String(format: "%.2f", opt.loss), color: .red)
                StatTile(label: "≈ GPU-days (H100 @ 40%)", value: Format.big(c / (0.4 * 9.9e14) / 86400), color: .gray)
            }
            Prose("For a fixed budget, a model that's **too big** sees too little data, and one that's **too small** can't absorb what it sees. The sweet spot is about **20 tokens per parameter**. In practice, labs often train smaller models on far *more* tokens than this, because a smaller model is cheaper to serve to millions of users. Training compute ≈ **6 × parameters × tokens**.")
        }
    }

    // MARK: - Parallelism

    private var parallel: some View {
        Card(title: "Data parallelism: many GPUs, one model", subtitle: "Click through the steps of one training step on \(workers) GPUs") {
            Stepper("GPUs: \(workers)", value: $workers, in: 2...8).frame(width: 160)
            let grads: [[Double]] = (0..<workers).map { w in
                var r = SplitMix64(seed: UInt64(w + 1)); return (0..<4).map { _ in (r.normal() * 0.5 + 0.3) }
            }
            let avg = (0..<4).map { j in grads.map { $0[j] }.reduce(0, +) / Double(workers) }
            HStack(alignment: .top, spacing: 10) {
                ForEach(0..<workers, id: \.self) { w in
                    VStack(spacing: 6) {
                        Image(systemName: "cpu").font(.title2).foregroundStyle(.orange)
                        Text("GPU \(w)").font(.caption.bold())
                        Text("full copy of model").font(.system(size: 9)).foregroundStyle(.secondary)
                        Text("batch slice \(w)").font(.system(size: 10)).padding(4).background(Palette.token(w).opacity(0.3), in: Capsule())
                            .opacity(dpStage >= 0 ? 1 : 0.2)
                        gradientBars(dpStage >= 1 ? (dpStage >= 2 ? avg : grads[w]) : [0, 0, 0, 0], color: dpStage >= 2 ? .green : .red)
                    }
                    .padding(8)
                    .frame(maxWidth: .infinity)
                    .background(Color.orange.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
                }
            }
            let texts = [
                "① Each GPU holds a full copy of the weights and gets a different slice of the batch.",
                "② Each runs forward + backward on its own slice → its own gradient (different on each GPU).",
                "③ All-reduce: the GPUs average their gradients over fast links. Now every GPU has the same averaged gradient.",
                "④ Every GPU applies the same optimizer step, so all copies stay identical. Repeat.",
            ]
            Text(texts[dpStage]).font(.callout).frame(minHeight: 36, alignment: .topLeading)
            HStack {
                Button("Next step") { dpStage = (dpStage + 1) % texts.count }.buttonStyle(.borderedProminent)
                Text("\(dpStage + 1) / \(texts.count)").font(.caption).foregroundStyle(.secondary)
            }
            Divider()
            Prose("When the model itself is too large for one GPU, it gets split up too: **tensor parallelism** cuts each matrix across GPUs; **pipeline parallelism** puts different blocks on different GPUs; **FSDP/ZeRO** shards weights, gradients and optimizer state and gathers them just in time. Big runs combine all of these across tens of thousands of GPUs, train in **bf16** (16-bit) arithmetic to save memory and time, and save checkpoints often, because hardware fails mid-run.")
        }
    }

    private func gradientBars(_ g: [Double], color: Color) -> some View {
        HStack(alignment: .bottom, spacing: 3) {
            ForEach(0..<g.count, id: \.self) { i in
                RoundedRectangle(cornerRadius: 2).fill(color.opacity(0.7))
                    .frame(width: 9, height: max(2, CGFloat(abs(g[i])) * 40))
            }
        }
        .frame(height: 50, alignment: .bottom)
        .animation(.easeInOut, value: g)
    }

    // MARK: - Comparison

    private var comparison: some View {
        let t = app.trainer
        let ours = Double(t.parameterCount), ourTokens = Double(t.settings.totalSteps * t.settings.batchSize * t.config.context)
        let rows: [(String, Double, Double, String)] = [
            ("Our tiny model", ours, ourTokens, "a laptop CPU, ~1 minute"),
            ("GPT-2 (2019)", 1.5e9, 1e10, "token count and hardware estimated"),
            ("GPT-3 (2020)", 175e9, 300e9, "thousands of GPUs, weeks"),
            ("Llama 3 405B (2024)", 405e9, 15.6e12, "~16,000 H100 GPUs, months"),
        ]
        return Card(title: "Same loop, different universe", subtitle: "Published or widely reported figures; compute ≈ 6·N·D") {
            Grid(alignment: .leading, horizontalSpacing: 22, verticalSpacing: 10) {
                GridRow {
                    Text("Model").bold(); Text("Parameters").bold(); Text("Training tokens").bold(); Text("Compute (FLOPs)").bold(); Text("Hardware").bold()
                }
                Divider().gridCellUnsizedAxes(.horizontal)
                ForEach(rows, id: \.0) { r in
                    GridRow {
                        Text(r.0).fontWeight(r.0.hasPrefix("Our") ? .bold : .regular).foregroundStyle(r.0.hasPrefix("Our") ? Color.teal : .primary)
                        Text(Format.big(r.1)).monospacedDigit()
                        Text(Format.big(r.2)).monospacedDigit()
                        Text(Format.sci(6 * r.1 * r.2)).monospacedDigit()
                        Text(r.3).foregroundStyle(.secondary)
                    }
                }
            }
            .font(.callout)
            Prose("From our model to a frontier model is roughly **10¹⁴× more compute**. The algorithm is the one you watched in Chapter 9.")
        }
    }
}
