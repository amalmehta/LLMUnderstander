import SwiftUI
import Charts

struct InferenceChapter: View {
    @Environment(AppState.self) private var app
    @State private var prompt = "the old man "
    @State private var temperature = 0.8
    @State private var topP = 0.9
    @State private var seed: UInt64 = 1
    @State private var bigModel = false
    @State private var acceptance = 0.7
    @State private var draft = 4.0
    @State private var draftCost = 0.05

    private let tok = CharTokenizer.shared

    var body: some View {
        ChapterPage(chapter: .inference) {
            ModelStatusBanner()
            Prose("**Inference** is running the trained model to get text out. It's the same forward pass as in training, minus the backward pass and the update, done one token at a time. Training happens once; inference happens every time anyone uses the model, so most of a popular model's lifetime compute goes here.")
            Prose("Three questions decide what you get and how fast: *which* token to pick from the probabilities, how to avoid redoing work for every new token, and how to produce more than one token per step.")

            samplingCard
            cacheCard
            speculativeCard

            KeyIdea("Inference picks from the model's probabilities, one token at a time. The tricks that make it practical, like caches and draft models, never change what the model computes, only how much work it takes.")

            GoDeeper(title: "what real serving systems add") {
                Prose("**Batching.** A GPU running one conversation is mostly waiting on memory. Servers process many conversations in the same forward pass, adding and removing them as they start and finish (*continuous batching*).")
                Prose("**The two phases.** Reading the prompt (*prefill*) processes all its tokens at once and is limited by arithmetic; generating (*decode*) adds one token at a time and is limited by how fast weights and the cache can be read from memory. That's why long answers cost more per token than long prompts.")
                Prose("**Smaller caches.** Grouped-query attention shares each key and value among several query heads, shrinking the cache several times over, and many models store weights and the cache in 8 or 4 bits (*quantisation*).")
                Prose("**Where our numbers come from.** The work counts are exact multiply-adds from the model's shape, not timings; on a model this small, timings would mostly measure overhead. The speculative-decoding numbers come from the formula, not a real draft model.")
            }
        }
    }

    // MARK: - Sampling

    private var samplingCard: some View {
        let model = app.trainer.snapshot
        let _ = app.trainer.snapshotVersion
        let ids = Array(tok.encode(tok.clean(prompt)).suffix(model.config.context))
        let probs = ids.isEmpty ? [] : model.nextTokenProbs(ids, temperature: temperature)
        let kept = Set(InferenceMaths.nucleus(probs, p: topP))
        let top = probs.indices.sorted { probs[$0] > probs[$1] }.prefix(12)
        return Card(title: "Live: choosing the next token", subtitle: "Temperature reshapes the probabilities; top-p then keeps only the likeliest tokens that together reach p") {
            HStack(spacing: 12) {
                TextField("Prompt", text: $prompt).textFieldStyle(.roundedBorder).font(.system(.body, design: .monospaced)).frame(width: 260)
                VStack(alignment: .leading, spacing: 0) {
                    Text("Temperature \(String(format: "%.2f", temperature))").font(.caption).monospacedDigit()
                    Slider(value: $temperature, in: 0.1...2).frame(width: 160)
                }
                VStack(alignment: .leading, spacing: 0) {
                    Text("Top-p \(String(format: "%.2f", topP))").font(.caption).monospacedDigit()
                    Slider(value: $topP, in: 0.05...1).frame(width: 160)
                }
            }
            if !probs.isEmpty {
                Chart(Array(top), id: \.self) { i in
                    BarMark(x: .value("token", tok.display(i)), y: .value("p", probs[i]))
                        .foregroundStyle(kept.contains(i) ? Color.purple : Color.gray.opacity(0.3))
                        .annotation(position: .top) { Text(String(format: "%.0f%%", probs[i] * 100)).font(.system(size: 9)).foregroundStyle(.secondary) }
                }
                .chartYScale(domain: 0...1)
                .chartYAxis(.hidden)
                .frame(height: 150)
                Text("\(kept.count) of \(probs.count) characters kept (purple). The rest can't be picked, however lucky the draw.")
                    .font(.caption).foregroundStyle(.secondary)
                Divider()
                Continuations(model: model, version: app.trainer.snapshotVersion, prompt: tok.clean(prompt), ids: ids,
                              temperature: temperature, topP: topP, seed: seed) { seed &+= 3 }
                Text("Three continuations with these settings. Low temperature or low top-p: safe and repetitive, often the same three times. High: varied, then nonsense.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - KV cache

    private var cacheCard: some View {
        let shape = bigModel ? InferenceMaths.Shape.gpt3 : InferenceMaths.Shape.tiny(app.trainer.config)
        let maxT = bigModel ? 4096.0 : Double(app.trainer.config.context)
        // Always end exactly at maxT, so the last point matches the "Token maxT" tiles.
        let positions = Array(stride(from: 1.0, to: maxT, by: max(1, maxT / 64))) + [maxT]
        let points = positions.map { t in
            (t, InferenceMaths.withCache(shape, position: t), InferenceMaths.withoutCache(shape, position: t))
        }
        let last = points.last!
        return Card(title: "The KV cache: don't redo old work", subtitle: "Work needed to produce each new token, counted from the model's shape") {
            Picker("", selection: $bigModel) {
                Text("Your tiny model").tag(false)
                Text("GPT-3 sized (175B)").tag(true)
            }
            .pickerStyle(.segmented).frame(width: 340)
            Chart {
                ForEach(points, id: \.0) { t, w, wo in
                    LineMark(x: .value("position", t), y: .value("work", wo), series: .value("s", "without"))
                        .foregroundStyle(Color.gray)
                    LineMark(x: .value("position", t), y: .value("work", w), series: .value("s", "with"))
                        .foregroundStyle(Color.purple)
                }
            }
            .chartXAxisLabel("position of the new token")
            .chartYAxisLabel("multiply-adds")
            .chartYAxis { AxisMarks { v in AxisGridLine(); AxisValueLabel { Text(Format.big(v.as(Double.self) ?? 0)) } } }
            .frame(height: 200)
            HStack(spacing: 16) {
                legend(.gray, "without a cache: recompute every earlier token")
                legend(.purple, "with a cache: only the new token, plus attention over the stored keys")
            }
            HStack(spacing: 10) {
                StatTile(label: "Token \(Int(maxT)), without", value: Format.big(last.2), color: .gray)
                StatTile(label: "Token \(Int(maxT)), with", value: Format.big(last.1), color: .purple)
                StatTile(label: "Saving", value: String(format: "%.0f×", last.2 / last.1), color: .green)
                StatTile(label: "Cache after \(Int(maxT)) tokens", value: Format.big(InferenceMaths.cacheSize(shape, tokens: maxT)) + " numbers", color: .blue)
            }
            Text("Keys and values of earlier tokens never change once computed (the causal mask means later tokens can't affect them), so they're stored and reused. The cost moves from computing to memory: the cache grows with every token, in every layer.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Speculative decoding

    private var speculativeCard: some View {
        let k = Int(draft)
        let perPass = InferenceMaths.tokensPerPass(acceptance: acceptance, draft: k)
        let speed = InferenceMaths.speedup(acceptance: acceptance, draft: k, draftCost: draftCost)
        let curve = (1...8).map { ($0, InferenceMaths.speedup(acceptance: acceptance, draft: $0, draftCost: draftCost)) }
        return Card(title: "Speculative decoding: guess ahead, check in one go", subtitle: "A small draft model writes k tokens; the big model checks them all in one pass and keeps the ones it agrees with") {
            HStack(alignment: .top, spacing: 24) {
                VStack(alignment: .leading, spacing: 10) {
                    slider("Draft accepted", value: $acceptance, range: 0.1...0.95, text: "\(Int((acceptance * 100).rounded()))%")
                    slider("Draft length k", value: $draft, range: 1...8, step: 1, text: "\(k)")
                    slider("Draft step cost", value: $draftCost, range: 0.01...0.5, text: "\(Int((draftCost * 100).rounded()))% of big")
                    HStack(spacing: 10) {
                        StatTile(label: "Tokens per big pass", value: String(format: "%.2f", perPass), color: .purple)
                        StatTile(label: "Expected speed-up", value: String(format: "%.2f×", speed), color: speed >= 1 ? .green : .pink)
                    }
                    Text("The output is exactly what the big model alone would write: a rejected token is replaced by one sampled from the big model, so only speed changes. Formula from Leviathan, Kalman and Matias (2023).")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                .frame(width: 400)
                Chart(curve, id: \.0) { k2, s in
                    BarMark(x: .value("k", "\(k2)"), y: .value("speed-up", s))
                        .foregroundStyle(k2 == k ? Color.purple : Color.purple.opacity(0.3))
                    RuleMark(y: .value("break-even", 1)).lineStyle(StrokeStyle(dash: [4, 3])).foregroundStyle(.gray)
                }
                .chartXAxisLabel("draft length k")
                .chartYAxisLabel("speed-up")
                .frame(height: 220)
            }
        }
    }

    private func slider(_ label: String, value: Binding<Double>, range: ClosedRange<Double>, step: Double? = nil, text: String) -> some View {
        HStack {
            Text(label).frame(width: 120, alignment: .leading)
            if let step { Slider(value: value, in: range, step: step) } else { Slider(value: value, in: range) }
            Text(text).monospacedDigit().frame(width: 80, alignment: .trailing)
        }
        .font(.callout)
    }

    private func legend(_ c: Color, _ s: String) -> some View {
        HStack(spacing: 6) { Circle().fill(c).frame(width: 8, height: 8); Text(s).font(.caption).foregroundStyle(.secondary) }
    }
}

/// Three sampled continuations. Generating them is ~150 forward passes, so it waits for the
/// sliders to settle and runs off the main thread; the old text stays until the new is ready.
private struct Continuations: View {
    let model: TinyGPT
    let version: Int
    let prompt: String
    let ids: [Int]
    let temperature: Double
    let topP: Double
    let seed: UInt64
    let again: () -> Void
    @State private var texts: [String] = []

    private struct Key: Hashable { let version: Int; let ids: [Int]; let t: Double; let p: Double; let seed: UInt64 }
    private struct Frozen: @unchecked Sendable { let model: TinyGPT }   // a snapshot nobody mutates

    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(texts.enumerated()), id: \.offset) { _, t in
                    Text(t).font(.system(size: 13, design: .monospaced)).fixedSize(horizontal: false, vertical: true)
                }
                if texts.isEmpty { Text("…").foregroundStyle(.secondary) }
            }
            .animation(.easeOut(duration: 0.15), value: texts)
            Spacer()
            Button("Sample again", action: again)
        }
        .task(id: Key(version: version, ids: ids, t: temperature, p: topP, seed: seed)) {
            try? await Task.sleep(for: .milliseconds(120))
            guard !Task.isCancelled else { return }
            let frozen = Frozen(model: model), start = ids, t = temperature, p = topP, s = seed, prompt = prompt
            let result = await Task.detached(priority: .userInitiated) {
                (0..<3).map { r in Self.generate(frozen.model, ids: start, prompt: prompt, temperature: t, topP: p, seed: s &+ UInt64(r)) }
            }.value
            if !Task.isCancelled { texts = result }
        }
    }

    nonisolated private static func generate(_ model: TinyGPT, ids start: [Int], prompt: String, temperature: Double, topP: Double, seed: UInt64) -> String {
        let tok = CharTokenizer.shared
        var rng = SplitMix64(seed: seed)
        var ids = start, out: [Int] = []
        for _ in 0..<50 {
            let probs = model.nextTokenProbs(Array(ids.suffix(model.config.context)), temperature: temperature)
            let next = InferenceMaths.sampleNucleus(probs, p: topP, rng: &rng)
            out.append(next); ids.append(next)
        }
        return prompt + tok.decode(out).replacingOccurrences(of: "\n", with: " ↵ ")
    }
}
