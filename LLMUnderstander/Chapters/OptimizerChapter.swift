import SwiftUI
import Charts

struct OptimizerChapter: View {
    @State private var landscape: Landscape = .ravine
    @State private var logLR = log10(0.35)
    @State private var shown: Set<OptimizerKind> = Set(OptimizerKind.allCases)
    @State private var progress = 120
    @State private var playing = false
    @State private var warmup = 100.0
    @State private var total = 1500.0

    private let maxSteps = 120
    private let timer = Timer.publish(every: 0.03, on: .main, in: .common).autoconnect()

    static let colors: [OptimizerKind: Color] = [.sgd: .red, .momentum: .blue, .adam: .green]

    var body: some View {
        let lr = pow(10, logLR)
        let paths = Dictionary(uniqueKeysWithValues: OptimizerKind.allCases.map { ($0, landscape.path($0, lr: lr, steps: maxSteps)) })

        ChapterPage(chapter: .optimizer) {
            Prose("Backprop gives every weight a gradient — the uphill direction for the loss. The **optimizer** decides how to step downhill. The simplest rule: move each weight a small amount against its gradient. That amount is the **learning rate**, the single most important number in training.")
            Prose("Picture the loss as a landscape and the weights as a hiker in fog who can only feel the slope underfoot. Real models have billions of dimensions; here are two, so you can watch.")

            Card(title: "Optimizer race", subtitle: "Same start, same landscape, same number of steps. Brighter = lower loss; the minimum is the brightest spot.") {
                HStack {
                    Picker("Landscape", selection: $landscape) { ForEach(Landscape.allCases) { Text($0.rawValue).tag($0) } }
                        .frame(width: 240)
                    ForEach(OptimizerKind.allCases) { k in
                        Toggle(isOn: Binding(get: { shown.contains(k) }, set: { if $0 { shown.insert(k) } else { shown.remove(k) } })) {
                            Text(k.rawValue).foregroundStyle(Self.colors[k]!)
                        }
                        .toggleStyle(.checkbox)
                    }
                    Spacer()
                    Button(playing ? "Pause" : (progress >= maxSteps ? "Replay" : "Play")) {
                        if progress >= maxSteps { progress = 0 }
                        playing.toggle()
                    }
                    .buttonStyle(.borderedProminent)
                    Button("Show all") { progress = maxSteps; playing = false }
                }
                HStack {
                    Text("Learning rate \(String(format: "%.3f", lr))").monospacedDigit().frame(width: 160, alignment: .leading)
                    Slider(value: $logLR, in: -2.5...0.2) { _ in progress = maxSteps }
                }
                .font(.callout)

                LandscapeCanvas(landscape: landscape, paths: paths.filter { shown.contains($0.key) }, progress: progress)
                    .frame(height: 360)

                HStack(spacing: 12) {
                    ForEach(OptimizerKind.allCases.filter { shown.contains($0) }) { k in
                        let p = paths[k]![min(progress, maxSteps)]
                        StatTile(label: "\(k.rawValue) loss after \(progress) steps", value: String(format: "%.4f", landscape.value(p.0, p.1)), color: Self.colors[k]!)
                    }
                }
                Text("Try it: in the **ravine**, plain SGD zig-zags across the steep walls and crawls along the floor. Raise the learning rate and it overshoots and blows up. **Momentum** builds up speed along the consistent direction. **Adam** gives each weight its own step size, so steep and flat directions both make progress.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            .onReceive(timer) { _ in
                guard playing else { return }
                if progress < maxSteps { progress += 1 } else { playing = false }
            }
            .onChange(of: landscape) { progress = 0; playing = true }

            Card(title: "The learning-rate schedule", subtitle: "Real training doesn't use one fixed learning rate") {
                let s = LRSchedule(peak: 3e-3, warmup: Int(warmup), total: Int(total))
                Chart(stride(from: 0, through: Int(total), by: 10).map { Point(x: Double($0), y: s.lr(at: $0)) }) {
                    AreaMark(x: .value("step", $0.x), y: .value("lr", $0.y)).foregroundStyle(Color.teal.opacity(0.25))
                    LineMark(x: .value("step", $0.x), y: .value("lr", $0.y)).foregroundStyle(Color.teal)
                }
                .chartXAxisLabel("training step").chartYAxisLabel("learning rate")
                .frame(height: 170)
                HStack {
                    Text("Warm-up \(Int(warmup)) steps").frame(width: 160, alignment: .leading)
                    Slider(value: $warmup, in: 0...600)
                }.font(.callout)
                Text("**Warm-up:** at the start the weights are random and gradients are wild, so the learning rate ramps up gently. **Cosine decay:** later, smaller steps let the model settle into a good spot instead of bouncing around it. Our tiny model uses exactly this schedule in Chapter 9.")
                    .font(.callout).foregroundStyle(.secondary)
            }

            KeyIdea("Gradient = which way. Learning rate = how far. Optimizer = how to combine the current gradient with what it has seen before. Nearly every LLM is trained with AdamW plus warm-up and decay.")

            GoDeeper(title: "the update rules") {
                Formula("SGD:       w ← w − lr · g\nMomentum:  v ← β·v + g;          w ← w − lr · v\nAdam:      m ← β₁·m + (1−β₁)·g     (average direction)\n           s ← β₂·s + (1−β₂)·g²    (average size²)\n           w ← w − lr · m̂ / (√ŝ + ε) − lr · λ · w   (AdamW decay)")
                Prose("**Adam** divides each weight's step by the typical size of its gradients, so weights with tiny gradients still move and weights with huge ones don't explode. The cost: two extra numbers stored per weight. For a 70B-parameter model that's hundreds of gigabytes of optimizer state.")
                Prose("**Weight decay** gently pulls weights toward zero, a mild preference for simpler solutions. **Gradient clipping** caps the total gradient size, so one bad batch can't throw training off a cliff.")
                Prose("**Batch size:** each gradient is averaged over a batch of examples. Bigger batches give less noisy directions and use hardware better. LLMs use batches of millions of tokens.")
            }
        }
    }
}

struct LandscapeCanvas: View {
    let landscape: Landscape
    let paths: [OptimizerKind: [(Double, Double)]]
    let progress: Int

    var body: some View {
        Canvas { ctx, size in
            let (xr, yr) = landscape.range
            func toScreen(_ x: Double, _ y: Double) -> CGPoint {
                CGPoint(x: (x - xr.lowerBound) / (xr.upperBound - xr.lowerBound) * size.width,
                        y: (1 - (y - yr.lowerBound) / (yr.upperBound - yr.lowerBound)) * size.height)
            }
            // background: log-loss heat
            let nx = 180, ny = 110
            var vals: [Double] = []
            for j in 0..<ny { for i in 0..<nx {
                let x = xr.lowerBound + (Double(i) + 0.5) / Double(nx) * (xr.upperBound - xr.lowerBound)
                let y = yr.upperBound - (Double(j) + 0.5) / Double(ny) * (yr.upperBound - yr.lowerBound)
                vals.append(log(landscape.value(x, y) + 0.01))
            } }
            let lo = vals.min()!, hi = vals.max()!
            let cw = size.width / CGFloat(nx), ch = size.height / CGFloat(ny)
            for j in 0..<ny { for i in 0..<nx {
                let t = (vals[j * nx + i] - lo) / (hi - lo)
                let band = Int(t * 14) % 2 == 0 ? 0.0 : 0.04   // contour banding
                ctx.fill(Path(CGRect(x: CGFloat(i) * cw, y: CGFloat(j) * ch, width: cw + 0.5, height: ch + 0.5)),
                         with: .color(Color(hue: 0.62 - 0.5 * (1 - t), saturation: 0.55, brightness: 1 - 0.55 * t - band)))
            } }
            // paths
            for (k, pts) in paths {
                let color = OptimizerChapter.colors[k]!
                var path = Path()
                let upto = min(progress, pts.count - 1)
                for (n, p) in pts.prefix(upto + 1).enumerated() {
                    let s = toScreen(p.0, p.1)
                    if n == 0 { path.move(to: s) } else { path.addLine(to: s) }
                }
                ctx.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 2.2, lineJoin: .round))
                let cur = toScreen(pts[upto].0, pts[upto].1)
                ctx.fill(Path(ellipseIn: CGRect(x: cur.x - 6, y: cur.y - 6, width: 12, height: 12)), with: .color(color))
                ctx.stroke(Path(ellipseIn: CGRect(x: cur.x - 6, y: cur.y - 6, width: 12, height: 12)), with: .color(.white), lineWidth: 1.5)
            }
            let st = toScreen(landscape.start.0, landscape.start.1)
            ctx.draw(Text("start").font(.caption.bold()).foregroundColor(.white), at: CGPoint(x: st.x, y: st.y - 14))
        }
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
