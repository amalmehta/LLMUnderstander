import SwiftUI

/// The pinned stage of the Tour. `step` picks the scene; `local` (0…1) is how far the
/// reader has scrolled through it, for things that draw as you scroll.
struct TourStage: View {
    let step: Int
    let local: Double
    @Environment(AppState.self) private var app
    /// Computed once per model version, not on every scroll frame.
    @State private var data: TourData?

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18).fill(Color(nsColor: .controlBackgroundColor))
            RoundedRectangle(cornerRadius: 18).strokeBorder(Color.primary.opacity(0.08))
            if let data {
                scene(data)
                    .id(step)
                    .transition(.opacity)
                    .padding(28)
            }
        }
        .task(id: app.trainer.snapshotVersion) { data = TourData(trainer: app.trainer) }
        .animation(.easeInOut(duration: 0.25), value: step)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Tour stage: \(TourStep.all[step].title)")
    }

    @ViewBuilder private func scene(_ d: TourData) -> some View {
        switch step {
        case 0: TextScene(d: d)
        case 1: TokenScene(d: d)
        case 2: EmbeddingScene(d: d)
        case 3: AttentionScene(d: d, local: local)
        case 4: BlockScene(local: local, backwards: false)
        case 5: PredictionScene(d: d)
        case 6: LossScene(d: d)
        case 7: BlockScene(local: local, backwards: true)
        case 8: UpdateScene(d: d, local: local)
        case 9: LoopScene(d: d, local: local)
        case 10: ScaleScene(d: d, local: local)
        case 11: SFTScene()
        case 12: PreferenceScene(local: local)
        case 13: ReasoningScene(local: local)
        default: AgentScene(local: local)
        }
    }
}

// MARK: - Real numbers from the reader's model

struct TourData {
    let ids: [Int]
    let prefixIDs: [Int]
    let embeddings: [[Double]]          // one row per token (first 8 numbers)
    let attention: [Double]             // last prefix token → each earlier one
    let probs: [Double]
    let target: Int
    let curve: [Point]
    let trainedSteps: Int
    let tokensSeen: Int
    let vocab: Int
    let width: Int
    let parameters: Int

    @MainActor init(trainer t: Trainer) {
        let tok = CharTokenizer.shared
        let model = t.snapshot
        _ = t.snapshotVersion
        ids = tok.encode(TourStep.sentence)
        prefixIDs = tok.encode(TourStep.prefix)
        embeddings = ids.map { Array(model.wte.row($0).prefix(8)) }
        let trace = model.trace(prefixIDs)
        let lastLayer = trace.attention.count - 1
        attention = trace.attention[lastLayer][0][prefixIDs.count - 1]
        probs = model.nextTokenProbs(prefixIDs)
        target = tok.encode("o")[0]
        curve = t.smoothedTrainLoss
        trainedSteps = t.step
        tokensSeen = t.tokensSeen
        vocab = tok.size
        width = model.config.dModel
        parameters = model.parameterCount
    }

    var tok: CharTokenizer { .shared }
}

// MARK: - Pop-in

/// Springs a piece into place when its scene appears, staggered by index. Fades only with Reduce Motion.
private struct Pop: ViewModifier {
    let index: Int
    @State private var shown = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    func body(content: Content) -> some View {
        content
            .scaleEffect(shown || reduceMotion ? 1 : 0.5)
            .opacity(shown ? 1 : 0)
            .onAppear {
                let delay = Double(min(index, 30)) * 0.03
                withAnimation(reduceMotion ? .easeOut(duration: 0.2).delay(delay) : .spring(response: 0.38, dampingFraction: 0.68).delay(delay)) { shown = true }
            }
    }
}

private extension View {
    func pop(_ i: Int = 0) -> some View { modifier(Pop(index: i)) }
}

private func chip(_ text: String, sub: String? = nil, color: Color = .blue, size: CGFloat = 20) -> some View {
    VStack(spacing: 2) {
        Text(text).font(.system(size: size, weight: .medium, design: .monospaced))
        if let sub { Text(sub).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary) }
    }
    .frame(minWidth: size * 1.4)
    .padding(.vertical, 5).padding(.horizontal, 3)
    .background(color.opacity(0.15), in: RoundedRectangle(cornerRadius: 7))
    .overlay(RoundedRectangle(cornerRadius: 7).strokeBorder(color.opacity(0.35)))
}

private func sceneTitle(_ s: String) -> some View {
    Text(s).font(.caption.weight(.semibold)).tracking(1).foregroundStyle(.secondary)
}

// MARK: - Scenes

private struct TextScene: View {
    let d: TourData
    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(TourStep.sentence.enumerated()), id: \.offset) { i, c in
                Text(String(c)).font(.system(size: 44, weight: .semibold, design: .rounded)).pop(i)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct TokenScene: View {
    let d: TourData
    var body: some View {
        VStack(spacing: 22) {
            sceneTitle("ONE TOKEN PER CHARACTER")
            FlowLayout(spacing: 6) {
                ForEach(Array(d.ids.enumerated()), id: \.offset) { i, id in
                    chip(d.tok.display(id), sub: "\(id)", color: Palette.token(id)).pop(i)
                }
            }
            Text("\(d.ids.count) tokens, from a vocabulary of \(d.vocab)").font(.callout).foregroundStyle(.secondary).pop(d.ids.count)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct EmbeddingScene: View {
    let d: TourData
    var body: some View {
        let shown = Array(d.ids.prefix(12).enumerated())
        let scale = d.embeddings.flatMap { $0 }.map(abs).max() ?? 1
        VStack(spacing: 18) {
            sceneTitle("EACH TOKEN LOOKS UP A ROW OF NUMBERS")
            HStack(alignment: .top, spacing: 8) {
                ForEach(shown, id: \.offset) { i, id in
                    VStack(spacing: 6) {
                        chip(d.tok.display(id), color: Palette.token(id), size: 16)
                        VStack(spacing: 2) {
                            ForEach(0..<d.embeddings[i].count, id: \.self) { k in
                                RoundedRectangle(cornerRadius: 2)
                                    .fill(Palette.diverging(d.embeddings[i][k], scale: scale))
                                    .frame(width: 26, height: 14)
                                    .pop(i + k)
                            }
                        }
                    }
                    .pop(i)
                }
            }
            Text("First 8 of \(d.width) numbers per token. Red is positive, blue negative. The same letter always gets the same row.")
                .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center).pop(20)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct AttentionScene: View {
    let d: TourData
    let local: Double
    var body: some View {
        let n = d.prefixIDs.count
        VStack(spacing: 14) {
            sceneTitle("\"m\" LOOKS BACK AT EVERY EARLIER TOKEN")
            GeometryReader { g in
                let spacing = g.size.width / CGFloat(n)
                let y = g.size.height * 0.78
                let draw = min(1, local * 1.6 + 0.15)
                ZStack {
                    ForEach(0..<(n - 1), id: \.self) { j in
                        let w = d.attention[j]
                        let from = CGPoint(x: spacing * (CGFloat(n - 1) + 0.5), y: y - 18)
                        let to = CGPoint(x: spacing * (CGFloat(j) + 0.5), y: y - 18)
                        let lift = min(g.size.height * 0.7, abs(from.x - to.x) * 0.55)
                        Path { p in
                            p.move(to: from)
                            p.addQuadCurve(to: to, control: CGPoint(x: (from.x + to.x) / 2, y: y - 18 - lift))
                        }
                        .trim(from: 0, to: draw)
                        .stroke(Color.orange.opacity(0.25 + 0.75 * w), style: StrokeStyle(lineWidth: 1 + 14 * w, lineCap: .round))
                    }
                    ForEach(Array(d.prefixIDs.enumerated()), id: \.offset) { i, id in
                        VStack(spacing: 3) {
                            chip(d.tok.display(id), color: i == n - 1 ? .orange : Palette.token(id), size: 15)
                            Text(i == n - 1 ? "" : String(format: "%.0f%%", d.attention[i] * 100))
                                .font(.system(size: 9).monospacedDigit()).foregroundStyle(.secondary)
                                .opacity(draw)
                        }
                        .position(x: spacing * (CGFloat(i) + 0.5), y: y + 4)
                        .pop(i)
                    }
                }
            }
            Text("Line thickness = attention weight, from the last block of your model. Keep scrolling to draw them.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }
}

private struct BlockScene: View {
    let local: Double
    let backwards: Bool
    var body: some View {
        let parts = ["embeddings", "attention", "MLP", "attention", "MLP", "scores for the next token"]
        let colors: [Color] = [.blue, .orange, .teal, .orange, .teal, .purple]
        let t = backwards ? 1 - local : local
        VStack(spacing: 14) {
            sceneTitle(backwards ? "GRADIENTS FLOW BACK DOWN" : "THE VECTORS FLOW UP THROUGH TWO BLOCKS")
            GeometryReader { g in
                let h = g.size.height, slot = h / CGFloat(parts.count)
                ZStack {
                    Rectangle().fill(Color.primary.opacity(0.15)).frame(width: 3, height: h - slot)
                        .position(x: g.size.width * 0.3, y: h / 2)
                    ForEach(parts.indices, id: \.self) { i in
                        let y = h - slot * (CGFloat(i) + 0.5)
                        HStack(spacing: 10) {
                            Text(parts[i]).font(.system(size: 15, weight: .medium))
                                .padding(.horizontal, 14).padding(.vertical, 8)
                                .background(colors[i].opacity(0.16), in: RoundedRectangle(cornerRadius: 9))
                                .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(colors[i].opacity(0.4)))
                            if backwards && i > 0 && i < parts.count - 1 {
                                Text("∂loss/∂W").font(.system(size: 12, design: .monospaced)).foregroundStyle(.pink)
                                    .opacity(t <= Double(i) / Double(parts.count - 1) + 0.05 ? 1 : 0.15)
                            }
                        }
                        .position(x: g.size.width * 0.3 + 120, y: y)
                        .pop(i)
                    }
                    Circle().fill(backwards ? Color.pink : Color.blue)
                        .frame(width: 18, height: 18)
                        .shadow(color: (backwards ? Color.pink : Color.blue).opacity(0.5), radius: 6)
                        .position(x: g.size.width * 0.3, y: h - slot * 0.5 - (h - slot) * CGFloat(t))
                    if !backwards {
                        Text("+ added to the running total").font(.caption).foregroundStyle(.secondary)
                            .position(x: g.size.width * 0.3 - 90, y: h / 2)
                    }
                }
            }
        }
    }
}

private struct PredictionScene: View {
    let d: TourData
    var body: some View {
        let top = d.probs.indices.sorted { d.probs[$0] > d.probs[$1] }.prefix(6)
        VStack(spacing: 16) {
            HStack(spacing: 0) {
                Text(TourStep.prefix).font(.system(size: 26, design: .monospaced))
                Text("?").font(.system(size: 26, weight: .bold, design: .monospaced)).foregroundStyle(.purple)
            }
            .pop()
            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(top.enumerated()), id: \.offset) { r, i in
                    HStack(spacing: 10) {
                        Text(d.tok.display(i)).font(.system(size: 18, design: .monospaced)).frame(width: 30)
                        RoundedRectangle(cornerRadius: 4)
                            .fill(i == d.target ? Color.green : Color.purple.opacity(0.5))
                            .frame(width: max(4, 340 * d.probs[i]), height: 20)
                        Text(String(format: "%.0f%%", d.probs[i] * 100)).font(.callout.monospacedDigit())
                    }
                    .pop(r + 1)
                }
            }
            Text(d.trainedSteps == 0 ? "Untrained, so the guesses are nearly even. Train the model and come back." : "Green is the real next letter.")
                .font(.callout).foregroundStyle(.secondary).pop(8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct LossScene: View {
    let d: TourData
    var body: some View {
        let p = d.probs[d.target]
        let loss = -log(max(p, 1e-12)), worst = log(Double(d.vocab))
        VStack(spacing: 18) {
            sceneTitle("HOW SURPRISED WAS IT?")
            Text("loss = −log p(\"o\") = −log \(String(format: "%.2f", p)) = \(String(format: "%.2f", loss))")
                .font(.system(size: 22, design: .monospaced)).pop()
            VStack(alignment: .leading, spacing: 8) {
                bar("this guess", loss, worst, .pink).pop(1)
                bar("guessing evenly", worst, worst, .gray).pop(2)
                bar("perfect", 0, worst, .green).pop(3)
            }
            Text("Training's only goal: make this number small, averaged over every token it sees.")
                .font(.callout).foregroundStyle(.secondary).pop(4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func bar(_ label: String, _ v: Double, _ maxV: Double, _ c: Color) -> some View {
        HStack(spacing: 10) {
            Text(label).font(.callout).frame(width: 130, alignment: .trailing)
            RoundedRectangle(cornerRadius: 4).fill(c.opacity(0.6)).frame(width: max(3, 300 * v / maxV), height: 18)
            Text(String(format: "%.2f", v)).font(.callout.monospacedDigit())
        }
    }
}

private struct UpdateScene: View {
    let d: TourData
    let local: Double
    var body: some View {
        let cols = 10, rows = 6
        VStack(spacing: 18) {
            sceneTitle("EVERY WEIGHT TAKES A SMALL STEP")
            Text("w ← w − learning rate × gradient").font(.system(size: 22, design: .monospaced)).pop()
            Grid(horizontalSpacing: 4, verticalSpacing: 4) {
                ForEach(0..<rows, id: \.self) { r in
                    GridRow {
                        ForEach(0..<cols, id: \.self) { c in
                            let base = sin(Double(r * 7 + c * 3)) * 0.8
                            let grad = cos(Double(r * 5 + c * 11)) * 0.6
                            RoundedRectangle(cornerRadius: 3)
                                .fill(Palette.diverging(base - grad * local, scale: 1.2))
                                .frame(width: 30, height: 30)
                                .pop(r * cols + c)
                        }
                    }
                }
            }
            Text("Scroll to apply the step. Each square is one weight; your model has \(d.parameters.formatted()) of them, and every one moves at once.")
                .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center).pop(10)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct LoopScene: View {
    let d: TourData
    let local: Double
    var body: some View {
        let phases = ["batch", "forward", "loss", "backward", "update"]
        let active = Int(local * 5) % 5
        HStack(spacing: 30) {
            ZStack {
                Circle().stroke(Color.teal.opacity(0.3), style: StrokeStyle(lineWidth: 3, dash: [6, 6])).frame(width: 220, height: 220)
                ForEach(phases.indices, id: \.self) { i in
                    let a = -Double.pi / 2 + Double(i) * 2 * .pi / 5
                    Text(phases[i]).font(.system(size: 14, weight: i == active ? .bold : .regular))
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background((i == active ? Color.teal : Color.secondary).opacity(i == active ? 0.3 : 0.1), in: Capsule())
                        .offset(x: 110 * cos(a), y: 110 * sin(a))
                        .pop(i)
                }
                Text("×\(max(d.trainedSteps, 1500))").font(.system(size: 20, weight: .semibold, design: .rounded)).foregroundStyle(.teal)
            }
            .frame(width: 300, height: 300)
            VStack(alignment: .leading, spacing: 6) {
                Text("Your model's loss").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                if d.curve.count > 1 {
                    let shown = Array(d.curve.prefix(max(2, Int(Double(d.curve.count) * min(1, local * 1.4 + 0.1)))))
                    let maxY = max(d.curve.map(\.y).max() ?? 4, 0.1)
                    let maxX = max(d.curve.last?.x ?? 1, 1)
                    GeometryReader { g in
                        Path { p in
                            for (k, q) in shown.enumerated() {
                                let pt = CGPoint(x: g.size.width * q.x / maxX, y: g.size.height * (1 - q.y / maxY))
                                if k == 0 { p.move(to: pt) } else { p.addLine(to: pt) }
                            }
                        }
                        .stroke(Color.orange, lineWidth: 2.5)
                    }
                    .frame(width: 280, height: 170)
                    .background(Color.primary.opacity(0.03))
                } else {
                    Text("Train the model in Chapter 9 and your loss curve appears here.")
                        .font(.callout).foregroundStyle(.secondary).frame(width: 260, height: 170)
                }
            }
            .pop(6)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct ScaleScene: View {
    let d: TourData
    let local: Double
    var body: some View {
        let count = 12 + Int(local * 260)
        VStack(spacing: 16) {
            sceneTitle("THE SAME LOOP, A TRILLION TIMES BIGGER")
            LazyVGrid(columns: Array(repeating: GridItem(.fixed(14), spacing: 3), count: 24), spacing: 3) {
                ForEach(0..<count, id: \.self) { i in
                    RoundedRectangle(cornerRadius: 2).fill(Color.orange.opacity(0.25 + 0.5 * Double((i * 37) % 10) / 10)).frame(width: 14, height: 10)
                }
            }
            .frame(height: 170, alignment: .top)
            .animation(.easeOut(duration: 0.15), value: count)
            HStack(spacing: 30) {
                stat("Your model saw", Format.big(Double(max(d.tokensSeen, 384_000))) + " tokens").pop(1)
                stat("A recent open 8B model", "~15T tokens").pop(2)
                stat("GPUs", "thousands, for months").pop(3)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func stat(_ a: String, _ b: String) -> some View {
        VStack(spacing: 3) {
            Text(a).font(.caption).foregroundStyle(.secondary)
            Text(b).font(.system(size: 18, weight: .semibold, design: .rounded))
        }
    }
}

private struct SFTScene: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            sceneTitle("A DEMONSTRATION, GRADED ONLY ON THE ANSWER")
            HStack(spacing: 4) {
                ForEach(Array("what does the cat see ↵".split(separator: " ").enumerated()), id: \.offset) { i, w in
                    Text(String(w)).font(.system(size: 20, design: .monospaced)).padding(4)
                        .background(Color.gray.opacity(0.15), in: RoundedRectangle(cornerRadius: 5)).pop(i)
                }
            }
            HStack(spacing: 4) {
                ForEach(Array("the cat sees the moon.".split(separator: " ").enumerated()), id: \.offset) { i, w in
                    Text(String(w)).font(.system(size: 20, design: .monospaced)).padding(4)
                        .background(Color.pink.opacity(0.25), in: RoundedRectangle(cornerRadius: 5)).pop(i + 6)
                }
            }
            Text("Grey: read, not graded. Pink: the part the loss counts. Same loop as before.")
                .font(.callout).foregroundStyle(.secondary).pop(12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct PreferenceScene: View {
    let local: Double
    var body: some View {
        let chosen = local > 0.3
        HStack(spacing: 24) {
            answer("the cat sees the moon at night.", good: true, chosen: chosen).pop(0)
            answer("cat moon cat moon moon.", good: false, chosen: chosen).pop(1)
        }
        .overlay(alignment: .bottom) {
            Text(chosen ? "Reward model: +2.1 vs −1.4 → push the model toward the left answer" : "Scroll: a person picks the better answer")
                .font(.callout).foregroundStyle(.secondary).offset(y: 60)
                .animation(.easeOut(duration: 0.2), value: chosen)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func answer(_ t: String, good: Bool, chosen: Bool) -> some View {
        VStack(spacing: 12) {
            Text(t).font(.system(size: 17, design: .monospaced)).multilineTextAlignment(.center)
            Image(systemName: good ? "hand.thumbsup.fill" : "hand.thumbsdown")
                .font(.title).foregroundStyle(good ? .green : .secondary)
                .scaleEffect(chosen ? 1 : 0.01).opacity(chosen ? 1 : 0)
                .animation(.spring(response: 0.35, dampingFraction: 0.6), value: chosen)
        }
        .frame(width: 230, height: 130)
        .background(RoundedRectangle(cornerRadius: 12).fill((chosen && good ? Color.green : Color.primary).opacity(chosen && good ? 0.12 : 0.04)))
    }
}

private struct ReasoningScene: View {
    let local: Double
    var body: some View {
        let lines = ["7 + 5 = 12", "12 + 9 = 21", "21 + 3 = 24"]
        let shown = min(lines.count, Int(local * 4.5))
        VStack(alignment: .leading, spacing: 10) {
            Text("Q: 7 + 5 + 9 + 3 = ?").font(.system(size: 24, design: .monospaced)).pop()
            ForEach(0..<shown, id: \.self) { i in
                Text("  " + lines[i]).font(.system(size: 22, design: .monospaced)).foregroundStyle(.purple).pop()
            }
            if shown == lines.count {
                HStack(spacing: 12) {
                    Text("A: 24").font(.system(size: 24, weight: .semibold, design: .monospaced))
                    Label("checker: correct, reward 1", systemImage: "checkmark.seal.fill").foregroundStyle(.green)
                }
                .pop()
            }
            Text("Scroll to let it think. Writing the steps out is what earned reward.")
                .font(.callout).foregroundStyle(.secondary).padding(.top, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
    }
}

private struct AgentScene: View {
    let local: Double
    var body: some View {
        let stages = [("model", "writes a tool call", Color.purple), ("harness", "checks and runs it", Color.green), ("tool", "run_tests → 1 failed", Color.orange), ("context", "result goes back in", Color.blue)]
        let active = Int(local * 8) % stages.count
        ZStack {
            Circle().stroke(Color.green.opacity(0.25), style: StrokeStyle(lineWidth: 3, dash: [6, 6])).frame(width: 260, height: 260)
            ForEach(stages.indices, id: \.self) { i in
                let a = -Double.pi / 2 + Double(i) * 2 * .pi / Double(stages.count)
                VStack(spacing: 2) {
                    Text(stages[i].0).font(.system(size: 15, weight: .semibold))
                    Text(stages[i].1).font(.caption).foregroundStyle(.secondary)
                }
                .padding(10)
                .background(stages[i].2.opacity(i == active ? 0.28 : 0.1), in: RoundedRectangle(cornerRadius: 10))
                .scaleEffect(i == active ? 1.08 : 1)
                .animation(.spring(response: 0.3, dampingFraction: 0.7), value: active)
                .offset(x: 150 * cos(a), y: 130 * sin(a))
                .pop(i)
            }
            Text("until\nthe task\nis done").font(.callout).multilineTextAlignment(.center).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
