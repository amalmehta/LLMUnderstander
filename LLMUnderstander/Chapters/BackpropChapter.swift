import SwiftUI
import Charts

struct BackpropChapter: View {
    @Environment(AppState.self) private var app
    @State private var w = 1.0
    @State private var b = 0.5
    @State private var step = 0
    @State private var history: [Point] = []
    @State private var checks: [GradCheckRow] = []

    private let x = 2.0, target = 7.0, lr = 0.05

    /// Every node of L = (w·x + b − y)², computed with the app's autograd engine.
    private struct Graph {
        var values: [String: Double] = [:]
        var grads: [String: Double] = [:]
    }

    private var graph: Graph {
        let tx = Tensor(rows: 1, cols: 1, data: [x]), tw = Tensor(rows: 1, cols: 1, data: [w]), tb = Tensor(rows: 1, cols: 1, data: [b])
        let ty = Tensor(rows: 1, cols: 1, data: [target])
        let m = Ops.matmul(tw, tx)
        let yhat = Ops.add(m, tb)
        let e = Ops.add(yhat, Ops.scale(ty, -1))
        let L = Ops.matmul(e, e)
        L.backward()
        var g = Graph()
        for (k, t) in [("x", tx), ("w", tw), ("b", tb), ("m", m), ("ŷ", yhat), ("e", e), ("L", L)] {
            g.values[k] = t.item; g.grads[k] = t.grad[0]
        }
        return g
    }

    private let steps: [(reveal: [String], grad: Bool, text: LocalizedStringKey)] = [
        ([], false, "Start: we have an input **x = 2**, a correct answer **y = 7**, and two weights **w** and **b**. Press *Next* to run the forward pass."),
        (["m"], false, "**Forward 1:** multiply: m = w × x."),
        (["ŷ"], false, "**Forward 2:** add the bias: ŷ = m + b. This is the model's prediction."),
        (["e"], false, "**Forward 3:** error e = ŷ − y."),
        (["L"], false, "**Forward 4:** loss L = e². Now go *backwards* to ask: how would L change if each number changed a little?"),
        (["L"], true, "**Backward 1:** dL/dL = 1 (trivially)."),
        (["e"], true, "**Backward 2:** L = e², so dL/de = 2e."),
        (["ŷ"], true, "**Backward 3:** e = ŷ − y, so dL/dŷ = dL/de × 1. Addition passes the gradient through unchanged."),
        (["m", "b"], true, "**Backward 4:** ŷ = m + b, so both get the same gradient: dL/dm = dL/db = dL/dŷ."),
        (["w", "x"], true, "**Backward 5:** m = w × x, so dL/dw = dL/dm × **x**, and dL/dx = dL/dm × **w**. Each multiply swaps in the *other* input. That's the chain rule."),
        (["w", "b"], true, "**Update:** now we know which way to push. Press *Take a gradient step*: w ← w − lr·dL/dw, b ← b − lr·dL/db. The loss drops."),
    ]

    var body: some View {
        let g = graph
        let revealedValues = Set(steps.prefix(step + 1).filter { !$0.grad }.flatMap(\.reveal) + ["x", "w", "b"])
        let revealedGrads = Set(steps.prefix(step + 1).filter(\.grad).flatMap(\.reveal))

        ChapterPage(chapter: .backprop) {
            Prose("We know how wrong the model is (the loss). Now: **which of its weights should change, and in which direction?** For every weight we want its **gradient** — how much the loss would go up if that weight were nudged up a tiny bit.")
            Prose("Computing that one weight at a time would take forever with billions of weights. **Backpropagation** gets all of them in one backward sweep, costing only about twice a forward pass. It's just the chain rule from calculus, applied from the loss backwards.")

            Card(title: "Step through backprop on a one-neuron model", subtitle: "L = (w·x + b − y)²   — gradients computed by the same autograd engine that trains the tiny GPT") {
                HStack(spacing: 8) {
                    VStack(spacing: 12) {
                        node("x", "input", g, revealedValues, revealedGrads, .gray)
                        node("w", "weight", g, revealedValues, revealedGrads, .blue)
                    }
                    arrow
                    node("m", "w × x", g, revealedValues, revealedGrads, .teal)
                    arrow
                    VStack(spacing: 12) {
                        node("ŷ", "m + b", g, revealedValues, revealedGrads, .teal)
                        node("b", "weight", g, revealedValues, revealedGrads, .blue)
                    }
                    arrow
                    node("e", "ŷ − y (y = 7)", g, revealedValues, revealedGrads, .teal)
                    arrow
                    node("L", "e²", g, revealedValues, revealedGrads, .orange)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)

                Text(steps[step].text).font(.system(size: 15)).frame(minHeight: 44, alignment: .topLeading)
                HStack {
                    Button("Back") { step = max(0, step - 1) }.disabled(step == 0)
                    Button("Next") { step = min(steps.count - 1, step + 1) }.buttonStyle(.borderedProminent).disabled(step == steps.count - 1)
                    Spacer()
                    Button("Take a gradient step") {
                        if history.isEmpty { history.append(Point(x: 0, y: g.values["L"]!)) }
                        w -= lr * g.grads["w"]!
                        b -= lr * g.grads["b"]!
                        history.append(Point(x: Double(history.count), y: graph.values["L"]!))
                    }
                    .disabled(step < steps.count - 1)
                    Button("Reset") { w = 1; b = 0.5; step = 0; history = [] }
                }
                if !history.isEmpty {
                    Chart(history) { LineMark(x: .value("step", $0.x), y: .value("loss", $0.y)); PointMark(x: .value("step", $0.x), y: .value("loss", $0.y)) }
                        .chartXAxisLabel("gradient steps").chartYAxisLabel("loss")
                        .frame(height: 140)
                    Text(String(format: "w = %.3f   b = %.3f   prediction ŷ = %.3f (target 7)", w, b, g.values["ŷ"]!))
                        .font(.system(.callout, design: .monospaced))
                }
            }

            Card(title: "The same thing, on every weight of the real model", subtitle: "Gradient check: compare backprop's answer with brute force (nudge the weight, re-run, measure)") {
                HStack {
                    Button("Run gradient check on 8 random weights") { runCheck() }.buttonStyle(.borderedProminent)
                    Text("Our model has \(app.trainer.parameterCount.formatted()) weights. One backward pass gives *all* their gradients at once.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if !checks.isEmpty {
                    Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 6) {
                        GridRow {
                            Text("Weight").bold(); Text("Backprop").bold(); Text("Brute force").bold(); Text("")
                        }
                        ForEach(checks) { c in
                            GridRow {
                                Text(c.name).font(.system(.callout, design: .monospaced))
                                Text(String(format: "%+.6f", c.analytic)).monospacedDigit()
                                Text(String(format: "%+.6f", c.numeric)).monospacedDigit()
                                Image(systemName: c.ok ? "checkmark.circle.fill" : "xmark.circle.fill").foregroundStyle(c.ok ? .green : .red)
                            }
                        }
                    }
                    .font(.callout)
                }
            }

            KeyIdea("A gradient is a direction: \"to reduce the loss, move this weight that way, this much.\" Backprop computes it for every weight in one sweep by passing blame backwards through each operation.")

            GoDeeper(title: "why it works at scale") {
                Formula("Chain rule:  dL/dw = dL/dŷ · dŷ/dm · dm/dw")
                Prose("Every operation (matrix multiply, softmax, LayerNorm, GELU…) knows its own small *local* derivative. Autograd records the operations during the forward pass, then visits them in reverse, multiplying local derivatives together. Frameworks like PyTorch and JAX do exactly what this app's ~300-line engine does.")
                Prose("**Memory:** the backward pass needs the activations saved from the forward pass. For big models that memory, not the weights, is often the bottleneck. *Activation checkpointing* throws some away and recomputes them when needed.")
                Prose("**Vanishing/exploding gradients:** multiplying many local derivatives can shrink or blow up the result. Residual connections (Chapter 5), normalisation and careful initialisation keep gradients healthy through hundreds of layers.")
            }
        }
    }

    private var arrow: some View { Image(systemName: "arrow.right").foregroundStyle(.secondary) }

    private func node(_ k: String, _ caption: String, _ g: Graph, _ vals: Set<String>, _ grads: Set<String>, _ color: Color) -> some View {
        VStack(spacing: 3) {
            Text(k).font(.system(size: 16, weight: .bold, design: .monospaced))
            Text(caption).font(.system(size: 9.5)).foregroundStyle(.secondary)
            Text(vals.contains(k) ? String(format: "%.2f", g.values[k]!) : "?")
                .font(.system(size: 13, design: .monospaced))
            Text(grads.contains(k) ? String(format: "grad %.2f", g.grads[k]!) : " ")
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(.red)
        }
        .frame(width: 100, height: 82)
        .background(color.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(grads.contains(k) ? Color.red : color.opacity(0.4), lineWidth: grads.contains(k) ? 2 : 1))
    }

    struct GradCheckRow: Identifiable {
        let id = UUID()
        let name: String
        let analytic: Double
        let numeric: Double
        var ok: Bool { abs(analytic - numeric) <= 1e-4 * max(1, abs(analytic) + abs(numeric)) }
    }

    private func runCheck() {
        let model = app.trainer.snapshot.copy()
        let tok = CharTokenizer.shared
        let s = tok.encode("the fox finds a green leaf near the house.")
        let batch = [(Array(s[0..<32]), Array(s[1..<33]))]
        let params = model.parameters
        for p in params { p.zeroGrad() }
        model.loss(batch: batch).backward()
        var rng = SplitMix64(seed: UInt64(Date().timeIntervalSince1970))
        var rows: [GradCheckRow] = []
        for _ in 0..<8 {
            let p = params[rng.int(params.count)]
            let i = rng.int(p.count)
            let old = p.data[i], h = 1e-5
            p.data[i] = old + h; let up = model.loss(batch: batch).item
            p.data[i] = old - h; let down = model.loss(batch: batch).item
            p.data[i] = old
            rows.append(GradCheckRow(name: "\(p.label)[\(i)]", analytic: p.grad[i], numeric: (up - down) / (2 * h)))
        }
        checks = rows
    }
}
