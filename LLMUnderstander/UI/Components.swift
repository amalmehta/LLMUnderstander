import SwiftUI
import Charts

// MARK: - Page scaffold

struct ChapterPage<Content: View>: View {
    let chapter: Chapter
    @ViewBuilder var content: Content
    @Environment(AppState.self) private var app

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                header
                content
                footer
            }
            .padding(.horizontal, 36)
            .padding(.vertical, 30)
            .frame(maxWidth: 1000, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle(chapter.title)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("CHAPTER \(chapter.number) · \(chapter.section.rawValue.uppercased())")
                .font(.caption.weight(.semibold))
                .tracking(1.2)
                .foregroundStyle(chapter.section.color)
            Text(chapter.title)
                .font(.system(size: 34, weight: .bold, design: .rounded))
            Text(chapter.tagline)
                .font(.title3)
                .foregroundStyle(.secondary)
        }
    }

    private var footer: some View {
        HStack {
            if let p = chapter.previous {
                Button { app.selected = p } label: { Label(p.title, systemImage: "chevron.left") }
            }
            Spacer()
            if let n = chapter.next {
                Button { app.selected = n } label: {
                    HStack { Text("Next: \(n.title)"); Image(systemName: "chevron.right") }
                }
                .buttonStyle(.borderedProminent)
                .tint(n.section.color)
            }
        }
        .padding(.top, 10)
    }
}

// MARK: - Text blocks

/// Body text. Supports **markdown**.
struct Prose: View {
    let text: LocalizedStringKey
    init(_ text: LocalizedStringKey) { self.text = text }
    var body: some View {
        Text(text)
            .font(.system(size: 15))
            .lineSpacing(4)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: 760, alignment: .leading)
    }
}

struct SectionTitle: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text).font(.title2.weight(.semibold)).padding(.top, 6)
    }
}

struct KeyIdea: View {
    let text: LocalizedStringKey
    var color: Color = .yellow
    init(_ text: LocalizedStringKey, color: Color = .yellow) { self.text = text; self.color = color }
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "lightbulb.fill").foregroundStyle(color).font(.title3)
            Text(text).font(.system(size: 15, weight: .medium)).fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
    }
}

/// Collapsible "go deeper" section for the maths.
struct GoDeeper<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content
    @State private var open = false
    var body: some View {
        DisclosureGroup(isExpanded: $open) {
            VStack(alignment: .leading, spacing: 10) { content }
                .padding(.top, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Go deeper: \(title)", systemImage: "function")
                .font(.headline)
                .contentShape(Rectangle())
                .onTapGesture { withAnimation { open.toggle() } }
        }
        .padding(14)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
    }
}

struct Formula: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text)
            .font(.system(size: 14, design: .monospaced))
            .textSelection(.enabled)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Cards

struct Card<Content: View>: View {
    var title: String? = nil
    var subtitle: String? = nil
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let title {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline)
                    if let subtitle { Text(subtitle).font(.subheadline).foregroundStyle(.secondary) }
                }
            }
            content
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Color.primary.opacity(0.08)))
    }
}

struct StatTile: View {
    let label: String
    let value: String
    var color: Color = .accentColor
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(.title3, design: .rounded).weight(.semibold)).foregroundStyle(color)
                .monospacedDigit()
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .frame(minWidth: 110, alignment: .leading)
        .background(color.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }
}

struct TokenChip: View {
    let text: String
    var color: Color = .blue
    var sub: String? = nil
    var body: some View {
        VStack(spacing: 1) {
            Text(text).font(.system(size: 14, design: .monospaced))
            if let sub { Text(sub).font(.system(size: 9, design: .monospaced)).foregroundStyle(.secondary) }
        }
        .padding(.horizontal, 5).padding(.vertical, 3)
        .background(color.opacity(0.18), in: RoundedRectangle(cornerRadius: 5))
        .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(color.opacity(0.35)))
    }
}

/// Wrapping horizontal layout for token chips.
struct FlowLayout: Layout {
    var spacing: CGFloat = 4
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 600
        var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0, maxX: CGFloat = 0
        for s in subviews {
            let sz = s.sizeThatFits(.unspecified)
            if x + sz.width > width, x > 0 { x = 0; y += rowH + spacing; rowH = 0 }
            x += sz.width + spacing
            maxX = max(maxX, x)
            rowH = max(rowH, sz.height)
        }
        return CGSize(width: min(maxX, width), height: y + rowH)
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowH: CGFloat = 0
        for s in subviews {
            let sz = s.sizeThatFits(.unspecified)
            if x + sz.width > bounds.maxX, x > bounds.minX { x = bounds.minX; y += rowH + spacing; rowH = 0 }
            s.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(sz))
            x += sz.width + spacing
            rowH = max(rowH, sz.height)
        }
    }
}

// MARK: - Colors

enum Palette {
    static let tokenColors: [Color] = [.blue, .orange, .green, .pink, .purple, .teal, .yellow, .red, .mint, .indigo, .brown, .cyan]
    static func token(_ i: Int) -> Color { tokenColors[i % tokenColors.count] }

    /// Blue (negative) → white → red (positive)
    static func diverging(_ v: Double, scale: Double) -> Color {
        let t = max(-1, min(1, v / max(scale, 1e-9)))
        return t >= 0
            ? Color(red: 1, green: 1 - 0.75 * t, blue: 1 - 0.8 * t)
            : Color(red: 1 + 0.8 * t, green: 1 + 0.55 * t, blue: 1)
    }

    /// light → deep blue
    static func sequential(_ v: Double) -> Color {
        let t = max(0, min(1, v))
        return Color(red: 0.96 - 0.85 * t, green: 0.97 - 0.6 * t, blue: 1 - 0.25 * t)
    }
}

// MARK: - Heatmap

struct Heatmap: View {
    let values: [[Double]]
    var rowLabels: [String] = []
    var colLabels: [String] = []
    var diverging = false
    var cell: CGFloat = 18
    var highlightRow: Int? = nil
    var showNumbers = false
    var onSelectRow: ((Int) -> Void)? = nil

    var body: some View {
        let rows = values.count, cols = values.first?.count ?? 0
        let scale = values.flatMap { $0 }.map(abs).max() ?? 1
        let labelW: CGFloat = rowLabels.isEmpty ? 0 : 22
        let labelH: CGFloat = colLabels.isEmpty ? 0 : 18
        Canvas { ctx, _ in
            for r in 0..<rows {
                for c in 0..<cols {
                    let v = values[r][c]
                    let rect = CGRect(x: labelW + CGFloat(c) * cell, y: labelH + CGFloat(r) * cell, width: cell - 1, height: cell - 1)
                    let color = diverging ? Palette.diverging(v, scale: scale) : Palette.sequential(v / max(scale, 1e-9))
                    ctx.fill(Path(rect), with: .color(color))
                    if showNumbers && cell >= 26 {
                        ctx.draw(Text(String(format: "%.2f", v)).font(.system(size: 8)).foregroundColor(.black),
                                 at: CGPoint(x: rect.midX, y: rect.midY))
                    }
                }
                if r < rowLabels.count {
                    ctx.draw(Text(rowLabels[r]).font(.system(size: 11, design: .monospaced)),
                             at: CGPoint(x: labelW / 2, y: labelH + (CGFloat(r) + 0.5) * cell))
                }
            }
            for c in 0..<min(cols, colLabels.count) {
                ctx.draw(Text(colLabels[c]).font(.system(size: 11, design: .monospaced)),
                         at: CGPoint(x: labelW + (CGFloat(c) + 0.5) * cell, y: labelH / 2))
            }
            if let h = highlightRow, h < rows {
                let rect = CGRect(x: labelW - 1, y: labelH + CGFloat(h) * cell - 1, width: CGFloat(cols) * cell + 1, height: cell + 1)
                ctx.stroke(Path(roundedRect: rect, cornerRadius: 2), with: .color(.orange), lineWidth: 2)
            }
        }
        .frame(width: labelW + CGFloat(cols) * cell, height: labelH + CGFloat(rows) * cell)
        .contentShape(Rectangle())
        .gesture(SpatialTapGesture().onEnded { e in
            let r = Int((e.location.y - labelH) / cell)
            if r >= 0 && r < rows { onSelectRow?(r) }
        })
    }
}

// MARK: - Probability bars

struct ProbabilityBars: View {
    struct Item: Identifiable { let id: Int; let label: String; let p: Double }
    let items: [Item]
    var highlight: Int? = nil
    var color: Color = .blue
    var height: CGFloat = 200

    var body: some View {
        Chart(items) { it in
            BarMark(x: .value("Token", it.label), y: .value("Probability", it.p))
                .foregroundStyle(it.id == highlight ? Color.orange : color.opacity(0.75))
                .annotation(position: .top) {
                    Text(it.p >= 0.01 ? String(format: "%.0f%%", it.p * 100) : "")
                        .font(.system(size: 9)).foregroundStyle(.secondary)
                }
        }
        .chartYScale(domain: 0...1)
        .chartYAxis { AxisMarks(values: [0, 0.25, 0.5, 0.75, 1]) { v in
            AxisGridLine(); AxisValueLabel { Text("\(Int((v.as(Double.self) ?? 0) * 100))%") }
        } }
        .chartXAxis { AxisMarks { _ in AxisValueLabel().font(.system(size: 12, design: .monospaced)) } }
        .frame(height: height)
    }
}

// MARK: - Model status

/// Shown on chapters that look inside the tiny model, so readers know whether it has been trained.
struct ModelStatusBanner: View {
    @Environment(AppState.self) private var app
    var body: some View {
        let t = app.trainer
        HStack(spacing: 12) {
            Image(systemName: t.isTrained ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(t.isTrained ? .green : .orange)
                .font(.title3)
            VStack(alignment: .leading, spacing: 2) {
                Text(t.isTrained
                     ? "Using the tiny model after \(t.step) training steps."
                     : (t.step == 0 ? "The tiny model is untrained — its insides are random noise." : "The tiny model has only trained \(t.step) steps."))
                    .font(.subheadline.weight(.semibold))
                Text("Everything below is computed live by a real \(Format.big(Double(t.parameterCount)))-parameter transformer running on this Mac.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if t.isRunning {
                ProgressView().controlSize(.small)
                Text("training… step \(t.step)").font(.caption).monospacedDigit()
            } else if !t.finished {
                Button(t.isTrained ? "Keep training" : "Train it now") { t.start() }
                Button("Watch it learn") { app.selected = .trainingLoop }.buttonStyle(.link)
            }
        }
        .padding(12)
        .background((t.isTrained ? Color.green : Color.orange).opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }
}

/// A row of tappable stage boxes joined by arrows.
struct FlowRow: View {
    struct Step: Identifiable { let id = UUID(); let title: String; let detail: String; let color: Color; var chapter: Chapter? = nil }
    let steps: [Step]
    @Environment(AppState.self) private var app
    var body: some View {
        HStack(spacing: 6) {
            ForEach(Array(steps.enumerated()), id: \.element.id) { i, s in
                Button { if let c = s.chapter { app.selected = c } } label: {
                    VStack(spacing: 4) {
                        Text(s.title).font(.system(size: 13, weight: .semibold)).multilineTextAlignment(.center)
                        Text(s.detail).font(.system(size: 10.5)).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, minHeight: 74)
                    .background(s.color.opacity(0.14), in: RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(s.color.opacity(0.5)))
                }
                .buttonStyle(.plain)
                .help(s.chapter.map { "Open chapter: \($0.title)" } ?? "")
                if i < steps.count - 1 {
                    Image(systemName: "arrow.right").foregroundStyle(.secondary).font(.caption)
                }
            }
        }
    }
}

// MARK: - Animation ticks

extension View {
    /// Calls `action` every `interval` seconds, but only while `active` is true,
    /// so a page that isn't animating uses no CPU (and lets the app go idle).
    func ticking(every interval: Double, while active: Bool, _ action: @escaping @MainActor () -> Void) -> some View {
        task(id: active) {
            guard active else { return }
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(interval))
                if Task.isCancelled { break }
                action()
            }
        }
    }
}
