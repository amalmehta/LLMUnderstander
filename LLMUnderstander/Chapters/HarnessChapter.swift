import SwiftUI
import Charts

struct HarnessChapter: View {
    @State private var part: Part = .context
    @State private var turns = 40.0
    @State private var compaction = true
    @State private var policy: Policy = .reads

    enum Part: String, CaseIterable, Identifiable {
        case system, tools, loop, permissions, memory, context, subagents, hooks
        var id: String { rawValue }
        var title: String {
            switch self {
            case .system: "System prompt"
            case .tools: "Tool definitions"
            case .loop: "The loop"
            case .permissions: "Permissions & sandbox"
            case .memory: "Memory files"
            case .context: "Context management"
            case .subagents: "Sub-agents"
            case .hooks: "Hooks & integrations"
            }
        }
        var icon: String {
            switch self {
            case .system: "text.quote"
            case .tools: "wrench.and.screwdriver"
            case .loop: "arrow.triangle.2.circlepath"
            case .permissions: "lock.shield"
            case .memory: "brain.head.profile"
            case .context: "rectangle.compress.vertical"
            case .subagents: "person.3"
            case .hooks: "link"
            }
        }
        var text: LocalizedStringKey {
            switch self {
            case .system: "The first text the model sees on every call: who it is, how to behave, the conventions to follow. It's the cheapest, most powerful lever a harness has. Change it and the same weights behave like a different product."
            case .tools: "A name, a description and a JSON schema for each tool. The model chooses tools *only* from these descriptions, so writing clear ones is a big part of building a good agent. Every definition costs context tokens on every call."
            case .loop: "Call the model; if it asked for tools, run them, append the results and call again; stop when it answers in plain text. Also: retries on errors, a cap on the number of steps, streaming output to the user, and handling interruptions."
            case .permissions: "The model proposes; the harness disposes. Rules decide which tool calls run automatically, which need a human's approval and which are forbidden. Running tools inside a **sandbox** (a container or restricted file system) limits the damage of a bad call."
            case .memory: "Models remember nothing between sessions; the weights are frozen. Harnesses fake long-term memory by saving notes to files (project instructions, preferences, past decisions) and loading them into the context at the start of each session."
            case .context: "The context window is finite (here 200k tokens), and agent runs fill it fast with tool output. Harnesses truncate huge tool results, drop stale ones, and **compact**: replace old conversation with a model-written summary so work can continue."
            case .subagents: "A main agent can hand a self-contained job (\"search the codebase for X\") to a fresh model call with its own clean context. Only the short conclusion comes back, which keeps the main context small and lets work run in parallel."
            case .hooks: "Deterministic code that runs at fixed points: before a tool call (block a dangerous command), after an edit (run the formatter), at the end of a turn (run the tests). Integrations such as MCP servers plug in extra tools — GitHub, databases, browsers — without retraining the model."
            }
        }
    }

    enum Policy: String, CaseIterable, Identifiable {
        case askAll = "Ask before every tool"
        case reads = "Allow reads, ask for writes"
        case sandbox = "Allow all, inside a sandbox"
        var id: String { rawValue }
    }

    var body: some View {
        ChapterPage(chapter: .harness) {
            Prose("The **harness** is everything around the model: the program that assembles the prompt, offers tools, runs the loop, enforces permissions and manages memory. The weights decide *how well* the agent thinks; the harness decides what it can see and do.")
            Prose("Put the same model in two different harnesses and you get two noticeably different agents. Much of the progress in practical agents has come from better harnesses, not just better models.")

            Card(title: "Anatomy of a harness", subtitle: "Click a part") {
                HStack(alignment: .top, spacing: 20) {
                    LazyVGrid(columns: [GridItem(.fixed(170)), GridItem(.fixed(170))], spacing: 8) {
                        ForEach(Part.allCases) { p in
                            Button { part = p } label: {
                                Label(p.title, systemImage: p.icon)
                                    .font(.system(size: 12.5, weight: .semibold))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(10)
                                    .background(Color.green.opacity(part == p ? 0.25 : 0.08), in: RoundedRectangle(cornerRadius: 8))
                                    .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.green.opacity(part == p ? 0.9 : 0.25)))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .frame(width: 350)
                    VStack(alignment: .leading, spacing: 8) {
                        Label(part.title, systemImage: part.icon).font(.title3.weight(.semibold))
                        Text(part.text).font(.system(size: 15)).lineSpacing(3).fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            contextCard
            permissionCard

            KeyIdea("Model = what it can think. Harness = what it can see and do. The harness turns a text predictor into a careful, useful worker by controlling its context, its tools and its permissions.", color: .green)

            GoDeeper(title: "engineering tricks") {
                Prose("**Prompt caching:** the start of the context (system prompt, tool definitions, early conversation) stays the same from call to call, so providers can cache its processed form. Re-sending it then costs a fraction of the price and time. Harnesses keep the context's beginning stable to benefit.")
                Prose("**Just-in-time context:** rather than stuffing everything in up front, give the agent tools to *look things up* (search, list files, read a range of lines) so only what's needed enters the context.")
                Prose("**Structured notes:** for long tasks, agents keep a running to-do list or progress file. It survives compaction and keeps the plan on track.")
            }
        }
    }

    // MARK: - Context budget

    private var contextCard: some View {
        let r = ContextBudget.segments(turns: Int(turns), compaction: compaction)
        let used = r.segments.reduce(0) { $0 + $1.tokens }
        let colors: [String: Color] = [
            "System prompt": .gray, "Tool definitions": .pink, "Memory files": .purple,
            "Summary of earlier work": .teal, "Conversation + tool results": .blue,
        ]
        let history: [(Int, Double)] = stride(from: 0, through: 100, by: 2).map { t in
            (t, ContextBudget.segments(turns: t, compaction: compaction).segments.reduce(0) { $0 + $1.tokens })
        }
        return Card(title: "The context window fills up", subtitle: "A long simulated session. Each turn adds messages and tool output (some tool outputs are huge).") {
            HStack {
                Text("After \(Int(turns)) turns").monospacedDigit().frame(width: 120, alignment: .leading)
                Slider(value: $turns, in: 0...100, step: 1)
                Toggle("Compaction", isOn: $compaction).toggleStyle(.switch)
            }
            .font(.callout)
            GeometryReader { g in
                HStack(spacing: 0) {
                    ForEach(r.segments.filter { $0.tokens > 0 }) { s in
                        Rectangle().fill((colors[s.id] ?? .gray).opacity(0.7))
                            .frame(width: g.size.width * min(s.tokens, ContextBudget.window) / max(ContextBudget.window, used))
                    }
                    Spacer(minLength: 0)
                }
                .overlay(alignment: .leading) {
                    Rectangle().fill(Color.red).frame(width: 2)
                        .offset(x: g.size.width * ContextBudget.window / max(ContextBudget.window, used) - 1)
                }
            }
            .frame(height: 30)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            HStack(spacing: 14) {
                ForEach(r.segments) { s in
                    HStack(spacing: 5) {
                        RoundedRectangle(cornerRadius: 2).fill((colors[s.id] ?? .gray).opacity(0.7)).frame(width: 12, height: 10)
                        Text("\(s.id) \(Format.big(s.tokens))").font(.caption)
                    }
                }
            }
            Chart {
                ForEach(history, id: \.0) { t, u in
                    LineMark(x: .value("turn", t), y: .value("tokens", u)).foregroundStyle(.blue)
                }
                RuleMark(y: .value("window", ContextBudget.window)).foregroundStyle(.red).lineStyle(StrokeStyle(dash: [5, 4]))
                    .annotation(position: .top, alignment: .leading) { Text("context limit 200K").font(.caption2).foregroundStyle(.red) }
                RuleMark(x: .value("now", turns)).foregroundStyle(.gray.opacity(0.5))
            }
            .chartYAxis { AxisMarks { v in AxisGridLine(); AxisValueLabel { Text(Format.big(v.as(Double.self) ?? 0)) } } }
            .chartXAxisLabel("turn")
            .frame(height: 160)
            HStack(spacing: 10) {
                StatTile(label: "Context used", value: String(format: "%.0f%%", 100 * used / ContextBudget.window), color: r.overflow ? .red : .blue)
                StatTile(label: "Compactions so far", value: "\(r.compactions)", color: .teal)
                if r.overflow {
                    Text("⚠️ Over the limit: without compaction the harness would have to drop messages or stop the task.")
                        .font(.callout).foregroundStyle(.red)
                }
            }
            Text("The system prompt, tool definitions and memory are paid for on *every* call. Compaction (at 80% full here) has the model summarise older work, then continues with the summary plus recent messages. Some detail is lost each time, which is why good notes matter.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    // MARK: - Permissions

    private var permissionCard: some View {
        let calls = AgentScript.events.filter { $0.kind == .toolCall }
        return Card(title: "Permissions: who gets asked?", subtitle: "The tool calls from Chapter 14's run, under three policies") {
            Picker("", selection: $policy) { ForEach(Policy.allCases) { Text($0.rawValue).tag($0) } }
                .pickerStyle(.segmented).labelsHidden()
            VStack(spacing: 6) {
                ForEach(calls) { c in
                    let isWrite = c.content.contains("edit_file")
                    let outcome = outcome(isWrite: isWrite)
                    HStack {
                        Text(c.content).font(.system(size: 11.5, design: .monospaced)).lineLimit(1).truncationMode(.tail)
                        Spacer()
                        Label(outcome.0, systemImage: outcome.1)
                            .font(.caption.weight(.semibold)).foregroundStyle(outcome.2)
                            .frame(width: 230, alignment: .leading)
                    }
                    .padding(8)
                    .background(outcome.2.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
                }
            }
            let asks = calls.filter { outcome(isWrite: $0.content.contains("edit_file")).0.hasPrefix("Asks") }.count
            Text(policy == .sandbox
                 ? "No interruptions. The agent works on its own, and the sandbox limits the damage if it does something wrong."
                 : "\(asks) interruption\(asks == 1 ? "" : "s") in a 4-call task. Safer, but a long task can mean dozens of approvals. That's why reads are usually auto-allowed and only risky actions ask.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    private func outcome(isWrite: Bool) -> (String, String, Color) {
        switch policy {
        case .askAll: ("Asks the user", "hand.raised.fill", .orange)
        case .reads: isWrite ? ("Asks the user", "hand.raised.fill", .orange) : ("Runs automatically", "checkmark.circle.fill", .green)
        case .sandbox: ("Runs automatically (sandboxed)", "shippingbox.fill", .teal)
        }
    }
}
