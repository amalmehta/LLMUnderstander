import SwiftUI

struct AgentLoopChapter: View {
    @State private var step = 2
    @State private var playing = false
    private let events = AgentScript.events

    var body: some View {
        ChapterPage(chapter: .agentLoop) {
            Prose("Everything so far produces a model that does one thing: given text, predict the next token. It can't open a file, run code or search the web. It can only *write*.")
            Prose("An **agent** is that same model placed inside a loop by a program called a **harness**. The model writes a request for a tool, the harness runs the tool, and the result goes back into the model's context. Then the model is asked again. The loop repeats until the model answers without asking for a tool.")

            Card(title: "The agent loop") {
                AgentLoopDiagram()
                    .frame(height: 210)
                Formula("""
                context = [system_prompt, tool_definitions, user_message]
                while True:
                    reply = model.generate(context)          # next-token prediction, as in Chapter 6
                    context.append(reply)
                    if not reply.tool_calls: break           # plain answer → done
                    for call in reply.tool_calls:
                        if harness.allowed(call):            # permissions live here, not in the model
                            context.append(harness.run(call))  # tool result becomes new context
                show(reply)
                """)
            }

            Card(title: "Watch a real-shaped agent run", subtitle: "A coding agent fixes a failing test. Step through every message that passes between harness, model and tools.") {
                HStack {
                    Button("Back") { step = max(0, step - 1); playing = false }.disabled(step == 0)
                    Button("Next") { step = min(events.count - 1, step + 1) }
                        .buttonStyle(.borderedProminent).tint(.green).disabled(step == events.count - 1)
                    Button(playing ? "Pause" : "Play") {
                        if step == events.count - 1 { step = 0 }
                        playing.toggle()
                    }
                    Button("Restart") { step = 0; playing = false }
                    Spacer()
                    StatTile(label: "Step", value: "\(step + 1) / \(events.count)", color: .green)
                    StatTile(label: "Context so far", value: "\(AgentScript.contextTokens(upTo: step)) tokens", color: .blue)
                    StatTile(label: "Model calls", value: "\(modelCalls(upTo: step))", color: .orange)
                }

                Text(.init(events[step].note))
                    .font(.system(size: 15))
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.green.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

                HStack(spacing: 0) {
                    laneHeader("User & Harness", "person.2", .gray)
                    laneHeader("Model", "brain", .orange)
                    laneHeader("Tools", "wrench.and.screwdriver", .blue)
                }
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(spacing: 8) {
                            ForEach(events.prefix(step + 1)) { e in
                                HStack(spacing: 0) {
                                    lane(e, for: [.user, .harness])
                                    lane(e, for: [.model])
                                    lane(e, for: [.tool])
                                }
                                .id(e.id)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    .frame(height: 380)
                    .onChange(of: step) { withAnimation { proxy.scrollTo(step, anchor: .bottom) } }
                }
            }
            .ticking(every: 1.6, while: playing) {
                if step < events.count - 1 { withAnimation { step += 1 } } else { playing = false }
            }

            KeyIdea("The model never runs anything. It writes text that *describes* an action; the harness decides whether to do it and reports back. An agent is a language model, a loop, and a set of tools — nothing more mysterious than that.", color: .green)

            GoDeeper(title: "what changes compared with a chatbot") {
                Prose("**Many model calls per request.** This small task took \(AgentScript.modelCalls) separate calls to the model. Each call re-reads the entire context (prompt caching makes that cheaper), so agents use far more tokens than chat.")
                Prose("**The context is the agent's only working memory.** Everything the model knows about the task — the error message, the file, its own earlier plan — must be in the context window. Chapter 15 is about managing that.")
                Prose("**Errors compound.** If each step is right 95% of the time, a 20-step task succeeds only about 36% of the time (0.95²⁰). That's why checking work, like re-running the tests here, matters so much, and why agents are trained specifically for long tasks (Chapter 17).")
            }
        }
    }

    private func modelCalls(upTo i: Int) -> Int {
        var n = 0, prev = false
        for e in events.prefix(i + 1) { let m = e.actor == .model; if m && !prev { n += 1 }; prev = m }
        return n
    }

    private func laneHeader(_ t: String, _ icon: String, _ c: Color) -> some View {
        Label(t, systemImage: icon).font(.caption.weight(.semibold)).foregroundStyle(c)
            .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private func lane(_ e: AgentScript.Event, for actors: [AgentScript.Actor]) -> some View {
        if actors.contains(e.actor) {
            EventBubble(event: e, highlighted: e.id == step)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 4)
        } else {
            Color.clear.frame(maxWidth: .infinity, maxHeight: 1)
        }
    }
}

/// One message in an agent transcript.
struct EventBubble: View {
    let event: AgentScript.Event
    var highlighted = false
    var dimmed = false

    static func color(_ k: AgentScript.Kind) -> Color {
        switch k {
        case .system, .tools: .gray
        case .user: .purple
        case .text, .final: .orange
        case .toolCall: .pink
        case .permission: .yellow
        case .toolResult: .blue
        }
    }

    var body: some View {
        let c = Self.color(event.kind)
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(event.kind.rawValue.uppercased()).font(.system(size: 9, weight: .bold)).foregroundStyle(c)
                Spacer()
                Text("~\(event.tokens) tok").font(.system(size: 9)).foregroundStyle(.secondary)
            }
            Text(event.content)
                .font(.system(size: 11.5, design: event.kind == .toolCall || event.kind == .toolResult || event.kind == .tools ? .monospaced : .default))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(c.opacity(highlighted ? 0.22 : 0.09), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(c.opacity(highlighted ? 0.9 : 0.25), lineWidth: highlighted ? 2 : 1))
        .opacity(dimmed ? 0.4 : 1)
    }
}

/// User ⇄ Harness ⇄ Model, Harness ⇄ Tools.
struct AgentLoopDiagram: View {
    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 0) {
                box("User", "asks for something,\napproves risky actions", "person.fill", .purple)
                arrows("request", "answer")
                box("Harness", "builds the context,\nruns the loop, checks\npermissions, runs tools", "shippingbox.fill", .green)
                arrows("context", "text + tool calls")
                box("Model", "next-token prediction —\nwrites text or a\ntool call", "brain", .orange)
            }
            // The harness is the middle of three equal boxes, so centring puts Tools right under it.
            HStack(spacing: 4) {
                Image(systemName: "arrow.up.arrow.down").foregroundStyle(.secondary)
                Text("calls / results").font(.system(size: 9)).foregroundStyle(.secondary)
            }
            box("Tools", "files, shell, tests, search, browser…", "wrench.and.screwdriver.fill", .blue)
                .frame(width: 230)
        }
    }

    private func box(_ t: String, _ d: String, _ icon: String, _ c: Color) -> some View {
        VStack(spacing: 4) {
            Label(t, systemImage: icon).font(.headline).foregroundStyle(c)
            Text(d).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .padding(10)
        .frame(maxWidth: .infinity, minHeight: 80)
        .background(c.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(c.opacity(0.4)))
    }

    private func arrows(_ top: String, _ bottom: String) -> some View {
        VStack(spacing: 2) {
            Text(top).font(.system(size: 9)).foregroundStyle(.secondary)
            Image(systemName: "arrow.right")
            Image(systemName: "arrow.left")
            Text(bottom).font(.system(size: 9)).foregroundStyle(.secondary)
        }
        .frame(width: 92)
    }
}
