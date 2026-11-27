import SwiftUI

@main
struct LLMUnderstanderApp: App {
    @State private var app = AppState()

    var body: some Scene {
        WindowGroup("LLM Understander") {
            ContentView()
                .environment(app)
                .frame(minWidth: 1050, minHeight: 700)
                .task { await Snapshotter.runIfRequested(app) }
        }
        .defaultSize(width: 1320, height: 880)
        .commands {
            CommandGroup(replacing: .help) {
                Button("Send Feedback…") { app.showFeedback = true }
            }
            CommandMenu("Chapters") {
                Button("Previous Chapter") { app.selected = (app.selected ?? .tour).previous ?? app.selected }
                    .keyboardShortcut("[", modifiers: .command)
                    .disabled((app.selected ?? .tour).previous == nil)
                Button("Next Chapter") { app.selected = (app.selected ?? .tour).next ?? app.selected }
                    .keyboardShortcut("]", modifiers: .command)
                    .disabled((app.selected ?? .tour).next == nil)
                Divider()
                ForEach(Chapter.allCases) { c in
                    if c.number <= 9 {
                        Button(c.menuTitle) { app.selected = c }
                            .keyboardShortcut(KeyEquivalent(Character("\(c.number)")), modifiers: .command)
                    } else {
                        Button(c.menuTitle) { app.selected = c }
                    }
                }
            }
        }

        Settings {
            SettingsView()
                .environment(app)
        }
    }
}

struct ContentView: View {
    @Environment(AppState.self) private var app

    var body: some View {
        @Bindable var app = app
        NavigationSplitView {
            List(selection: $app.selected) {
                ForEach(ChapterSection.allCases) { section in
                    let pages = Chapter.inSection(section)
                    if section == .tour {
                        ForEach(pages) { c in
                            Label(c.title, systemImage: c.symbol)
                                .foregroundStyle(.primary)
                                .tag(c)
                        }
                    } else {
                        Section(section.rawValue) {
                            ForEach(pages) { c in SidebarRow(chapter: c).tag(c) }
                        }
                    }
                }
            }
            .navigationSplitViewColumnWidth(min: 230, ideal: 250)
            .safeAreaInset(edge: .bottom) { FeedbackTab() }
        } detail: {
            ChapterView(chapter: app.selected ?? .tour)
                .id(app.selected)
        }
        .sheet(isPresented: $app.showFeedback) { FeedbackSheet() }
        .overlay { OpeningAnimation() }
    }
}

private struct SidebarRow: View {
    @Environment(AppState.self) private var app
    let chapter: Chapter

    var body: some View {
        let done = app.read.contains(chapter.key)
        Label {
            HStack {
                Text(chapter.menuTitle)
                Spacer(minLength: 4)
                if done {
                    Image(systemName: "checkmark")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.secondary)
                        .transition(.scale.combined(with: .opacity))
                }
            }
            .animation(.easeOut(duration: 0.2), value: done)
        } icon: {
            Image(systemName: chapter.symbol).foregroundStyle(chapter.section.color)
        }
        .accessibilityValue(done ? "read" : "")
    }
}

struct ChapterView: View {
    let chapter: Chapter
    var body: some View {
        switch chapter {
        case .tour: TourView()
        case .bigPicture: BigPictureChapter()
        case .tokens: TokensChapter()
        case .embeddings: EmbeddingsChapter()
        case .attention: AttentionChapter()
        case .block: BlockChapter()
        case .nextToken: NextTokenChapter()
        case .backprop: BackpropChapter()
        case .optimizer: OptimizerChapter()
        case .trainingLoop: TrainingLoopChapter()
        case .pretraining: PretrainingChapter()
        case .sft: SFTChapter()
        case .preferences: PreferencesChapter()
        case .reasoning: ReasoningChapter()
        case .evals: EvalsChapter()
        case .pipeline: PipelineChapter()
        case .inference: InferenceChapter()
        case .agentLoop: AgentLoopChapter()
        case .harness: HarnessChapter()
        case .toolUse: ToolUseChapter()
        case .agentTraining: AgentTrainingChapter()
        case .workflows: WorkflowsChapter()
        }
    }
}

/// The app icon springs in and fades away once per launch. It never takes clicks,
/// and with Reduce Motion on it simply fades.
private struct OpeningAnimation: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase = 0   // 0 hidden, 1 shown, 2 gone

    var body: some View {
        ZStack {
            if phase < 2 && !AppState.isAutomated {
                Color(nsColor: .windowBackgroundColor)
                    .opacity(phase == 0 ? 1 : 0.85)
                VStack(spacing: 12) {
                    Image(nsImage: NSApp.applicationIconImage)
                        .resizable().frame(width: 112, height: 112)
                        .scaleEffect(reduceMotion ? 1 : (phase == 1 ? 1 : 0.6))
                    Text("LLM Understander").font(.system(size: 22, weight: .semibold, design: .rounded))
                        .opacity(phase == 1 ? 1 : 0)
                }
                .opacity(phase == 1 ? 1 : 0)
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
        .task {
            guard !AppState.isAutomated else { return }
            withAnimation(reduceMotion ? .easeOut(duration: 0.2) : .spring(response: 0.45, dampingFraction: 0.65)) { phase = 1 }
            try? await Task.sleep(for: .milliseconds(650))
            withAnimation(.easeInOut(duration: 0.3)) { phase = 2 }
        }
    }
}
