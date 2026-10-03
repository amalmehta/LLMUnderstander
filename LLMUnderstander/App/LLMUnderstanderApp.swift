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
                ForEach(Chapter.allCases) { c in
                    if c.number <= 9 {
                        Button("\(c.number). \(c.title)") { app.selected = c }
                            .keyboardShortcut(KeyEquivalent(Character("\(c.number)")), modifiers: .command)
                    } else {
                        Button("\(c.number). \(c.title)") { app.selected = c }
                    }
                }
            }
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
                    Section(section.rawValue) {
                        ForEach(Chapter.inSection(section)) { c in
                            Label {
                                Text("\(c.number). \(c.title)")
                            } icon: {
                                Image(systemName: c.symbol).foregroundStyle(section.color)
                            }
                            .tag(c)
                        }
                    }
                }
            }
            .navigationSplitViewColumnWidth(min: 230, ideal: 250)
            .safeAreaInset(edge: .bottom) { FeedbackTab() }
        } detail: {
            ChapterView(chapter: app.selected ?? .bigPicture)
                .id(app.selected)
        }
        .sheet(isPresented: $app.showFeedback) { FeedbackSheet() }
    }
}

struct ChapterView: View {
    let chapter: Chapter
    var body: some View {
        switch chapter {
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
        case .pipeline: PipelineChapter()
        }
    }
}
