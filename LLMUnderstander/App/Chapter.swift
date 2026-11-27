import SwiftUI

enum ChapterSection: String, CaseIterable, Identifiable {
    case tour = "Tour"
    case start = "Start Here"
    case blocks = "Building Blocks"
    case learning = "How It Learns"
    case pretraining = "Pretraining"
    case posttraining = "Post-training"
    case wrap = "Putting It Together"
    case agents = "Agents & Harnesses"
    var id: String { rawValue }

    var color: Color {
        switch self {
        case .tour: .indigo
        case .start: .indigo
        case .blocks: .blue
        case .learning: .teal
        case .pretraining: .orange
        case .posttraining: .pink
        case .wrap: .purple
        case .agents: .green
        }
    }
}

/// Every page in reading order. The Tour is page 0, so a chapter's number is its raw value.
enum Chapter: Int, CaseIterable, Identifiable, Hashable {
    case tour
    case bigPicture
    case tokens, embeddings, attention, block
    case nextToken, backprop, optimizer, trainingLoop
    case pretraining
    case sft, preferences, reasoning
    case evals, pipeline, inference
    case agentLoop, harness, toolUse, agentTraining, workflows

    var id: Int { rawValue }
    var number: Int { rawValue }
    var isTour: Bool { self == .tour }

    /// The numbered chapters (everything but the Tour).
    static var chapters: [Chapter] { allCases.filter { !$0.isTour } }

    /// A stable name for saved progress and web links; numbers change when chapters are added.
    var key: String {
        switch self {
        case .tour: "tour"
        case .bigPicture: "big-picture"
        case .tokens: "tokens"
        case .embeddings: "embeddings"
        case .attention: "attention"
        case .block: "block"
        case .nextToken: "next-token"
        case .backprop: "backprop"
        case .optimizer: "optimizer"
        case .trainingLoop: "training-loop"
        case .pretraining: "pretraining"
        case .sft: "sft"
        case .preferences: "preferences"
        case .reasoning: "reasoning"
        case .evals: "evals"
        case .pipeline: "pipeline"
        case .inference: "inference"
        case .agentLoop: "agent-loop"
        case .harness: "harness"
        case .toolUse: "tool-use"
        case .agentTraining: "agent-training"
        case .workflows: "workflows"
        }
    }

    /// How the page is named in menus and links: "4. Attention", or just "Tour".
    var menuTitle: String { isTour ? title : "\(number). \(title)" }

    var title: String {
        switch self {
        case .tour: "Tour"
        case .bigPicture: "The Big Picture"
        case .tokens: "Text into Tokens"
        case .embeddings: "Embeddings"
        case .attention: "Attention"
        case .block: "The Transformer Block"
        case .nextToken: "Predicting the Next Token"
        case .backprop: "Backpropagation"
        case .optimizer: "The Optimizer"
        case .trainingLoop: "The Training Loop"
        case .pretraining: "Pretraining at Scale"
        case .sft: "Supervised Fine-Tuning"
        case .preferences: "Learning from Preferences"
        case .reasoning: "Reasoning Models"
        case .evals: "Evals"
        case .pipeline: "The Full Pipeline"
        case .inference: "Inference"
        case .agentLoop: "From Model to Agent"
        case .harness: "The Harness"
        case .toolUse: "Tool Use, Step by Step"
        case .agentTraining: "Training Models to Be Agents"
        case .workflows: "Workflows & Bandits"
        }
    }

    var tagline: String {
        switch self {
        case .tour: "Follow one sentence through the whole machine. Scroll to move it along."
        case .bigPicture: "An LLM is one simple loop, run trillions of times, in three stages."
        case .tokens: "Models can't read letters — text becomes a list of numbers first."
        case .embeddings: "Each token number becomes a vector the model can do math with."
        case .attention: "Each token looks back at earlier tokens and borrows what it needs."
        case .block: "Attention + a small network + shortcuts, stacked many times."
        case .nextToken: "The whole job: guess the next token. The loss scores the guess."
        case .backprop: "The chain rule tells every weight which way to move."
        case .optimizer: "How big a step to take, and in which direction, for every weight."
        case .trainingLoop: "Batch → forward → loss → backward → update. Watch a real model learn."
        case .pretraining: "Same loop, but on trillions of tokens across thousands of GPUs."
        case .sft: "Teach the text predictor to act like an assistant, using examples."
        case .preferences: "Teach it what people prefer: reward models, RLHF and DPO."
        case .reasoning: "Reward right answers, not nice-sounding ones, and the model learns to think longer."
        case .evals: "How do you know training worked? Scores, the noise in them, and the traps."
        case .pipeline: "Every stage on one page: what changes, and what stays the same."
        case .inference: "Using the trained model: sampling, the KV cache, and getting tokens out faster."
        case .agentLoop: "A model only writes text. Put it in a loop with tools and it can act."
        case .harness: "Everything around the model: prompt, tools, permissions, memory, context."
        case .toolUse: "Schema in, JSON out, result back in — the anatomy of one tool call."
        case .agentTraining: "Teach tool use with transcripts, then sharpen it with RL on real tasks."
        case .workflows: "Wiring model calls into patterns — and choosing between options when you can't be sure which is best."
        }
    }

    var symbol: String {
        switch self {
        case .tour: "play.circle"
        case .bigPicture: "map"
        case .tokens: "textformat.abc"
        case .embeddings: "square.grid.3x3"
        case .attention: "eye"
        case .block: "square.stack.3d.up"
        case .nextToken: "arrow.right.to.line"
        case .backprop: "arrow.uturn.backward"
        case .optimizer: "figure.hiking"
        case .trainingLoop: "arrow.triangle.2.circlepath"
        case .pretraining: "server.rack"
        case .sft: "person.fill.checkmark"
        case .preferences: "hand.thumbsup"
        case .reasoning: "brain"
        case .evals: "checklist"
        case .pipeline: "flowchart"
        case .inference: "bolt"
        case .agentLoop: "arrow.triangle.turn.up.right.circle"
        case .harness: "shippingbox"
        case .toolUse: "wrench.and.screwdriver"
        case .agentTraining: "figure.strengthtraining.traditional"
        case .workflows: "point.3.connected.trianglepath.dotted"
        }
    }

    var section: ChapterSection {
        switch self {
        case .tour: .tour
        case .bigPicture: .start
        case .tokens, .embeddings, .attention, .block: .blocks
        case .nextToken, .backprop, .optimizer, .trainingLoop: .learning
        case .pretraining: .pretraining
        case .sft, .preferences, .reasoning: .posttraining
        case .evals, .pipeline, .inference: .wrap
        case .agentLoop, .harness, .toolUse, .agentTraining, .workflows: .agents
        }
    }

    var next: Chapter? { Chapter(rawValue: rawValue + 1) }
    var previous: Chapter? { Chapter(rawValue: rawValue - 1) }

    static func inSection(_ s: ChapterSection) -> [Chapter] { allCases.filter { $0.section == s } }
}

@MainActor
@Observable
final class AppState {
    var selected: Chapter? {
        didSet {
            if let selected, !AppState.isAutomated { defaults.set(selected.key, forKey: Keys.last) }
        }
    }
    var showFeedback = false
    /// Chapter keys the reader has read (20 seconds open, or scrolled to the end).
    private(set) var read: Set<String>
    /// The option the reader picked on each chapter's guess, by chapter key.
    private(set) var guesses: [String: Int]

    /// Screenshot runs and tests start from a fresh, unsaved model so they're repeatable.
    let trainer = Trainer(saveURL: AppState.isAutomated ? nil : Trainer.defaultSaveURL)
    /// Fine-tuning the reader's own model in the SFT chapter; kept here so it survives changing chapters.
    let chatTuning = ChatTuningSession()

    @ObservationIgnored private let defaults: UserDefaults
    private enum Keys {
        static let last = "lastChapter", read = "readChapters", guesses = "guesses"
    }

    init() {
        // Tests and screenshots use a throwaway store so they never touch the reader's progress.
        defaults = AppState.isAutomated ? (UserDefaults(suiteName: "LLMUnderstander.automated") ?? .standard) : .standard
        if AppState.isAutomated { defaults.removePersistentDomain(forName: "LLMUnderstander.automated") }
        read = Set(defaults.stringArray(forKey: Keys.read) ?? [])
        guesses = (defaults.dictionary(forKey: Keys.guesses) as? [String: Int]) ?? [:]
        let last = defaults.string(forKey: Keys.last).flatMap { k in Chapter.allCases.first { $0.key == k } }
        // First launch opens the Tour; after that, wherever the reader left off.
        selected = last ?? .tour
    }

    var isFirstLaunch: Bool { read.isEmpty && guesses.isEmpty }
    var readCount: Int { Chapter.chapters.filter { read.contains($0.key) }.count }

    func markRead(_ c: Chapter) {
        guard !read.contains(c.key) else { return }
        read.insert(c.key)
        defaults.set(Array(read).sorted(), forKey: Keys.read)
    }

    func recordGuess(_ option: Int, for c: Chapter) {
        guesses[c.key] = option
        defaults.set(guesses, forKey: Keys.guesses)
    }

    func clearGuess(for c: Chapter) {
        guesses[c.key] = nil
        defaults.set(guesses, forKey: Keys.guesses)
    }

    /// Settings → Reset reading progress: forgets chapters read, guesses and the last chapter.
    func resetProgress() {
        read = []; guesses = [:]
        for k in [Keys.read, Keys.guesses, Keys.last] { defaults.removeObject(forKey: k) }
    }

    static var isAutomated: Bool {
        let args = CommandLine.arguments
        return args.contains("--snapshots") || args.contains("--ui-testing")
            || ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }
}
