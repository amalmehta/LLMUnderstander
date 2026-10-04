import SwiftUI

enum ChapterSection: String, CaseIterable, Identifiable {
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

enum Chapter: Int, CaseIterable, Identifiable, Hashable {
    case bigPicture
    case tokens, embeddings, attention, block
    case nextToken, backprop, optimizer, trainingLoop
    case pretraining
    case sft, preferences
    case pipeline
    case agentLoop, harness, toolUse, agentTraining, workflows

    var id: Int { rawValue }
    var number: Int { rawValue + 1 }

    var title: String {
        switch self {
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
        case .pipeline: "The Full Pipeline"
        case .agentLoop: "From Model to Agent"
        case .harness: "The Harness"
        case .toolUse: "Tool Use, Step by Step"
        case .agentTraining: "Training Models to Be Agents"
        case .workflows: "Workflows & Bandits"
        }
    }

    var tagline: String {
        switch self {
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
        case .pipeline: "Every stage on one page: what changes, and what stays the same."
        case .agentLoop: "A model only writes text. Put it in a loop with tools and it can act."
        case .harness: "Everything around the model: prompt, tools, permissions, memory, context."
        case .toolUse: "Schema in, JSON out, result back in — the anatomy of one tool call."
        case .agentTraining: "Teach tool use with transcripts, then sharpen it with RL on real tasks."
        case .workflows: "Wiring model calls into patterns — and choosing between options when you can't be sure which is best."
        }
    }

    var symbol: String {
        switch self {
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
        case .pipeline: "flowchart"
        case .agentLoop: "arrow.triangle.turn.up.right.circle"
        case .harness: "shippingbox"
        case .toolUse: "wrench.and.screwdriver"
        case .agentTraining: "figure.strengthtraining.traditional"
        case .workflows: "point.3.connected.trianglepath.dotted"
        }
    }

    var section: ChapterSection {
        switch self {
        case .bigPicture: .start
        case .tokens, .embeddings, .attention, .block: .blocks
        case .nextToken, .backprop, .optimizer, .trainingLoop: .learning
        case .pretraining: .pretraining
        case .sft, .preferences: .posttraining
        case .pipeline: .wrap
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
    var selected: Chapter? = .bigPicture
    var showFeedback = false
    /// Screenshot runs and tests start from a fresh, unsaved model so they're repeatable.
    let trainer = Trainer(saveURL: AppState.isAutomated ? nil : Trainer.defaultSaveURL)

    static var isAutomated: Bool {
        let args = CommandLine.arguments
        return args.contains("--snapshots") || args.contains("--ui-testing")
            || ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }
}
