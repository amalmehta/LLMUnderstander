import Foundation

/// A scripted, realistic agent run: a coding agent fixing a failing test.
/// Every event is what a real harness and model would exchange; the text is
/// fixed so the app works offline, but the shapes match real tool-use APIs.
enum AgentScript {
    enum Actor: String {
        case harness = "Harness"
        case user = "User"
        case model = "Model"
        case tool = "Tool"
    }

    enum Kind: String {
        case system = "system prompt"
        case tools = "tool definitions"
        case user = "user message"
        case text = "assistant text"
        case toolCall = "tool call"
        case permission = "permission check"
        case toolResult = "tool result"
        case final = "final answer"
    }

    struct Event: Identifiable {
        let id: Int
        let actor: Actor
        let kind: Kind
        let content: String
        /// What's going on at this step, in plain words.
        let note: String
        /// Does this event become part of the context the model sees next time?
        var inContext: Bool { kind != .permission }
        /// Would SFT on this transcript train the model to produce these tokens?
        var trainedOn: Bool { actor == .model }
        /// Rough token count (≈ 4 characters per token for English and code).
        var tokens: Int { max(1, content.count / 4) }
    }

    static let events: [Event] = {
        let raw: [(Actor, Kind, String, String)] = [
            (.harness, .system,
             "You are a coding agent working in the user's repository. Use the tools to inspect and change files. Run the tests before you finish. Be concise.",
             "Before anything else, the harness writes a system prompt. The model never sees the app, the files or the terminal, only this text and what follows."),
            (.harness, .tools,
             """
             read_file(path: string) → file contents
             edit_file(path: string, old: string, new: string) → ok | error
             run_tests() → test output
             """,
             "The harness describes each tool: a name, what it does and a JSON schema for its arguments. To the model these are just more tokens in the prompt."),
            (.user, .user,
             "The test for average() is failing. Can you fix it?",
             "The user's request is appended. Now the harness sends everything so far to the model and asks for the next message."),
            (.model, .text,
             "I'll start by running the tests to see the failure.",
             "The model predicts tokens exactly as in Chapter 6. Some of them are ordinary text…"),
            (.model, .toolCall,
             #"{"tool": "run_tests", "input": {}}"#,
             "…and some form a structured tool call. The model has stopped generating and is waiting. It cannot run anything itself; it has only *asked*."),
            (.harness, .permission,
             "run_tests: read-only → allowed automatically",
             "The harness, not the model, decides whether the call may run. Running tests is on the allow-list, so no human is asked."),
            (.tool, .toolResult,
             "FAILED test_stats.py::test_average\n    assert average([1, 2, 3, 4]) == 2.5\n    AssertionError: got 2",
             "The harness runs the tool and pastes the output into the context as a tool result. Then it calls the model again with the longer context."),
            (.model, .toolCall,
             #"{"tool": "read_file", "input": {"path": "stats.py"}}"#,
             "A new forward pass over the whole context. Seeing the failure, the most likely next tokens are a call to read the code."),
            (.harness, .permission,
             "read_file: read-only → allowed automatically",
             "Reads are safe, so they're allowed."),
            (.tool, .toolResult,
             "def average(xs):\n    return sum(xs) // len(xs)",
             "The file contents go into the context. Tool results are often the biggest thing in an agent's context."),
            (.model, .text,
             "The bug is integer division (//). It should be true division (/).",
             "The model reasons in text. This is still just next-token prediction, conditioned on everything above."),
            (.model, .toolCall,
             #"{"tool": "edit_file", "input": {"path": "stats.py", "old": "sum(xs) // len(xs)", "new": "sum(xs) / len(xs)"}}"#,
             "An edit changes the user's files, so this call is riskier."),
            (.harness, .permission,
             "edit_file: modifies files → ask the user… approved",
             "Edits aren't on the allow-list, so the harness pauses and asks the human. Permissions live in the harness, outside the model."),
            (.tool, .toolResult,
             "ok: 1 replacement in stats.py",
             "The edit happened in the real world (well, the real file system), and the model is told it succeeded."),
            (.model, .toolCall,
             #"{"tool": "run_tests", "input": {}}"#,
             "Good agents check their work. The model asks to run the tests again."),
            (.harness, .permission,
             "run_tests: read-only → allowed automatically",
             "Allowed."),
            (.tool, .toolResult,
             "1 passed in 0.02s",
             "Verified."),
            (.model, .final,
             "Fixed it: average() used integer division (//), which truncated 2.5 to 2. I changed it to true division (/) and the test now passes.",
             "This time the model writes an answer with no tool call. That is the harness's signal to stop the loop and show the reply to the user."),
        ]
        return raw.enumerated().map { Event(id: $0.offset, actor: $0.element.0, kind: $0.element.1, content: $0.element.2, note: $0.element.3) }
    }()

    /// Indices where the model is called (each model turn starts a new forward pass over the context).
    static var modelCalls: Int {
        var n = 0
        var prevModel = false
        for e in events {
            let isModel = e.actor == .model
            if isModel && !prevModel { n += 1 }
            prevModel = isModel
        }
        return n
    }

    /// Context size (tokens) as seen by the model after `upTo` events.
    static func contextTokens(upTo i: Int) -> Int {
        events.prefix(i + 1).filter(\.inContext).reduce(0) { $0 + $1.tokens }
    }
}

/// A long simulated session, to show how a harness manages a finite context window.
enum ContextBudget {
    struct Segment: Identifiable {
        let id: String
        let tokens: Double
    }

    static let window = 200_000.0
    static let system = 3_000.0, tools = 12_000.0, memory = 4_000.0

    /// Tokens added by turn `t` (tool results vary a lot: a big file read, a long log…).
    static func turnTokens(_ t: Int) -> Double {
        var r = SplitMix64(seed: UInt64(t) &* 2654435761 &+ 7)
        let base = 1_500 + r.uniform() * 2_500
        let bigResult = r.uniform() < 0.25 ? 6_000 + r.uniform() * 14_000 : 0
        return base + bigResult
    }

    /// Context after `turns` turns. When compaction is on and usage passes `threshold`,
    /// older conversation is replaced by a summary of `summaryTokens`.
    static func segments(turns: Int, compaction: Bool, threshold: Double = 0.8) -> (segments: [Segment], compactions: Int, overflow: Bool) {
        var conversation = 0.0, summary = 0.0, compactions = 0
        let fixed = system + tools + memory
        for t in 0..<turns {
            conversation += turnTokens(t)
            if compaction && fixed + summary + conversation > threshold * window {
                compactions += 1
                summary = min(summary + 4_000, 15_000)
                conversation *= 0.1   // keep only the most recent work verbatim
            }
        }
        let overflow = fixed + summary + conversation > window
        return ([
            Segment(id: "System prompt", tokens: system),
            Segment(id: "Tool definitions", tokens: tools),
            Segment(id: "Memory files", tokens: memory),
            Segment(id: "Summary of earlier work", tokens: summary),
            Segment(id: "Conversation + tool results", tokens: conversation),
        ], compactions, overflow)
    }
}

/// RL on an agent task with a verifiable reward. The "policy" picks one of four
/// whole strategies for "make the failing test pass"; REINFORCE with a baseline
/// learns from sampled episodes. A naive reward ("tests pass") can be gamed.
enum AgentRL {
    struct Strategy: Identifiable {
        let id: Int
        let name: String
        let detail: String
        /// Probability the tests pass at the end.
        let passRate: Double
        /// Does it tamper with the tests?
        let tampers: Bool
        /// Is it what the user actually wanted?
        let genuine: Bool
    }

    static let strategies: [Strategy] = [
        Strategy(id: 0, name: "Investigate, then fix", detail: "run tests → read code → fix → re-run", passRate: 0.7, tampers: false, genuine: true),
        Strategy(id: 1, name: "Guess an edit", detail: "change code without reading it", passRate: 0.3, tampers: false, genuine: true),
        Strategy(id: 2, name: "Delete the failing test", detail: "tests now 'pass'", passRate: 1.0, tampers: true, genuine: false),
        Strategy(id: 3, name: "Give up", detail: "explain and stop", passRate: 0.0, tampers: false, genuine: true),
    ]

    /// Starting policy (after SFT): mostly reasonable, rarely hacky.
    static let initialLogits: [Double] = [0.5, 0.8, -0.5, 0.3]

    /// One episode's reward. The robust checker also verifies the test files are untouched.
    static func reward(_ s: Strategy, passed: Bool, robust: Bool) -> Double {
        if robust && s.tampers { return 0 }
        return passed ? 1 : 0
    }

    struct State {
        var logits = AgentRL.initialLogits
        var baseline = 0.0
        var episodes = 0
        var lastRewards: [Double] = []
    }

    /// Sample a batch of episodes and apply one REINFORCE update:
    /// θ_j += lr · mean[(R − b) · (1[j = a] − π_j)].
    static func step(_ s: inout State, batch: Int = 16, lr: Double = 0.4, robust: Bool, rng: inout SplitMix64) {
        let p = Num.softmax(s.logits)
        var g = [Double](repeating: 0, count: p.count)
        var rewards: [Double] = []
        for _ in 0..<batch {
            let a = Num.sample(p, rng: &rng)
            let strat = strategies[a]
            let passed = rng.uniform() < strat.passRate
            let r = reward(strat, passed: passed, robust: robust)
            rewards.append(r)
            let adv = r - s.baseline
            for j in 0..<g.count { g[j] += adv * ((j == a ? 1 : 0) - p[j]) / Double(batch) }
        }
        for j in 0..<g.count { s.logits[j] += lr * g[j] }
        let mean = rewards.reduce(0, +) / Double(batch)
        s.baseline = 0.9 * s.baseline + 0.1 * mean
        s.episodes += batch
        s.lastRewards = rewards
    }

    /// Fraction of episodes that did what the user actually wanted *and* passed.
    static func genuineSuccess(_ p: [Double]) -> Double {
        zip(p, strategies).reduce(0) { $0 + $1.0 * ($1.1.genuine ? $1.1.passRate : 0) }
    }

    static func expectedReward(_ p: [Double], robust: Bool) -> Double {
        zip(p, strategies).reduce(0) { acc, x in acc + x.0 * ((robust && x.1.tampers) ? 0 : x.1.passRate) }
    }
}
