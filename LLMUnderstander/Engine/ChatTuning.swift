import Foundation

/// Supervised fine-tuning of the reader's own tiny model.
///
/// The pretraining text never contains a question, so a pretrained model given
/// "what does the fox eat" just carries on with more sentence-like text. The
/// demonstrations below pair a question with a one-sentence answer. Only the
/// answer's characters count toward the loss (the loss mask), exactly as in real
/// SFT. The tiny vocabulary has no room for special role tokens, so a newline
/// marks where the answer starts.
enum ChatTuning {
    struct Example: Identifiable, Hashable {
        let question: String
        let answer: String
        var id: String { question }
        var text: String { question + "\n" + answer + "\n" }
    }

    /// "what does <subject> <verb>" → "<subject> <verb>s <object>."
    private static let pairs: [(verb: String, third: String)] = [
        ("see", "sees"), ("like", "likes"), ("find", "finds"), ("eat", "eats"), ("want", "wants"),
        ("hold", "holds"), ("chase", "chases"), ("bring", "brings"), ("love", "loves"), ("make", "makes"),
    ]

    static func example(subject: String, pair: Int, object: String) -> Example {
        let p = pairs[pair]
        return Example(question: "what does \(subject) \(p.verb)", answer: "\(subject) \(p.third) \(object).")
    }

    static let demonstrationCount = 96

    /// Demonstrations, and held-out questions that never appear in them.
    static let demonstrations: [Example] = {
        var rng = SplitMix64(seed: 2024)
        var seen = Set<String>()
        var out: [Example] = []
        while out.count < demonstrationCount {
            let s = Corpus.subjects[rng.int(Corpus.subjects.count)]
            let v = rng.int(pairs.count)
            let o = Corpus.objects[rng.int(Corpus.objects.count)]
            let e = example(subject: s, pair: v, object: o)
            if seen.insert(e.question).inserted && !heldOutKeys.contains(e.question) { out.append(e) }
        }
        return out
    }()

    static let heldOut: [String] = [
        "what does the queen want", "what does a child chase", "what does the fox find", "what does my friend bring",
    ]
    private static let heldOutKeys = Set(heldOut)

    /// The input, targets and loss mask for one demonstration, cut to the model's context.
    static func trainingRow(_ e: Example, tokenizer: CharTokenizer, context: Int) -> (ids: [Int], targets: [Int], mask: [Double]) {
        let all = tokenizer.encode(e.text)
        let answerStart = tokenizer.encode(e.question + "\n").count
        // Start at the question so the answer can copy its subject; a long answer's tail falls
        // outside the 32-character window, which is fine: the start of the answer is what's new.
        let end = min(all.count - 1, context)
        let ids = Array(all[0..<end])
        let targets = Array(all[1...end])
        let mask = (0..<end).map { $0 + 1 >= answerStart ? 1.0 : 0.0 }
        return (ids, targets, mask)
    }

    /// Greedy answer: the most likely next character each time, until a newline.
    static func answer(_ model: TinyGPT, question: String, tokenizer: CharTokenizer, maxLength: Int = 40) -> String {
        var ids = tokenizer.encode(tokenizer.clean(question) + "\n")
        let newline = tokenizer.encode("\n")[0]
        var out: [Int] = []
        for _ in 0..<maxLength {
            let p = model.nextTokenProbs(Array(ids.suffix(model.config.context)), temperature: 1)
            let next = p.indices.max { p[$0] < p[$1] }!
            if next == newline { break }
            out.append(next); ids.append(next)
        }
        return tokenizer.decode(out)
    }

    /// The question's subject and verb ("the fox", "find"), or nil if it isn't a "what does …" question.
    static func parse(_ question: String) -> (subject: String, verb: String)? {
        guard question.hasPrefix("what does ") else { return nil }
        let words = question.dropFirst("what does ".count).split(separator: " ")
        guard words.count >= 2 else { return nil }
        return (words.dropLast().joined(separator: " "), String(words.last!))
    }

    /// The format the demonstrations teach: answer with a sentence about the question's subject.
    static func startsWithSubject(_ answer: String, question: String) -> Bool {
        guard let q = parse(question) else { return false }
        return answer.hasPrefix(q.subject + " ")
    }

    /// The harder part: also using the question's verb ("find" → "finds").
    static func usesVerb(_ answer: String, question: String) -> Bool {
        guard let q = parse(question), let p = pairs.first(where: { $0.verb == q.verb }) else { return false }
        return answer.hasPrefix("\(q.subject) \(p.third) ")
    }
}

/// Fine-tunes a copy of the model off the main thread. Starts from whatever the reader trained.
final class ChatTuner: @unchecked Sendable {
    let model: TinyGPT
    let tokenizer: CharTokenizer
    private let optimizer = Optimizer(kind: .adam)
    private var rng = SplitMix64(seed: 11)
    private(set) var step = 0
    var learningRate = 3e-4
    private let lock = NSLock()
    private var _running = false
    /// Set from the main thread, read by the background loop.
    var running: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _running }
        set { lock.lock(); _running = newValue; lock.unlock() }
    }
    static let suggestedSteps = 150

    init(base: TinyGPT, tokenizer: CharTokenizer = .shared) {
        model = base.copy()
        self.tokenizer = tokenizer
    }

    /// One SFT step on a batch of 8 demonstrations; returns the masked loss.
    func trainStep() -> Double {
        let params = model.parameters
        let rows = (0..<8).map { _ in
            ChatTuning.trainingRow(ChatTuning.demonstrations[rng.int(ChatTuning.demonstrations.count)],
                                   tokenizer: tokenizer, context: model.config.context)
        }
        let loss = Ops.mean(rows.map { Ops.crossEntropy(model.forward($0.ids), $0.targets, weights: $0.mask) })
        for p in params { p.zeroGrad() }
        loss.backward()
        clipGradients(params, maxNorm: 1)
        optimizer.step(params, lr: learningRate)
        step += 1
        return loss.item
    }

    /// Average masked loss over all demonstrations (no update).
    func demonstrationLoss() -> Double {
        let rows = ChatTuning.demonstrations.map { ChatTuning.trainingRow($0, tokenizer: tokenizer, context: model.config.context) }
        return rows.map { Ops.crossEntropy(model.forward($0.ids), $0.targets, weights: $0.mask).item }.reduce(0, +) / Double(rows.count)
    }
}

/// What the SFT chapter shows while fine-tuning runs. Work happens on a background queue.
@MainActor
@Observable
final class ChatTuningSession {
    struct Row: Identifiable, Hashable {
        let question: String
        var before: String
        var after: String?
        var id: String { question }
    }

    private(set) var rows: [Row] = []
    private(set) var loss: [Point] = []
    private(set) var step = 0
    private(set) var isRunning = false
    private(set) var baseVersion = -1

    @ObservationIgnored private var tuner: ChatTuner?
    @ObservationIgnored private let queue = DispatchQueue(label: "chat tuning", qos: .userInitiated)
    @ObservationIgnored private var generation = 0

    var subjectCount: Int { rows.filter { r in r.after.map { ChatTuning.startsWithSubject($0, question: r.question) } ?? false }.count }
    var verbCount: Int { rows.filter { r in r.after.map { ChatTuning.usesVerb($0, question: r.question) } ?? false }.count }
    var finished: Bool { step >= ChatTuner.suggestedSteps }

    /// Start over from the reader's current model (called when the page opens or the model changes).
    func prepare(base: TinyGPT, version: Int) {
        guard version != baseVersion else { return }
        stop()
        generation += 1
        baseVersion = version
        let t = ChatTuner(base: base)
        tuner = t
        step = 0
        loss = []
        rows = ChatTuning.heldOut.map { Row(question: $0, before: "…") }
        let gen = generation
        queue.async {
            let l = t.demonstrationLoss()
            let before = ChatTuning.heldOut.map { ChatTuning.answer(t.model, question: $0, tokenizer: .shared) }
            DispatchQueue.main.async { [weak self] in
                guard let self, self.generation == gen else { return }
                self.loss = [Point(x: 0, y: l)]
                for i in self.rows.indices { self.rows[i].before = before[i] }
            }
        }
    }

    func restart(base: TinyGPT, version: Int) {
        baseVersion = -1
        prepare(base: base, version: version)
    }

    func start() {
        guard let t = tuner, !isRunning, !finished else { return }
        isRunning = true
        t.running = true
        let gen = generation
        let questions = rows.map(\.question)
        queue.async {
            while t.running && t.step < ChatTuner.suggestedSteps {
                var losses: [Point] = []
                for _ in 0..<5 { losses.append(Point(x: Double(t.step + 1), y: t.trainStep())) }
                let answers = questions.map { ChatTuning.answer(t.model, question: $0, tokenizer: .shared) }
                let s = t.step
                DispatchQueue.main.async { [weak self] in
                    guard let self, self.generation == gen else { return }
                    self.step = s
                    self.loss.append(contentsOf: losses)
                    for i in self.rows.indices { self.rows[i].after = answers[i] }
                }
            }
            DispatchQueue.main.async { [weak self] in
                guard let self, self.generation == gen else { return }
                self.isRunning = false
            }
        }
    }

    func stop() {
        tuner?.running = false
        isRunning = false
    }
}
