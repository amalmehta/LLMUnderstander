import Foundation
import Observation

struct Point: Identifiable, Hashable, Codable {
    let x: Double
    let y: Double
    var id: Double { x }
}

struct GeneratedSample: Identifiable, Hashable, Codable {
    let step: Int
    let text: String
    var id: Int { step }
}

/// The five things every training step does, in order.
enum LoopPhase: Int, CaseIterable, Identifiable {
    case batch, forward, loss, backward, update
    var id: Int { rawValue }
    var title: String {
        switch self {
        case .batch: "Grab a batch of text"
        case .forward: "Forward pass: predict every next token"
        case .loss: "Loss: how surprised was the model?"
        case .backward: "Backward pass: gradients for every weight"
        case .update: "Optimizer step: nudge every weight"
        }
    }
    var code: String {
        switch self {
        case .batch: "x, y = next_batch(data)"
        case .forward: "logits = model(x)"
        case .loss: "loss = cross_entropy(logits, y)"
        case .backward: "loss.backward()"
        case .update: "optimizer.step(); optimizer.zero_grad()"
        }
    }
}

struct TrainSettings: Equatable, Codable {
    var peakLR = 3e-3
    var batchSize = 8
    var optimizer: OptimizerKind = .adam
    var totalSteps = 1500
    var warmup = 100
    var clip = 1.0
}

/// Owns the model being trained. All real work happens on a background queue;
/// the main-actor `Trainer` only receives progress reports.
final class TrainingWorker: @unchecked Sendable {
    let tokenizer: CharTokenizer
    let train: [Int]
    let val: [Int]
    var model: TinyGPT
    var optimizer: Optimizer
    var settings: TrainSettings
    var rng = SplitMix64(seed: 42)
    var step = 0
    private let lock = NSLock()
    private var _running = false
    private var _slow = false
    var slow: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _slow }
        set { lock.lock(); _slow = newValue; lock.unlock() }
    }
    var running: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _running }
        set { lock.lock(); _running = newValue; lock.unlock() }
    }

    init(config: GPTConfig, settings: TrainSettings, tokenizer: CharTokenizer) {
        self.tokenizer = tokenizer
        train = tokenizer.encode(Corpus.training)
        val = tokenizer.encode(Corpus.validation)
        model = TinyGPT(config: config)
        optimizer = Optimizer(kind: settings.optimizer)
        self.settings = settings
    }

    func batch(from data: [Int], size: Int, rng: inout SplitMix64) -> [([Int], [Int])] {
        let T = model.config.context
        return (0..<size).map { _ in
            let o = rng.int(data.count - T - 1)
            return (Array(data[o..<(o + T)]), Array(data[(o + 1)..<(o + T + 1)]))
        }
    }

    struct StepResult { let loss: Double; let gradNorm: Double; let lr: Double }

    /// One full iteration of the training loop. `onPhase` lets the UI follow along.
    func trainStep(onPhase: ((LoopPhase) -> Void)? = nil) -> StepResult {
        let params = model.parameters
        onPhase?(.batch)
        let b = batch(from: train, size: settings.batchSize, rng: &rng)
        onPhase?(.forward)
        let logits = b.map { model.forward($0.0) }
        onPhase?(.loss)
        let loss = Ops.mean(zip(logits, b).map { Ops.crossEntropy($0, $1.1) })
        onPhase?(.backward)
        for p in params { p.zeroGrad() }
        loss.backward()
        let norm = clipGradients(params, maxNorm: settings.clip)
        onPhase?(.update)
        let lr = LRSchedule(peak: settings.peakLR, warmup: settings.warmup, total: settings.totalSteps).lr(at: step)
        optimizer.step(params, lr: lr)
        step += 1
        return StepResult(loss: loss.item, gradNorm: norm, lr: lr)
    }

    func validationLoss() -> Double {
        var r = SplitMix64(seed: 1234)
        let b = batch(from: val, size: 16, rng: &r)
        return model.loss(batch: b).item
    }

    func sample(prompt: String = "the ", count: Int = 60) -> String {
        var r = SplitMix64(seed: UInt64(step) &+ 5)
        let ids = model.generate(prompt: tokenizer.encode(prompt), count: count, temperature: 0.8, rng: &r)
        return prompt + tokenizer.decode(ids)
    }
}

@MainActor
@Observable
final class Trainer {
    let tokenizer = CharTokenizer.shared
    let config: GPTConfig
    var settings = TrainSettings()

    private(set) var step = 0
    private(set) var trainLoss: [Point] = []
    private(set) var valLoss: [Point] = []
    private(set) var learningRates: [Point] = []
    private(set) var gradNorms: [Point] = []
    private(set) var samples: [GeneratedSample] = []
    private(set) var isRunning = false
    private(set) var phase: LoopPhase?
    var slowMotion = false {
        didSet { worker.slow = slowMotion }
    }
    /// A frozen copy of the model for the other chapters to look inside.
    private(set) var snapshot: TinyGPT
    private(set) var snapshotVersion = 0

    @ObservationIgnored private var worker: TrainingWorker
    @ObservationIgnored private let queue = DispatchQueue(label: "training", qos: .userInitiated)

    /// Where the trained model is kept between launches (nil = don't save, e.g. tests and screenshots).
    @ObservationIgnored private let saveURL: URL?

    init(saveURL: URL? = Trainer.defaultSaveURL) {
        config = GPTConfig(vocab: CharTokenizer.shared.size)
        worker = TrainingWorker(config: config, settings: TrainSettings(), tokenizer: CharTokenizer.shared)
        snapshot = worker.model.copy()
        self.saveURL = saveURL
        if !restore() { recordInitial() }
    }

    nonisolated static var defaultSaveURL: URL? {
        guard let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else { return nil }
        return dir.appendingPathComponent("LLM Understander", isDirectory: true).appendingPathComponent("Tiny Model.json")
    }

    /// Everything needed to bring a trained model and its charts back after a relaunch.
    struct Saved: Codable {
        var config: GPTConfig
        var settings: TrainSettings
        var step: Int
        var weights: [[Double]]
        var trainLoss: [Point], valLoss: [Point], learningRates: [Point], gradNorms: [Point]
        var samples: [GeneratedSample]
    }

    /// Save the model after training stops. Optimizer momentum isn't saved; resuming rebuilds it within a few steps.
    func save() {
        guard let url = saveURL, step > 0 else { return }
        let saved = Saved(config: config, settings: settings, step: step, weights: worker.model.parameters.map(\.data),
                          trainLoss: trainLoss, valLoss: valLoss, learningRates: learningRates, gradNorms: gradNorms, samples: samples)
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(saved).write(to: url, options: .atomic)
        } catch {
            NSLog("LLM Understander: could not save model: \(error)")
        }
    }

    /// Load a previously saved model if it matches this app's model shape.
    private func restore() -> Bool {
        guard let url = saveURL, let data = try? Data(contentsOf: url),
              let saved = try? JSONDecoder().decode(Saved.self, from: data),
              saved.config == config else { return false }
        let params = worker.model.parameters
        guard saved.weights.count == params.count, zip(saved.weights, params).allSatisfy({ $0.count == $1.count }) else { return false }
        for (w, p) in zip(saved.weights, params) { p.data = w }
        worker.step = saved.step
        settings = saved.settings
        step = saved.step
        trainLoss = saved.trainLoss; valLoss = saved.valLoss
        learningRates = saved.learningRates; gradNorms = saved.gradNorms
        samples = saved.samples
        snapshot = worker.model.copy()
        return true
    }

    var parameterCount: Int { snapshot.parameterCount }
    var tokensSeen: Int { step * settings.batchSize * config.context }
    var isTrained: Bool { step >= 300 }
    var finished: Bool { step >= settings.totalSteps }

    /// Loss of a model that knows nothing: guessing uniformly among all characters.
    var uniformLoss: Double { log(Double(config.vocab)) }

    private func recordInitial() {
        let w = worker
        let v = w.validationLoss()
        valLoss = [Point(x: 0, y: v)]
        samples = [GeneratedSample(step: 0, text: w.sample())]
    }

    func reset() {
        pause()
        queue.sync {}
        isRunning = false
        worker = TrainingWorker(config: config, settings: settings, tokenizer: tokenizer)
        if let url = saveURL { try? FileManager.default.removeItem(at: url) }
        step = 0
        trainLoss = []; valLoss = []; learningRates = []; gradNorms = []; samples = []
        phase = nil
        snapshot = worker.model.copy()
        snapshotVersion += 1
        recordInitial()
    }

    func start() {
        guard !isRunning, !finished else { return }
        isRunning = true
        let w = worker
        w.running = true
        w.settings = settings
        w.slow = slowMotion
        queue.async { [weak self] in
            var pending: [(Int, TrainingWorker.StepResult)] = []
            while w.running && w.step < w.settings.totalSteps {
                let result: TrainingWorker.StepResult
                let slow = w.slow
                if slow {
                    result = w.trainStep { p in
                        DispatchQueue.main.async { if self?.worker === w { self?.phase = p } }
                        Thread.sleep(forTimeInterval: 0.45)
                    }
                } else {
                    result = w.trainStep()
                }
                pending.append((w.step, result))
                let s = w.step
                let report = s % 10 == 0 || slow
                let deep = s % 100 == 0 || s == w.settings.totalSteps
                if report || deep {
                    let batch = pending
                    pending = []
                    let val = deep ? w.validationLoss() : nil
                    let sample = deep ? w.sample() : nil
                    let snap = deep ? w.model.copy() : nil
                    DispatchQueue.main.async {
                        guard let self, self.worker === w else { return }   // ignore a run that was reset
                        self.receive(step: s, results: batch, val: val, sample: sample, snapshot: snap)
                    }
                }
            }
            DispatchQueue.main.async {
                guard let self, self.worker === w else { return }
                self.isRunning = false
                self.phase = nil
                self.snapshot = w.model.copy()
                self.snapshotVersion += 1
                self.save()
            }
        }
    }

    func pause() {
        worker.running = false
    }

    /// Train synchronously (used for screenshots and tests).
    func trainNow(steps: Int) {
        let w = worker
        w.settings = settings
        var left = steps
        while left > 0 && w.step < w.settings.totalSteps {
            let chunk = min(left, 100 - w.step % 100)
            var results: [(Int, TrainingWorker.StepResult)] = []
            queue.sync {
                for _ in 0..<chunk { results.append((w.step + 1, w.trainStep())) }
            }
            left -= chunk
            receive(step: w.step, results: results, val: w.validationLoss(), sample: w.sample(), snapshot: w.model.copy())
        }
    }

    private func receive(step s: Int, results: [(Int, TrainingWorker.StepResult)], val: Double?, sample: String?, snapshot snap: TinyGPT?) {
        step = max(step, s)
        let last = trainLoss.last?.x ?? 0
        for (i, r) in results where Double(i) > last {
            trainLoss.append(Point(x: Double(i), y: r.loss))
            learningRates.append(Point(x: Double(i), y: r.lr))
            gradNorms.append(Point(x: Double(i), y: r.gradNorm))
        }
        if let val { valLoss.append(Point(x: Double(s), y: val)) }
        if let sample { samples.append(GeneratedSample(step: s, text: sample)) }
        if let snap { snapshot = snap; snapshotVersion += 1 }
    }

    /// Smoothed training loss for charts (raw per-batch loss is noisy).
    var smoothedTrainLoss: [Point] {
        var out: [Point] = []
        var ema: Double?
        for p in trainLoss {
            ema = ema.map { 0.95 * $0 + 0.05 * p.y } ?? p.y
            out.append(Point(x: p.x, y: ema!))
        }
        return out
    }
}
