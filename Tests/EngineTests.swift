import XCTest
@testable import LLMUnderstander

final class EngineTests: XCTestCase {

    /// Compare autograd gradients with finite differences on a few entries of each parameter.
    private func gradCheck(_ params: [Tensor], _ f: () -> Tensor, samples: Int = 4, tol: Double = 1e-4,
                           file: StaticString = #filePath, line: UInt = #line) {
        for p in params { p.zeroGrad() }
        f().backward()
        var rng = SplitMix64(seed: 9)
        for p in params {
            let analytic = p.grad
            for _ in 0..<samples {
                let i = rng.int(p.count)
                let old = p.data[i], h = 1e-5
                p.data[i] = old + h; let up = f().item
                p.data[i] = old - h; let down = f().item
                p.data[i] = old
                let numeric = (up - down) / (2 * h)
                let denom = max(1e-8, abs(numeric) + abs(analytic[i]))
                XCTAssertLessThan(abs(numeric - analytic[i]) / denom, tol,
                                  "\(p.label)[\(i)] numeric \(numeric) vs analytic \(analytic[i])", file: file, line: line)
            }
        }
    }

    func testOpsGradients() {
        var rng = SplitMix64(seed: 1)
        let a = Tensor.randn(4, 3, std: 1, rng: &rng, label: "a")
        let b = Tensor.randn(3, 4, std: 1, rng: &rng, label: "b")
        let g = Tensor.randn(1, 4, std: 1, rng: &rng, label: "gamma")
        let bb = Tensor.randn(1, 4, std: 1, rng: &rng, label: "beta")
        let tgt = [0, 2, 1, 3]
        gradCheck([a, b, g, bb]) {
            let m = Ops.matmul(a, b)                         // 4×4
            let s = Ops.causalSoftmax(Ops.scale(m, 0.7))
            let ln = Ops.layerNorm(Ops.gelu(Ops.matmul(s, Ops.transpose(m))), g, bb)
            let cat = Ops.concatCols([Ops.sliceCols(ln, 0, 2), Ops.sliceCols(ln, 2, 2)])
            return Ops.crossEntropy(Ops.addRow(cat, bb), tgt, weights: [1, 0, 1, 1])
        }
    }

    func testModelGradients() {
        let model = TinyGPT(config: GPTConfig(vocab: 7, context: 6, dModel: 8, nHeads: 2, nLayers: 2), seed: 3)
        let x = [1, 4, 2, 0, 6, 3], y = [4, 2, 0, 6, 3, 5]
        gradCheck(model.parameters) { model.loss(batch: [(x, y)]) }
    }

    func testAttentionRowsSumToOneAndAreCausal() {
        let tok = CharTokenizer.shared
        let model = TinyGPT(config: GPTConfig(vocab: tok.size))
        let trace = model.trace(tok.encode("the cat sees"))
        for head in trace.attention[0] {
            for (i, row) in head.enumerated() {
                XCTAssertEqual(row.reduce(0, +), 1, accuracy: 1e-9)
                for j in (i + 1)..<row.count { XCTAssertEqual(row[j], 0) }
            }
        }
    }

    func testTrainingReducesLoss() {
        var s = TrainSettings(); s.totalSteps = 300; s.warmup = 20
        let w = TrainingWorker(config: GPTConfig(vocab: CharTokenizer.shared.size), settings: s, tokenizer: .shared)
        let before = w.validationLoss()
        let start = Date()
        for _ in 0..<300 { _ = w.trainStep() }
        let after = w.validationLoss()
        print("val loss \(before) → \(after) in \(Date().timeIntervalSince(start))s; sample: \(w.sample())")
        XCTAssertLessThan(after, before - 1.0)
    }

    func testBPELearnsCommonChunks() {
        let bpe = BPETokenizer.train(on: Corpus.training, numMerges: 40)
        XCTAssertEqual(bpe.merges.count, 40)
        let toks = bpe.encode("the cat sees the fish.")
        XCTAssertLessThan(toks.count, "the cat sees the fish.".count)
        XCTAssertEqual(toks.joined().replacingOccurrences(of: "␣", with: " "), "the cat sees the fish.")
    }

    func testSFTMovesTowardDemonstrations() {
        var l = PostTraining.baseLogits
        let before = PostTraining.sftLoss(l)
        for _ in 0..<100 { l = PostTraining.sftStep(l, lr: 0.5) }
        XCTAssertLessThan(PostTraining.sftLoss(l), before)
        let p = Num.softmax(l)
        XCTAssertGreaterThan(p[1] + p[2], 0.8)
    }

    func testRewardModelLearnsAndRLHFCanRewardHack() {
        let data = PostTraining.collectComparisons()
        let rm = PostTraining.trainedRewardModel()
        let r = rm.rewards
        print("RM w \(rm.w) acc \(rm.accuracy(data)) rewards \(r.map { String(format: "%.2f", $0) })")
        XCTAssertGreaterThan(rm.accuracy(data), 0.6)
        // The reward model over-rates the rambling essay it never saw labelled…
        XCTAssertEqual(r.indices.max { r[$0] < r[$1] }, 5)
        func run(beta: Double) -> [Double] {
            var l = PostTraining.sftLogits
            for _ in 0..<600 { l = PostTraining.rlhfStep(l, rewards: r, beta: beta) }
            return Num.softmax(l)
        }
        let low = run(beta: 0.05), mid = run(beta: 0.5)
        // …so with a weak KL leash the policy collapses onto it, and true quality drops.
        XCTAssertGreaterThan(low[5], 0.8)
        XCTAssertLessThan(PostTraining.expected(low, PostTraining.quality),
                          PostTraining.expected(mid, PostTraining.quality))
    }

    func testDPOReducesLoss() {
        let data = PostTraining.collectComparisons()
        var l = PostTraining.sftLogits
        let before = PostTraining.dpoLoss(l, data: data, beta: 0.5)
        for _ in 0..<200 { l = PostTraining.dpoStep(l, data: data, beta: 0.5, lr: 1) }
        XCTAssertLessThan(PostTraining.dpoLoss(l, data: data, beta: 0.5), before)
    }

    func testScalingLawOptimumIsAboutTwentyTokensPerParam() {
        let o = ScalingLaw.optimum(compute: 1e23)
        let ratio = o.tokens / o.params
        XCTAssertGreaterThan(ratio, 5); XCTAssertLessThan(ratio, 60)
    }

    func testAdamReachesRavineMinimum() {
        let path = Landscape.ravine.path(.adam, lr: 0.5, steps: 300)
        let end = path.last!
        XCTAssertLessThan(Landscape.ravine.value(end.0, end.1), 0.05)
    }

    // MARK: Agents

    func testAgentRLHacksNaiveRewardButNotRobustOne() {
        func train(robust: Bool) -> [Double] {
            var st = AgentRL.State()
            var rng = SplitMix64(seed: 21)
            for _ in 0..<150 { AgentRL.step(&st, robust: robust, rng: &rng) }
            return Num.softmax(st.logits)
        }
        let naive = train(robust: false), robust = train(robust: true)
        XCTAssertGreaterThan(naive[2], 0.6, "naive reward should be hacked by deleting the test")
        XCTAssertGreaterThan(robust[0], 0.6, "robust reward should favour a genuine fix")
        XCTAssertGreaterThan(AgentRL.genuineSuccess(robust), AgentRL.genuineSuccess(naive))
    }

    func testAgentScriptEndsWithFinalAnswerAndContextGrows() {
        let ev = AgentScript.events
        XCTAssertEqual(ev.last?.kind, .final)
        XCTAssertGreaterThan(AgentScript.contextTokens(upTo: ev.count - 1), AgentScript.contextTokens(upTo: 2))
        XCTAssertFalse(ev.filter { $0.kind == .toolResult }.contains { $0.trainedOn })
    }

    func testCompactionKeepsContextUnderWindow() {
        let off = ContextBudget.segments(turns: 80, compaction: false)
        let on = ContextBudget.segments(turns: 80, compaction: true)
        XCTAssertTrue(off.overflow)
        XCTAssertFalse(on.overflow)
        XCTAssertGreaterThan(on.compactions, 0)
    }

    @MainActor
    func testTrainedModelSurvivesRelaunch() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("llmu-test-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let a = Trainer(saveURL: url)
        a.trainNow(steps: 20)
        a.save()
        let b = Trainer(saveURL: url)
        XCTAssertEqual(b.step, 20)
        XCTAssertEqual(b.trainLoss.count, a.trainLoss.count)
        XCTAssertEqual(b.snapshot.wte.data, a.snapshot.wte.data)
        b.reset()
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path), "Reset should delete the saved model")
    }

    /// Pausing, quitting and relaunching should continue the same run, not a slightly different one.
    @MainActor
    func testResumedRunMatchesUnbrokenRun() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("llmu-test-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let unbroken = Trainer(saveURL: nil)
        unbroken.trainNow(steps: 40)
        let first = Trainer(saveURL: url)
        first.trainNow(steps: 20)
        first.save()
        let resumed = Trainer(saveURL: url)
        resumed.trainNow(steps: 20)
        XCTAssertEqual(resumed.step, 40)
        XCTAssertEqual(resumed.trainLoss.last!.y, unbroken.trainLoss.last!.y, accuracy: 1e-9)
        XCTAssertEqual(resumed.snapshot.wte.data, unbroken.snapshot.wte.data)
    }

    /// "Trained" is defined by loss (validation ≤ 60% of random guessing), and a full run gets there.
    @MainActor
    func testFullRunCountsAsTrained() {
        let t = Trainer(saveURL: nil)
        XCTAssertFalse(t.isTrained)
        t.trainNow(steps: t.settings.totalSteps)
        print("final validation loss \(t.latestValLoss!) vs trained bar \(t.trainedLoss); crossed at step \(t.valLoss.first { $0.y <= t.trainedLoss }?.x ?? -1)")
        XCTAssertTrue(t.isTrained)
    }

    /// SFT on the real tiny model: a pretrained model doesn't answer questions; a short fine-tune
    /// teaches the format, and it carries over to questions it never saw.
    @MainActor
    func testChatTuningTeachesAnswerFormat() {
        let t = Trainer(saveURL: nil)
        t.trainNow(steps: t.settings.totalSteps)
        let tuner = ChatTuner(base: t.snapshot)
        func count(_ check: (String, String) -> Bool) -> Int {
            ChatTuning.heldOut.filter { check(ChatTuning.answer(tuner.model, question: $0, tokenizer: .shared), $0) }.count
        }
        XCTAssertEqual(count { ChatTuning.startsWithSubject($0, question: $1) }, 0, "A pretrained model shouldn't answer questions yet")
        let before = tuner.demonstrationLoss()
        for _ in 0..<ChatTuner.suggestedSteps { _ = tuner.trainStep() }
        let after = tuner.demonstrationLoss()
        print("SFT loss \(before) → \(after); subject \(count { ChatTuning.startsWithSubject($0, question: $1) })/4, verb \(count { ChatTuning.usesVerb($0, question: $1) })/4")
        XCTAssertLessThan(after, before * 0.5)
        XCTAssertGreaterThanOrEqual(count { ChatTuning.startsWithSubject($0, question: $1) }, 3)
    }

    /// GRPO on the reasoning toy: accuracy rises by 25+ points and the model learns to think longer,
    /// and groups where every answer scores the same change nothing.
    func testGRPOLearnsToThinkLonger() {
        var toy = ReasoningToy(), rng = SplitMix64(seed: 42)
        let acc0 = toy.expectedAccuracy, len0 = toy.expectedLength
        var silent = 0
        for _ in 0..<150 { if toy.step(&rng).silent { silent += 1 } }
        print("GRPO accuracy \(acc0) → \(toy.expectedAccuracy), length \(len0) → \(toy.expectedLength), silent groups \(silent)/150")
        XCTAssertGreaterThan(toy.expectedAccuracy - acc0, 0.25)
        XCTAssertGreaterThan(toy.expectedLength, len0 + 1)

        var frozen = ReasoningToy(); frozen.groupSize = 8
        let before = frozen.logits
        // A group with identical rewards has zero advantage everywhere, so the logits can't move.
        frozen.lengthCost = 0
        var r2 = SplitMix64(seed: 1)
        for _ in 0..<200 {
            let saved = frozen.logits
            let g = frozen.step(&r2)
            if g.silent { XCTAssertEqual(frozen.logits, saved) }
        }
        XCTAssertNotEqual(frozen.logits, before)
    }

    /// The Evals chapter's point: on the same answers, per-character accuracy rises steadily
    /// while exact match lags and then catches up.
    @MainActor
    func testBenchmarkScoresDuringTraining() {
        XCTAssertEqual(Benchmark.items.count, 40)
        XCTAssertEqual(Set(Benchmark.items.map(\.prompt)).count, 40)
        let t = Trainer(saveURL: nil)
        let start = Date()
        _ = Benchmark.score(t.snapshot)
        print("benchmark scoring took \(Date().timeIntervalSince(start))s")
        t.trainNow(steps: t.settings.totalSteps)
        let rows = zip(t.benchmark, t.benchmarkPerChar).map { "\(Int($0.x)): exact \(String(format: "%.2f", $0.y)) perChar \(String(format: "%.2f", $1.y))" }
        print("benchmark curve:\n" + rows.joined(separator: "\n"))
        XCTAssertGreaterThanOrEqual(t.benchmark.count, t.settings.totalSteps / 100)   // step 0 arrives asynchronously
        XCTAssertGreaterThan(t.benchmark.last!.y, 0.5)
        for (e, c) in zip(t.benchmark, t.benchmarkPerChar) { XCTAssertGreaterThanOrEqual(c.y + 1e-9, e.y * 0.5) }
    }

    func testInferenceMaths() {
        let tiny = InferenceMaths.Shape.tiny(GPTConfig(vocab: 27))
        // For the first token there's nothing to reuse, so the two counts agree.
        XCTAssertEqual(InferenceMaths.withCache(tiny, position: 1), InferenceMaths.withoutCache(tiny, position: 1))
        XCTAssertGreaterThan(InferenceMaths.withoutCache(tiny, position: 32), 10 * InferenceMaths.withCache(tiny, position: 32))
        XCTAssertEqual(InferenceMaths.tokensPerPass(acceptance: 0.7, draft: 4), (1 - pow(0.7, 5)) / 0.3, accuracy: 1e-12)
        XCTAssertEqual(InferenceMaths.tokensPerPass(acceptance: 0.999, draft: 4), 5)
        XCTAssertEqual(InferenceMaths.nucleus([0.5, 0.3, 0.2], p: 0.7), [0, 1])
        XCTAssertEqual(InferenceMaths.nucleus([0.5, 0.3, 0.2], p: 0.1), [0])
        XCTAssertEqual(Benchmark.standardError(p: 0.6, n: 40), 0.0775, accuracy: 1e-4)
    }

    /// The website's engine is a port of this one with the same random number generator, so the
    /// same seeds must give the same corpus and the same first ten losses (shared fixture).
    func testMatchesSharedFixture() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/first-losses.json")
        struct Fixture: Decodable { let vocab: Int; let corpusStart: String; let losses: [Double] }
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        XCTAssertEqual(CharTokenizer.shared.size, f.vocab)
        XCTAssertEqual(String(Corpus.training.prefix(60)), f.corpusStart)
        let w = TrainingWorker(config: GPTConfig(vocab: CharTokenizer.shared.size), settings: TrainSettings(), tokenizer: .shared)
        for expected in f.losses { XCTAssertEqual(w.trainStep().loss, expected, accuracy: 1e-4) }
    }

    func testBanditExplorersBeatGreedyOnAverage() {
        func avgRegret(_ s: Bandit.Strategy) -> Double {
            (1...30).map { Bandit.run(s, steps: 600, seed: UInt64($0)).regret }.reduce(0, +) / 30
        }
        let greedy = avgRegret(.greedy), ucb = avgRegret(.ucb), ts = avgRegret(.thompson)
        print("bandit regret: greedy \(greedy) ε \(avgRegret(.epsilon)) ucb \(ucb) thompson \(ts)")
        XCTAssertLessThan(ts, greedy)
        XCTAssertLessThan(ts, avgRegret(.epsilon))
    }

    func testVotingMaths() {
        XCTAssertEqual(Sampling.majority(p: 0.6, n: 1), 0.6, accuracy: 1e-12)
        XCTAssertEqual(Sampling.majority(p: 0.6, n: 3), 0.648, accuracy: 1e-12)   // 3·0.36·0.4 + 0.216
        XCTAssertLessThan(Sampling.majority(p: 0.4, n: 9), 0.4)                    // voting hurts below 50%
        XCTAssertEqual(Sampling.anyCorrect(p: 0.5, n: 3), 0.875, accuracy: 1e-12)
        var rng = SplitMix64(seed: 3)
        let draws = (0..<4000).map { _ in Bandit.betaSample(3, 7, rng: &rng) }
        XCTAssertEqual(draws.reduce(0, +) / 4000, 0.3, accuracy: 0.01)            // Beta(3,7) mean = 0.3
    }
}
