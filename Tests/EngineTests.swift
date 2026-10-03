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
}
