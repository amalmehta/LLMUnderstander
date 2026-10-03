import SwiftUI
import Charts

struct PreferencesChapter: View {
    @State private var tab = 0
    private let data = PostTraining.collectComparisons()

    // reward model
    @State private var rm = PostTraining.RewardModel()
    @State private var rmSteps = 0
    @State private var rmHistory: [Point] = []

    // RLHF
    @State private var logBeta = log10(0.5)
    @State private var policy = PostTraining.sftLogits
    @State private var rlStep = 0
    @State private var rlRunning = false
    @State private var rlHistory: [(step: Int, proxy: Double, truth: Double, kl: Double)] = []

    // DPO
    @State private var dpoBeta = 0.5
    @State private var dpoPolicy = PostTraining.sftLogits
    @State private var dpoStep = 0
    @State private var dpoRunning = false
    @State private var dpoHistory: [Point] = []

    private let timer = Timer.publish(every: 0.03, on: .main, in: .common).autoconnect()
    private let rlTotal = 600, dpoTotal = 300

    var body: some View {
        ChapterPage(chapter: .preferences) {
            Prose("After SFT the model answers like an assistant, but only as well as its demonstrations. And for many questions it's much easier for a person to say **which of two answers is better** than to write the perfect one. Preference training uses those judgements.")
            Prose("Below is the same toy model as in the SFT chapter (one prompt, 8 possible answers). The steps are the real algorithms.")

            Picker("", selection: $tab) {
                Text("1 · Collect comparisons").tag(0)
                Text("2 · Train a reward model").tag(1)
                Text("3 · RLHF").tag(2)
                Text("4 · DPO").tag(3)
            }
            .pickerStyle(.segmented).labelsHidden()

            switch tab {
            case 0: comparisons
            case 1: rewardModel
            case 2: rlhf
            default: dpo
            }

            KeyIdea("Preference training turns \"A is better than B\" into a training signal. RLHF learns a reward model and then optimises against it, held on a **KL leash** to the SFT model so it can't exploit the reward model's blind spots. DPO skips the reward model and learns from the pairs directly.")

            GoDeeper(title: "how it's done for real") {
                Formula("Reward model (Bradley–Terry):  loss = −log σ( r(chosen) − r(rejected) )\nRLHF objective:                 maximise  E_π[ r(x, y) ]  −  β · KL( π ‖ π_SFT )\nDPO:  loss = −log σ( β·[ log π(y_w)/π_SFT(y_w)  −  log π(y_l)/π_SFT(y_l) ] )")
                Prose("**Real RLHF** samples whole answers from the model, scores them with the reward model (itself a transformer), and updates with **PPO**, a policy-gradient method that limits how far each update can move. Here, with only 8 answers, the expectation can be computed exactly instead of sampled.")
                Prose("**RLAIF / Constitutional AI:** the comparisons can come from an AI model judging answers against a written set of principles, rather than from people.")
                Prose("**Verifiable rewards:** for math and code the reward can be computed directly — did the answer match, did the tests pass? Reinforcement learning on such rewards, with long chains of thought, is how recent *reasoning* models are trained. Same idea: generate, score, push up what scored well.")
            }
        }
        .onReceive(timer) { _ in tick() }
    }

    // MARK: - 1 Comparisons

    private var comparisons: some View {
        Card(title: "Human comparisons", subtitle: "\(data.count) judgements: for the same prompt, which of two answers is better?") {
            ForEach(data.prefix(4)) { c in
                HStack(alignment: .top, spacing: 12) {
                    answerBox(c.chosen, chosen: true)
                    Text("vs").foregroundStyle(.secondary).padding(.top, 20)
                    answerBox(c.rejected, chosen: false)
                }
            }
            let counts = PostTraining.answers.map { a in (a, data.filter { $0.chosen == a.id }.count, data.filter { $0.chosen == a.id || $0.rejected == a.id }.count) }
            Chart(counts, id: \.0.id) { a, won, shown in
                BarMark(x: .value("win rate", shown > 0 ? Double(won) / Double(shown) : 0), y: .value("answer", a.short))
                    .foregroundStyle(Color.pink.opacity(0.6))
                    .annotation(position: .trailing) { Text(shown == 0 ? "never shown to labellers" : "\(won)/\(shown) wins").font(.caption2) }
            }
            .chartXScale(domain: 0...1)
            .frame(height: 300)
            Prose("Two realistic flaws are built in. Labellers **slightly favour longer answers**, a known bias. And the **long rambling essay never appeared** in the labelled data. Watch what the next steps do with that.")
        }
    }

    private func answerBox(_ id: Int, chosen: Bool) -> some View {
        let a = PostTraining.answers[id]
        return VStack(alignment: .leading, spacing: 4) {
            Label(chosen ? "Preferred" : "Rejected", systemImage: chosen ? "hand.thumbsup.fill" : "hand.thumbsdown")
                .font(.caption.bold()).foregroundStyle(chosen ? .green : .red)
            Text(a.text).font(.callout).fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background((chosen ? Color.green : Color.red).opacity(0.07), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - 2 Reward model

    private var rewardModel: some View {
        Card(title: "Reward model", subtitle: "Learns a score r(answer) such that preferred answers score higher. Here: a weighted sum of 3 features.") {
            HStack {
                Button("Train 50 steps") { trainRM(50) }.buttonStyle(.borderedProminent).tint(.pink)
                Button("Train to convergence") { trainRM(400 - rmSteps) }.disabled(rmSteps >= 400)
                Button("Reset") { rm = PostTraining.RewardModel(); rmSteps = 0; rmHistory = [] }
                Spacer()
                StatTile(label: "Steps", value: "\(rmSteps)", color: .gray)
                StatTile(label: "Agrees with labellers", value: String(format: "%.0f%%", rm.accuracy(data) * 100), color: .green)
            }
            HStack(alignment: .top, spacing: 20) {
                VStack(alignment: .leading) {
                    Text("Learned weights").font(.subheadline.weight(.semibold))
                    Chart(Array(zip(["helpfulness", "politeness", "length"], rm.w)), id: \.0) { n, v in
                        BarMark(x: .value("feature", n), y: .value("weight", v)).foregroundStyle(n == "length" ? Color.orange : Color.blue)
                    }
                    .chartYScale(domain: -0.5...3.5)
                    .frame(height: 180)
                    Text("It learns to like length — because the labellers did.").font(.caption).foregroundStyle(.secondary)
                }
                .frame(width: 260)
                VStack(alignment: .leading) {
                    Text("Reward model's score vs. what's really good").font(.subheadline.weight(.semibold))
                    let rewards = rm.rewards
                    Chart {
                        ForEach(PostTraining.answers) { a in
                            BarMark(x: .value("score", rewards[a.id]), y: .value("answer", a.short))
                                .foregroundStyle(by: .value("kind", "reward model"))
                                .position(by: .value("kind", "reward model"))
                            BarMark(x: .value("score", PostTraining.quality[a.id]), y: .value("answer", a.short))
                                .foregroundStyle(by: .value("kind", "true quality (hidden)"))
                                .position(by: .value("kind", "true quality (hidden)"))
                        }
                    }
                    .chartForegroundStyleScale(["reward model": Color.pink, "true quality (hidden)": Color.green.opacity(0.6)])
                    .frame(height: 240)
                }
            }
            if rmSteps >= 200 {
                Text("⚠️ The reward model rates the **long rambling essay** highest of all — it extrapolated \"longer is better\" to an answer no labeller ever saw. It is a good but imperfect proxy for what people want.")
                    .font(.callout).padding(10).background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private func trainRM(_ n: Int) {
        for _ in 0..<n {
            rm.step(data, lr: 0.5); rmSteps += 1
            rmHistory.append(Point(x: Double(rmSteps), y: rm.loss(data)))
        }
    }

    private static let fullyTrainedRewards = PostTraining.trainedRewardModel().rewards
    /// RLHF uses the reward model from step 2 if you trained it; otherwise a fully trained one.
    private var trainedRewards: [Double] { rmSteps >= 100 ? rm.rewards : Self.fullyTrainedRewards }

    // MARK: - 3 RLHF

    private var rlhf: some View {
        let beta = pow(10, logBeta)
        let p = Num.softmax(policy), ref = Num.softmax(PostTraining.sftLogits)
        return Card(title: "RLHF: optimise the reward, but stay close to the SFT model", subtitle: "Objective: expected reward − β × KL(policy ‖ SFT model)") {
            HStack {
                Button(rlRunning ? "Running…" : "Run RL (\(rlTotal) steps)") { policy = PostTraining.sftLogits; rlStep = 0; rlHistory = []; rlRunning = true }
                    .buttonStyle(.borderedProminent).tint(.pink).disabled(rlRunning)
                Button("Reset") { rlRunning = false; policy = PostTraining.sftLogits; rlStep = 0; rlHistory = [] }
                Spacer()
                Text("KL leash β = \(String(format: "%.2f", beta))").monospacedDigit().frame(width: 150, alignment: .trailing)
                Slider(value: $logBeta, in: -1.6...0.7).frame(width: 220)
            }
            HStack(spacing: 8) {
                presetBeta("weak leash 0.05", 0.05); presetBeta("balanced 0.5", 0.5); presetBeta("tight leash 3", 3)
            }
            HStack(alignment: .top, spacing: 20) {
                Chart {
                    ForEach(PostTraining.answers) { a in
                        BarMark(x: .value("p", p[a.id]), y: .value("answer", a.short))
                            .foregroundStyle(a.id == 5 ? Color.orange : Color.pink)
                        PointMark(x: .value("p", ref[a.id]), y: .value("answer", a.short))
                            .symbol(.diamond).foregroundStyle(.gray)
                    }
                }
                .chartXScale(domain: 0...1)
                .frame(height: 250)
                VStack(alignment: .leading) {
                    Chart {
                        ForEach(rlHistory, id: \.step) { h in
                            LineMark(x: .value("step", h.step), y: .value("v", h.proxy), series: .value("m", "reward model score")).foregroundStyle(by: .value("m", "reward model score"))
                            LineMark(x: .value("step", h.step), y: .value("v", h.truth), series: .value("m", "true quality")).foregroundStyle(by: .value("m", "true quality"))
                        }
                    }
                    .chartForegroundStyleScale(["reward model score": Color.pink, "true quality": Color.green])
                    .chartXScale(domain: 0...rlTotal)
                    .chartXAxisLabel("RL steps")
                    .frame(height: 200)
                    HStack(spacing: 8) {
                        StatTile(label: "Reward score", value: String(format: "%.2f", PostTraining.expected(p, trainedRewards)), color: .pink)
                        StatTile(label: "True quality", value: String(format: "%.2f", PostTraining.expected(p, PostTraining.quality)), color: .green)
                        StatTile(label: "KL from SFT", value: String(format: "%.2f", PostTraining.kl(p, ref)), color: .gray)
                    }
                }
            }
            Text("Gray diamonds = the SFT model it started from.").font(.caption).foregroundStyle(.secondary)
            if rlStep >= rlTotal {
                let hacked = p[5] > 0.5
                Text(hacked
                     ? "🚨 **Reward hacking.** With a weak leash the policy piled onto the rambling essay. The reward model's score went *up*, but true quality went *down*. The policy found the reward model's blind spot."
                     : (beta > 2 ? "With a very tight leash the policy barely moves from the SFT model: safe, but it leaves much of the improvement on the table."
                                 : "A balanced leash: the policy moves toward genuinely good answers, and the KL penalty stops it from collapsing onto the reward model's favourite mistake."))
                    .font(.callout).padding(10)
                    .background((hacked ? Color.red : Color.green).opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private func presetBeta(_ label: String, _ b: Double) -> some View {
        Button(label) { logBeta = log10(b); policy = PostTraining.sftLogits; rlStep = 0; rlHistory = []; rlRunning = true }
            .buttonStyle(.bordered).disabled(rlRunning)
    }

    // MARK: - 4 DPO

    private var dpo: some View {
        let p = Num.softmax(dpoPolicy), ref = Num.softmax(PostTraining.sftLogits)
        return Card(title: "DPO: Direct Preference Optimisation", subtitle: "No reward model, no RL loop — a classification loss on the comparisons themselves") {
            HStack {
                Button(dpoRunning ? "Running…" : "Run DPO (\(dpoTotal) steps)") { dpoPolicy = PostTraining.sftLogits; dpoStep = 0; dpoHistory = []; dpoRunning = true }
                    .buttonStyle(.borderedProminent).tint(.pink).disabled(dpoRunning)
                Button("Reset") { dpoRunning = false; dpoPolicy = PostTraining.sftLogits; dpoStep = 0; dpoHistory = [] }
                Spacer()
                Text("β = \(String(format: "%.2f", dpoBeta))").monospacedDigit()
                Slider(value: $dpoBeta, in: 0.05...2).frame(width: 220)
            }
            HStack(alignment: .top, spacing: 20) {
                Chart {
                    ForEach(PostTraining.answers) { a in
                        BarMark(x: .value("p", p[a.id]), y: .value("answer", a.short)).foregroundStyle(a.id == 5 ? Color.orange : Color.purple)
                        PointMark(x: .value("p", ref[a.id]), y: .value("answer", a.short)).symbol(.diamond).foregroundStyle(.gray)
                    }
                }
                .chartXScale(domain: 0...1)
                .frame(height: 250)
                VStack(alignment: .leading) {
                    Chart(dpoHistory) { LineMark(x: .value("step", $0.x), y: .value("loss", $0.y)).foregroundStyle(.purple) }
                        .chartXScale(domain: 0...dpoTotal).chartXAxisLabel("DPO steps").chartYAxisLabel("DPO loss")
                        .frame(height: 180)
                    HStack(spacing: 8) {
                        StatTile(label: "True quality", value: String(format: "%.2f", PostTraining.expected(p, PostTraining.quality)), color: .green)
                        StatTile(label: "Essay probability", value: String(format: "%.1f%%", p[5] * 100), color: .orange)
                        StatTile(label: "KL from SFT", value: String(format: "%.2f", PostTraining.kl(p, ref)), color: .gray)
                    }
                }
            }
            Prose("DPO shows the same comparisons directly to the policy: raise the log-probability of each preferred answer and lower the rejected one, both measured relative to the SFT model. It's simpler and more stable than RLHF, and it **can't be tricked by the rambling essay**, because there's no reward model to exploit. The flip side: it only learns about answers that appear in the data, and it inherits the labellers' length bias directly.")
        }
    }

    // MARK: - animation

    private func tick() {
        if rlRunning {
            let beta = pow(10, logBeta)
            let r = trainedRewards
            for _ in 0..<12 where rlStep < rlTotal {
                policy = PostTraining.rlhfStep(policy, rewards: r, beta: beta)
                rlStep += 1
                if rlStep % 6 == 0 || rlStep == 1 {
                    let p = Num.softmax(policy)
                    rlHistory.append((rlStep, PostTraining.expected(p, r), PostTraining.expected(p, PostTraining.quality), PostTraining.kl(p, Num.softmax(PostTraining.sftLogits))))
                }
            }
            if rlStep >= rlTotal { rlRunning = false }
        }
        if dpoRunning {
            for _ in 0..<6 where dpoStep < dpoTotal {
                dpoPolicy = PostTraining.dpoStep(dpoPolicy, data: data, beta: dpoBeta, lr: 1)
                dpoStep += 1
                if dpoStep % 3 == 0 { dpoHistory.append(Point(x: Double(dpoStep), y: PostTraining.dpoLoss(dpoPolicy, data: data, beta: dpoBeta))) }
            }
            if dpoStep >= dpoTotal { dpoRunning = false }
        }
    }
}
