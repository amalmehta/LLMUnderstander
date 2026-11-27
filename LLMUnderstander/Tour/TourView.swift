import SwiftUI

/// One stop on the Tour: a caption on the left, a scene on the pinned stage.
struct TourStep: Identifiable {
    let id: Int
    let title: String
    let caption: LocalizedStringKey
    let chapter: Chapter

    static let sentence = "the cat sees the moon."
    /// The prediction steps ask what follows this prefix (the answer is "o").
    static let prefix = "the cat sees the m"

    static let all: [TourStep] = [
        TourStep(id: 0, title: "It starts as text", caption: "A language model only ever does one thing: read some text and guess what comes next. Follow this one sentence through the whole machine.", chapter: .bigPicture),
        TourStep(id: 1, title: "Text becomes tokens", caption: "The model can't read letters. Each piece of text becomes a number from a fixed list. Our tiny model uses one token per character; real models use chunks of words.", chapter: .tokens),
        TourStep(id: 2, title: "Tokens become vectors", caption: "Each number looks up a row of a table: a list of numbers (a vector) the model can compute with. These are your model's real values.", chapter: .embeddings),
        TourStep(id: 3, title: "Tokens look back", caption: "**Attention:** each token compares itself with every earlier token and borrows information from the ones that match. Thicker lines are stronger matches, read from your model.", chapter: .attention),
        TourStep(id: 4, title: "Through the blocks", caption: "Attention, then a small network on each token, each adding to a running total (the residual stream). Stack the block a few times (twice here, about a hundred times in big models).", chapter: .block),
        TourStep(id: 5, title: "A guess for every next token", caption: "At the end, the model scores every possible next character. Softmax turns the scores into probabilities. After \"the cat sees the m\", what comes next?", chapter: .nextToken),
        TourStep(id: 6, title: "Score the guess", caption: "**The loss** is how surprised the model was by the real next token: −log of the probability it gave it. Sure and right: near 0. Sure and wrong: large.", chapter: .nextToken),
        TourStep(id: 7, title: "Blame flows backwards", caption: "**Backpropagation** uses the chain rule to work out, for every weight, which way to move it to lower the loss. One backward pass covers all of them.", chapter: .backprop),
        TourStep(id: 8, title: "Nudge every weight", caption: "**The optimizer** moves each weight a small step against its gradient. AdamW, the usual choice, scales each step by how noisy that weight's gradient has been.", chapter: .optimizer),
        TourStep(id: 9, title: "Repeat", caption: "Batch, predict, score, backpropagate, update. That's the training loop. Run it thousands of times and the loss falls. This is your model's own curve.", chapter: .trainingLoop),
        TourStep(id: 10, title: "Now do it at scale", caption: "**Pretraining** is the same loop on trillions of tokens across thousands of GPUs. The result is a base model: a superb autocomplete.", chapter: .pretraining),
        TourStep(id: 11, title: "Teach it to answer", caption: "**Supervised fine-tuning:** the same loop on example conversations, graded only on the answers. It learns the format of an assistant.", chapter: .sft),
        TourStep(id: 12, title: "Teach it what people prefer", caption: "**Preference tuning:** people pick the better of two answers; a reward model learns their taste; RL nudges the model toward higher reward, but not too far.", chapter: .preferences),
        TourStep(id: 13, title: "Reward right answers", caption: "**Reasoning models:** where a checker can mark answers right or wrong, RL rewards correct results, and the model learns to work things out step by step.", chapter: .reasoning),
        TourStep(id: 14, title: "Put it to work", caption: "**Agents:** put the model in a loop with tools. It writes a tool call, the harness runs it, the result goes back into the context, and the loop repeats until the task is done.", chapter: .agentLoop),
    ]
}

/// The Tour: scroll to carry one sentence through the whole pipeline.
/// The stage stays pinned while captions scroll past; each step pops its pieces in.
struct TourView: View {
    @Environment(AppState.self) private var app
    @State private var offset: CGFloat = 0
    @State private var viewport: CGSize = .zero

    private let steps = TourStep.all

    var body: some View {
        let stepHeight = max(viewport.height * 0.8, 360)
        let headerHeight = max(viewport.height * 0.45, 260)
        // Position along the tour: 0 at the first step, fractional in between.
        let position = max(0, min(Double(steps.count - 1) + 0.999, Double((offset - headerHeight + viewport.height * 0.5) / stepHeight)))
        let current = Int(position.rounded(.down))
        let local = position - Double(current)
        let stageWidth = max(viewport.width - 320 - 32 - 36 - 28, 420)

        ScrollView {
            HStack(alignment: .top, spacing: 32) {
                VStack(alignment: .leading, spacing: 0) {
                    header.frame(height: headerHeight, alignment: .bottomLeading)
                    ForEach(steps) { s in
                        caption(s, active: s.id == current)
                            .frame(height: stepHeight, alignment: .center)
                    }
                    ending.frame(height: viewport.height * 0.9, alignment: .top)
                }
                .frame(width: 320, alignment: .leading)

                // Pinned: moves down exactly as far as the content scrolls up.
                TourStage(step: current, local: local)
                    .frame(width: stageWidth, height: max(viewport.height - 80, 300))
                    .offset(y: offset + 40)
            }
            .padding(.leading, 36)
            .frame(maxWidth: .infinity, alignment: .leading)
            .onGeometryChange(for: CGFloat.self) { -$0.frame(in: .named("tour")).minY } action: { offset = $0 }
        }
        .coordinateSpace(name: "tour")
        .onGeometryChange(for: CGSize.self) { $0.size } action: { viewport = $0 }
        .overlay(alignment: .bottom) { progressBar(position: position) }
        .navigationTitle("Tour")
        .task {
            try? await Task.sleep(for: .seconds(20))
            if !Task.isCancelled { app.markRead(.tour) }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("TOUR").font(.caption.weight(.semibold)).tracking(1.2).foregroundStyle(.indigo)
            Text("One sentence, all the way through").font(.system(size: 30, weight: .bold, design: .rounded))
                .fixedSize(horizontal: false, vertical: true)
            Text(Chapter.tour.tagline).font(.title3).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Label("Scroll down", systemImage: "arrow.down").font(.callout).foregroundStyle(.secondary).padding(.top, 6)
        }
        .padding(.bottom, 30)
    }

    private func caption(_ s: TourStep, active: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("\(s.id + 1) of \(steps.count)").font(.caption.weight(.semibold)).foregroundStyle(s.chapter.section.color)
            Text(s.title).font(.title2.weight(.semibold))
            Text(s.caption).font(.system(size: 15)).lineSpacing(3).fixedSize(horizontal: false, vertical: true)
            Button { app.selected = s.chapter } label: {
                Label("Open \(s.chapter.menuTitle)", systemImage: "arrow.right")
            }
            .buttonStyle(.link).padding(.top, 2)
        }
        .opacity(active ? 1 : 0.35)
        .animation(.easeOut(duration: 0.2), value: active)
    }

    private var ending: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("That's the whole chain.").font(.title2.weight(.semibold))
            Text("Every piece has a chapter where you can take it apart and play with it, starting with the model you train yourself.")
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Start with Chapter 1") { app.selected = .bigPicture }.buttonStyle(.borderedProminent)
                Button("Train the model") { app.selected = .trainingLoop }
            }
        }
        .padding(.top, 40)
    }

    private func progressBar(position: Double) -> some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.primary.opacity(0.08))
                Capsule().fill(Color.indigo.opacity(0.6))
                    .frame(width: g.size.width * CGFloat(min(1, (position + 0.5) / Double(steps.count))))
            }
        }
        .frame(height: 4)
        .padding(.horizontal, 36).padding(.bottom, 12)
        .accessibilityElement()
        .accessibilityLabel("Tour progress")
        .accessibilityValue("step \(Int(position) + 1) of \(steps.count)")
    }
}
