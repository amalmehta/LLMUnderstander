import XCTest

/// Clicks through the app the way a reader would: every chapter, and the main
/// interactive controls, checking that each one visibly does something.
final class AppUITests: XCTestCase {
    var app: XCUIApplication!

    override func setUp() {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["--ui-testing"]
        app.launch()
    }

    override func tearDown() {
        app.terminate()
    }

    private func open(_ number: Int, _ title: String) {
        app.menuBars.menuBarItems["Chapters"].click()
        app.menuItems["\(number). \(title)"].click()
        XCTAssertTrue(app.staticTexts[title].firstMatch.waitForExistence(timeout: 5), "chapter \(number) didn't open")
    }

    private func text(containing s: String) -> XCUIElement {
        app.staticTexts.containing(NSPredicate(format: "value CONTAINS %@", s)).firstMatch
    }

    private func waitFor(_ s: String, timeout: TimeInterval = 20, file: StaticString = #filePath, line: UInt = #line) {
        let el = app.staticTexts.matching(NSPredicate(format: "value CONTAINS %@", s)).firstMatch
        XCTAssertTrue(el.waitForExistence(timeout: timeout), "never saw text containing “\(s)”", file: file, line: line)
    }

    /// Any element whose accessibility label contains `s` (headings and combined captions aren't static texts).
    private func waitForLabel(_ s: String, timeout: TimeInterval = 20, file: StaticString = #filePath, line: UInt = #line) {
        // On macOS a static text's words are its value; a combined element's are its label.
        // Typed queries stay fast; walking every element (descendants .any) can time out on big pages.
        let candidates = [
            app.staticTexts.matching(NSPredicate(format: "value CONTAINS %@", s)).firstMatch,
            app.otherElements.matching(NSPredicate(format: "label CONTAINS %@", s)).firstMatch,
            app.groups.matching(NSPredicate(format: "label CONTAINS %@", s)).firstMatch,
        ]
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if candidates.contains(where: { $0.exists }) { return }
            sleep(1)
        }
        XCTFail("never saw a label containing “\(s)”", file: file, line: line)
    }

    private func button(_ label: String) -> XCUIElement { app.buttons[label].firstMatch }

    func testEveryChapterOpens() {
        let titles = [
            "The Big Picture", "Text into Tokens", "Embeddings", "Attention", "The Transformer Block",
            "Predicting the Next Token", "Backpropagation", "The Optimizer", "The Training Loop",
            "Pretraining at Scale", "Supervised Fine-Tuning", "Learning from Preferences", "Reasoning Models",
            "Evals", "The Full Pipeline", "Inference",
            "From Model to Agent", "The Harness", "Tool Use, Step by Step", "Training Models to Be Agents", "Workflows & Bandits",
        ]
        for (i, t) in titles.enumerated() { open(i + 1, t) }
    }

    func testOpensOnTheTourAndScrolls() {
        // A fresh install starts on the Tour.
        waitFor("One sentence, all the way through")
        let stage = app.otherElements.matching(NSPredicate(format: "label BEGINSWITH %@", "Tour stage:")).firstMatch
        XCTAssertTrue(stage.waitForExistence(timeout: 5))
        XCTAssertTrue(stage.label.contains("It starts as text"))
        for _ in 0..<6 { stage.scroll(byDeltaX: 0, deltaY: -400) }   // the first scroll view is the sidebar
        sleep(1)
        XCTAssertFalse(stage.label.contains("It starts as text"), "scrolling should move the stage on (now: \(stage.label))")
        app.menuBars.menuBarItems["Chapters"].click()
        app.menuItems["Next Chapter"].click()
        XCTAssertTrue(app.staticTexts["The Big Picture"].firstMatch.waitForExistence(timeout: 5))
    }

    func testGuessRevealsTheAnswer() {
        open(4, "Attention")
        button("No").click()
        waitFor("causal mask hides")
        // A link-styled button may be exposed as a link, so find it by name whatever its type.
        app.descendants(matching: .any).matching(NSPredicate(format: "label == %@ OR title == %@", "Guess again", "Guess again")).firstMatch.click()
        XCTAssertTrue(button("Yes").waitForExistence(timeout: 5))
    }

    func testNewChapters() {
        open(13, "Reasoning Models")
        button("Train with GRPO").click()
        sleep(3)
        button("Pause").click()
        waitFor("Latest group")
        open(14, "Evals")
        waitFor("Standard error")
        open(16, "Inference")
        waitFor("characters kept")
        button("Sample again").click()
        waitFor("Tokens per big pass")
    }

    func testSettingsWindow() {
        app.typeKey(",", modifierFlags: .command)
        XCTAssertTrue(app.staticTexts["Chapters read"].waitForExistence(timeout: 5))
    }

    func testTrainingLoopTrainsAndPauses() {
        open(9, "The Training Loop")
        button("Start training").click()
        sleep(4)
        button("Pause").click()
        XCTAssertTrue(button("Resume").waitForExistence(timeout: 5))
        waitFor("step 100", timeout: 10)            // a sample was generated at step 100
        // The other chapters now see a trained-ish model.
        open(3, "Embeddings")
        waitFor("Validation loss")
        // Reset brings it back to zero.
        open(9, "The Training Loop")
        button("Reset").click()
        XCTAssertTrue(button("Start training").waitForExistence(timeout: 5))
    }

    func testBackpropStepThroughAndGradientCheck() {
        open(7, "Backpropagation")
        for _ in 0..<10 { button("Next").click() }
        waitFor("Update:")
        button("Take a gradient step").click()
        button("Take a gradient step").click()
        waitFor("prediction ŷ")
        button("Run gradient check on 8 random weights").click()
        waitFor("Brute force")
    }

    func testOptimizerAndNextToken() {
        open(8, "The Optimizer")
        button("Replay").click()
        sleep(1)
        button("Show all").click()
        waitFor("loss after 120 steps")
        open(6, "Predicting the Next Token")
        button("Sample next").click()
        button("Most likely").click()
        button("Reset").click()
    }

    func testSFTAndPreferenceTuning() {
        open(11, "Supervised Fine-Tuning")
        button("Train 25 steps").click()
        button("Train 25 steps").click()
        open(12, "Learning from Preferences")
        app.radioButtons["2 · Train a reward model"].click()
        button("Train to convergence").click()
        waitFor("rates the")                                        // the "rambling essay" warning
        app.radioButtons["3 · RLHF"].click()
        button("weak leash 0.05").click()
        waitFor("Reward hacking")
        button("balanced 0.5").click()
        waitFor("A balanced leash")
        app.radioButtons["4 · DPO"].click()
        button("Run DPO (300 steps)").click()
        XCTAssertTrue(button("Run DPO (300 steps)").waitForExistence(timeout: 20))
    }

    func testAgentChapters() {
        open(17, "From Model to Agent")
        for _ in 0..<15 { button("Next").click() }
        waitFor("signal to stop the loop")
        open(18, "The Harness")
        button("Sub-agents").click()
        waitFor("clean context")
        app.radioButtons["Ask before every tool"].click()
        waitFor("interruptions in a 4-call task")
        open(19, "Tool Use, Step by Step")
        for _ in 0..<4 { button("Next").click() }
        waitFor("the model is called again")
        app.scrollViews.firstMatch.scroll(byDeltaX: 0, deltaY: -700)   // bring "When things go wrong" into view
        app.radioButtons["Parallel calls"].click()
        waitFor("saves a whole round trip")
        open(20, "Training Models to Be Agents")
        button("Run 150 updates").click()
        waitFor("Reward hacking", timeout: 30)
        app.descendants(matching: .any)["robustChecker"].firstMatch.click()
        button("Run 150 updates").click()
        waitFor("reinforces genuine", timeout: 30)
    }

    func testFeedbackSheetOpensAndCancels() {
        app.buttons["Feedback"].firstMatch.click()
        XCTAssertTrue(app.staticTexts["Send Feedback"].waitForExistence(timeout: 5))
        app.buttons["Cancel"].click()
        XCTAssertFalse(app.staticTexts["Send Feedback"].waitForExistence(timeout: 2))
    }

    func testWorkflowsAndBandits() {
        open(21, "Workflows & Bandits")
        app.radioButtons["Evaluator–optimizer"].click()
        waitFor("cap the rounds")
        button("Run 600 tries").click()
        waitFor("Tries: 600", timeout: 30)
        button("New random run").click()
        waitFor("Tries: 600", timeout: 30)
    }
}
