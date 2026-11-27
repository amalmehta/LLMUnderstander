import SwiftUI
import AppKit

/// ⌘, — everything the app keeps, in one place, with a way to see it and to delete it.
struct SettingsView: View {
    @Environment(AppState.self) private var app
    @State private var confirmModel = false
    @State private var confirmProgress = false

    var body: some View {
        let t = app.trainer
        Form {
            Section {
                LabeledContent("Status") {
                    Text(t.step == 0 ? "Untrained" : (t.isTrained ? "Trained, \(t.step) steps" : "\(t.step) steps, not trained yet"))
                }
                if let v = t.latestValLoss {
                    LabeledContent("Validation loss") {
                        Text(String(format: "%.2f (random guessing: %.2f)", v, t.uniformLoss)).monospacedDigit()
                    }
                }
                HStack {
                    Button("Show in Finder") {
                        if let url = Trainer.defaultSaveURL, FileManager.default.fileExists(atPath: url.path) {
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        }
                    }
                    .disabled(!(Trainer.defaultSaveURL.map { FileManager.default.fileExists(atPath: $0.path) } ?? false))
                    Button("Reset Model…", role: .destructive) { confirmModel = true }
                        .disabled(t.step == 0)
                }
            } header: {
                Label("Your tiny model", systemImage: "cpu")
            } footer: {
                Text("Saved when training stops, so it's still trained next time you open the app.")
            }

            Section {
                LabeledContent("Chapters read") { Text("\(app.readCount) of \(Chapter.chapters.count)").monospacedDigit() }
                LabeledContent("Guesses made") { Text("\(app.guesses.count)").monospacedDigit() }
                Button("Reset Reading Progress…", role: .destructive) { confirmProgress = true }
                    .disabled(app.readCount == 0 && app.guesses.isEmpty)
            } header: {
                Label("Reading", systemImage: "book")
            } footer: {
                Text("A chapter counts as read after 20 seconds on it, or when you reach its end.")
            }

            Section {
                Text("Everything stays on this Mac. The model is a file in ~/Library/Application Support/LLM Understander; reading progress and guesses are in the app's preferences. Nothing is sent anywhere. Feedback opens a GitHub issue in your browser, and nothing is sent until you submit it there.")
                    .fixedSize(horizontal: false, vertical: true)
                Text("Animations follow System Settings → Accessibility → Display → Reduce Motion.")
                    .foregroundStyle(.secondary)
            } header: {
                Label("Privacy and motion", systemImage: "lock")
            }
        }
        .formStyle(.grouped)
        .frame(width: 480)
        .fixedSize(horizontal: false, vertical: true)
        .confirmationDialog("Reset the tiny model?", isPresented: $confirmModel) {
            Button("Reset Model", role: .destructive) { t.reset() }
        } message: {
            Text("This deletes the trained weights and its charts. Training again takes about a minute.")
        }
        .confirmationDialog("Reset reading progress?", isPresented: $confirmProgress) {
            Button("Reset Progress", role: .destructive) { app.resetProgress() }
        } message: {
            Text("This forgets which chapters you've read and your guesses. Your model is kept.")
        }
    }
}
