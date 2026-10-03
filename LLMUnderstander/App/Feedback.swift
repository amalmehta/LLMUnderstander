import SwiftUI

/// Small, unobtrusive feedback tab at the bottom of the sidebar.
struct FeedbackTab: View {
    @Environment(AppState.self) private var app
    var body: some View {
        HStack {
            Button { app.showFeedback = true } label: {
                Label("Feedback", systemImage: "bubble.left")
                    .font(.caption)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }
}

struct FeedbackSheet: View {
    @Environment(AppState.self) private var app
    @Environment(\.dismiss) private var dismiss
    @State private var kind = "Idea"
    @State private var text = ""

    static let repo = "https://github.com/amalmehta/LLMUnderstander"

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Send Feedback").font(.title2.weight(.semibold))
            Text("This opens a pre-filled GitHub issue in your browser — nothing is sent until you submit it there.")
                .font(.callout).foregroundStyle(.secondary)
            Picker("Type", selection: $kind) {
                ForEach(["Idea", "Something's confusing", "Bug"], id: \.self) { Text($0) }
            }
            .pickerStyle(.segmented)
            TextEditor(text: $text)
                .font(.body)
                .frame(minHeight: 140)
                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.quaternary))
            HStack {
                Text("About: \(app.selected?.title ?? "—")").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Open on GitHub") { open(); dismiss() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(22)
        .frame(width: 500)
    }

    private func open() {
        let chapter = app.selected.map { "Chapter \($0.number): \($0.title)" } ?? "General"
        let title = "\(kind): \(chapter)"
        let body = "\(text)\n\n---\nChapter: \(chapter)\nApp version: \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?")"
        var c = URLComponents(string: "\(Self.repo)/issues/new")!
        c.queryItems = [URLQueryItem(name: "title", value: title), URLQueryItem(name: "body", value: body)]
        if let url = c.url { NSWorkspace.shared.open(url) }
    }
}
