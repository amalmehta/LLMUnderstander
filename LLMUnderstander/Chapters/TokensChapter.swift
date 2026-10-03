import SwiftUI

struct TokensChapter: View {
    @State private var text = "the old man finds a warm cake in the garden."
    @State private var mode = 1
    @State private var mergesUsed = 60.0
    @State private var bpe: BPETokenizer?

    private let tok = CharTokenizer.shared

    var body: some View {
        ChapterPage(chapter: .tokens) {
            Prose("A neural network only does arithmetic, so the first step is turning text into numbers. We chop the text into **tokens** and give each distinct token an ID. The set of all tokens is the **vocabulary**.")
            Prose("There are three obvious ways to chop. **Characters** give a tiny vocabulary but very long sequences. **Whole words** give short sequences but a huge vocabulary, and new words are impossible. Real LLMs use the middle path: **subwords**, learned by an algorithm called **byte-pair encoding (BPE)**.")

            Card(title: "Try it", subtitle: "Type anything — see how it becomes token IDs") {
                TextField("Type some text", text: $text)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 15, design: .monospaced))
                Picker("", selection: $mode) {
                    Text("Characters (what our tiny model uses)").tag(0)
                    Text("BPE subwords (what real LLMs use)").tag(1)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                if mode == 1 {
                    HStack {
                        Text("Merges learned: \(Int(mergesUsed))").monospacedDigit().frame(width: 150, alignment: .leading)
                        Slider(value: $mergesUsed, in: 0...200, step: 1)
                    }
                    .font(.callout)
                }

                let pieces = tokens
                FlowLayout(spacing: 4) {
                    ForEach(Array(pieces.enumerated()), id: \.offset) { i, p in
                        TokenChip(text: p.text, color: Palette.token(p.id), sub: "\(p.id)")
                    }
                }
                .padding(.vertical, 4)
                HStack(spacing: 12) {
                    StatTile(label: "Characters", value: "\(text.count)")
                    StatTile(label: "Tokens", value: "\(pieces.count)", color: .orange)
                    StatTile(label: "Characters per token", value: String(format: "%.1f", Double(text.count) / Double(max(pieces.count, 1))), color: .green)
                }
                Text(mode == 0
                     ? "Each character is one token. Characters the model never saw in training (capitals, digits, most punctuation) are dropped here."
                     : "Drag the slider: with 0 merges BPE is just characters. Each merge glues the most common neighbouring pair into a new token, so frequent words become single tokens.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if let bpe {
                Card(title: "Watch BPE learn its vocabulary", subtitle: "Trained live on this app's corpus. Each row is one merge, in the order it was learned.") {
                    ScrollView {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 175), spacing: 8)], spacing: 6) {
                            ForEach(Array(bpe.merges.prefix(200).enumerated()), id: \.offset) { i, m in
                                HStack(spacing: 4) {
                                    Text("\(i + 1).").font(.caption2).foregroundStyle(.secondary).frame(width: 28, alignment: .trailing)
                                    Text(m.left).font(.system(size: 12, design: .monospaced))
                                    Text("+").font(.caption2).foregroundStyle(.secondary)
                                    Text(m.right).font(.system(size: 12, design: .monospaced))
                                    Text("→").font(.caption2).foregroundStyle(.secondary)
                                    Text(m.joined).font(.system(size: 12, weight: .semibold, design: .monospaced))
                                    Text("×\(bpe.mergeCounts[i])").font(.system(size: 9)).foregroundStyle(.secondary)
                                }
                                .opacity(Double(i) < mergesUsed ? 1 : 0.3)
                            }
                        }
                    }
                    .frame(height: 220)
                }
            }

            Card(title: "Our tiny model's whole vocabulary", subtitle: "\(tok.size) tokens — one per character. GPT-2 has 50,257; Llama 3 has 128,256.") {
                FlowLayout(spacing: 4) {
                    ForEach(0..<tok.size, id: \.self) { i in
                        TokenChip(text: tok.display(i), color: Palette.token(i), sub: "\(i)")
                    }
                }
            }

            KeyIdea("From here on the model never sees letters — only lists of IDs like [19, 7, 4, 0, …]. Everything it learns is about which IDs tend to follow which.")

            GoDeeper(title: "how real tokenizers work") {
                Prose("**BPE training:** start with every byte (256 tokens) as the vocabulary. Count every adjacent pair in the training text. Merge the most frequent pair into a new token. Repeat until the vocabulary hits the target size (~50k–200k).")
                Prose("**Byte-level** means the base alphabet is the 256 possible bytes, so *any* text — emoji, Chinese, code — can be encoded with no 'unknown' token.")
                Prose("**Special tokens** like `<|endoftext|>` or chat-role markers are added by hand. They mark document boundaries and who is speaking, and become important in post-training (Chapter 11).")
                Prose("**Rule of thumb:** in English, one token ≈ 4 characters ≈ ¾ of a word. Token counts are how training data, context windows and prices are measured.")
            }
        }
        .task {
            let corpus = Corpus.training
            bpe = await Task.detached { BPETokenizer.train(on: corpus, numMerges: 200) }.value
        }
    }

    private struct Piece { let text: String; let id: Int }

    private var tokens: [Piece] {
        if mode == 0 {
            return tok.encode(tok.clean(text)).map { Piece(text: tok.display($0), id: $0) }
        }
        guard let bpe else { return [] }
        let pieces = bpe.encode(text.lowercased(), upTo: Int(mergesUsed))
        // Give each distinct piece a stable ID: characters first, then merges in order.
        let base = tok.vocab.map { String($0) }
        let merged = bpe.merges.map(\.joined)
        return pieces.map { p in
            let raw = p.replacingOccurrences(of: "␣", with: " ").replacingOccurrences(of: "↵", with: "\n")
            if let i = base.firstIndex(of: raw) { return Piece(text: p, id: i) }
            if let i = merged.firstIndex(of: p) { return Piece(text: p, id: base.count + i) }
            return Piece(text: p, id: 999)
        }
    }
}
