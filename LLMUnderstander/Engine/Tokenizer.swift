import Foundation

/// Character-level tokenizer: one token per character. This is what the tiny model uses.
struct CharTokenizer {
    let vocab: [Character]
    private let index: [Character: Int]

    init(text: String) {
        vocab = Array(Set(text)).sorted()
        var idx: [Character: Int] = [:]
        for (i, c) in vocab.enumerated() { idx[c] = i }
        index = idx
    }

    var size: Int { vocab.count }

    func encode(_ s: String) -> [Int] { s.compactMap { index[$0] } }
    func decode(_ ids: [Int]) -> String { String(ids.map { vocab[$0] }) }
    func contains(_ c: Character) -> Bool { index[c] != nil }

    /// Human-readable name for a token (spaces and newlines are invisible otherwise).
    func display(_ id: Int) -> String { CharTokenizer.display(vocab[id]) }

    static func display(_ c: Character) -> String {
        switch c {
        case " ": return "␣"
        case "\n": return "↵"
        default: return String(c)
        }
    }

    /// Lowercase and drop characters the model has never seen.
    func clean(_ s: String) -> String { String(s.lowercased().filter { index[$0] != nil }) }

    static let shared = CharTokenizer(text: Corpus.training + Corpus.validation)
}

/// Byte-pair encoding, trained for real on the corpus. Real LLM tokenizers
/// (GPT, Llama, Claude) are BPE variants with ~50k–200k merges instead of a few dozen.
struct BPETokenizer {
    struct Merge: Hashable {
        let left: String
        let right: String
        var joined: String { left + right }
    }

    private(set) var merges: [Merge] = []
    /// How often each merged pair appeared at the moment it was chosen.
    private(set) var mergeCounts: [Int] = []

    /// Learn `numMerges` merges from `text`. Words are split on spaces (spaces are kept
    /// as a leading "␣" so they survive), as GPT-2 style tokenizers do.
    static func train(on text: String, numMerges: Int) -> BPETokenizer {
        var words: [[String]: Int] = [:]
        for w in pretokenize(text) { words[w.map { String($0) }, default: 0] += 1 }
        var tok = BPETokenizer()
        for _ in 0..<numMerges {
            var pairs: [Merge: Int] = [:]
            for (w, n) in words where w.count > 1 {
                for i in 0..<(w.count - 1) { pairs[Merge(left: w[i], right: w[i + 1]), default: 0] += n }
            }
            // Most frequent pair; break ties alphabetically so training is deterministic.
            guard let best = pairs.max(by: { a, b in
                a.value != b.value ? a.value < b.value : a.key.joined > b.key.joined
            }) else { break }
            tok.merges.append(best.key)
            tok.mergeCounts.append(best.value)
            var next: [[String]: Int] = [:]
            for (w, n) in words { next[apply(best.key, to: w), default: 0] += n }
            words = next
        }
        return tok
    }

    static func pretokenize(_ text: String) -> [String] {
        var out: [String] = []
        var cur = ""
        for c in text {
            if c == " " || c == "\n" {
                if !cur.isEmpty { out.append(cur) }
                cur = c == " " ? "␣" : ""
                if c == "\n" { out.append("↵") }
            } else {
                cur.append(c)
            }
        }
        if !cur.isEmpty { out.append(cur) }
        return out
    }

    static func apply(_ m: Merge, to w: [String]) -> [String] {
        var out: [String] = []
        var i = 0
        while i < w.count {
            if i < w.count - 1 && w[i] == m.left && w[i + 1] == m.right {
                out.append(m.joined); i += 2
            } else {
                out.append(w[i]); i += 1
            }
        }
        return out
    }

    /// Tokenize text using the first `upTo` merges (lets the UI scrub through training).
    func encode(_ text: String, upTo: Int? = nil) -> [String] {
        let use = merges.prefix(upTo ?? merges.count)
        var out: [String] = []
        for word in BPETokenizer.pretokenize(text) {
            var w = word.map { String($0) }
            if word == "↵" { w = ["↵"] }
            for m in use { w = BPETokenizer.apply(m, to: w) }
            out.append(contentsOf: w)
        }
        return out
    }
}
