import Foundation

/// A fixed 40-item word-completion benchmark for the tiny model.
///
/// Each item is the start of a held-out sentence, cut one or two letters into a word;
/// the model must finish the word. It's scored two ways on the same answers:
/// - **exact match**: the whole rest of the word is right (all or nothing), and
/// - **per character**: the share of the word's remaining letters it predicts right,
///   each given the correct letters before it.
/// Loss and per-character accuracy improve smoothly; exact match can look like it
/// "suddenly" appears. Schaeffer, Miranda and Koyejo (2023) argued that many
/// "emergent abilities" are this effect: a sharp metric on a smooth improvement.
enum Benchmark {
    struct Item: Hashable, Identifiable {
        let prompt: String
        let answer: String          // rest of the word, without the following space or full stop
        var id: String { prompt }
    }

    static let items: [Item] = {
        var rng = SplitMix64(seed: 4040)
        var out: [Item] = []
        var seen = Set<String>()
        while out.count < 40 {
            let s = Corpus.sentence(&rng)
            // Pick a word after the first one, so there's some context, and cut 1–2 letters in.
            let words = s.dropLast().split(separator: " ").map(String.init)   // drop the full stop
            guard words.count >= 3 else { continue }
            let w = 1 + rng.int(words.count - 1)
            let word = words[w]
            guard word.count >= 4 else { continue }
            let cut = 1 + rng.int(2)
            let prompt = words[0..<w].joined(separator: " ") + " " + String(word.prefix(cut))
            let item = Item(prompt: prompt, answer: String(word.dropFirst(cut)))
            if seen.insert(prompt).inserted { out.append(item) }
        }
        return out
    }()

    struct Score: Codable, Equatable {
        var exact: Double
        var perCharacter: Double
    }

    /// The model's greedy completion of the word (stops at a space or full stop).
    static func complete(_ model: TinyGPT, _ item: Item, tokenizer: CharTokenizer = .shared) -> String {
        var ids = tokenizer.encode(item.prompt)
        var out = ""
        for _ in 0..<(item.answer.count + 2) {
            let p = model.nextTokenProbs(Array(ids.suffix(model.config.context)))
            let next = p.indices.max { p[$0] < p[$1] }!
            let c = tokenizer.vocab[next]
            if c == " " || c == "." || c == "\n" { break }
            out.append(c); ids.append(next)
        }
        return out
    }

    static func score(_ model: TinyGPT, tokenizer: CharTokenizer = .shared) -> Score {
        var exact = 0.0, chars = 0.0, total = 0.0
        for item in items {
            if complete(model, item, tokenizer: tokenizer) == item.answer { exact += 1 }
            // Teacher-forced: predict each letter given the right ones before it.
            var ids = tokenizer.encode(item.prompt)
            for c in item.answer {
                let p = model.nextTokenProbs(Array(ids.suffix(model.config.context)))
                let best = p.indices.max { p[$0] < p[$1] }!
                let truth = tokenizer.encode(String(c))[0]
                if best == truth { chars += 1 }
                total += 1
                ids.append(truth)
            }
        }
        return Score(exact: exact / Double(items.count), perCharacter: chars / max(total, 1))
    }

    /// Standard error of a score p measured on n independent items: √(p(1 − p) / n).
    static func standardError(p: Double, n: Int) -> Double {
        (p * (1 - p) / Double(max(n, 1))).squareRoot()
    }
}
