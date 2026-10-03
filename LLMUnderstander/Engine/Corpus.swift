import Foundation

/// The "internet" our tiny model is pretrained on: simple English sentences
/// generated from a small grammar. Small enough to learn in a minute on a laptop,
/// structured enough that you can watch real words and grammar emerge.
enum Corpus {
    static let subjects = [
        "the cat", "a dog", "the old man", "my sister", "a small bird", "the farmer",
        "the baker", "a child", "the queen", "my friend", "the fox", "a girl",
    ]
    static let verbs = [
        "sees", "likes", "finds", "eats", "wants", "holds", "chases", "brings", "loves", "makes",
    ]
    static let objects = [
        "the ball", "a red apple", "the fish", "some bread", "a big hat", "the moon",
        "a warm cake", "the river", "a green leaf", "the old boat",
    ]
    static let endings = [
        "", "", "", " today", " in the garden", " at night", " near the house", " again",
    ]

    static func sentence(_ rng: inout SplitMix64) -> String {
        let s = subjects[rng.int(subjects.count)]
        let v = verbs[rng.int(verbs.count)]
        let o = objects[rng.int(objects.count)]
        let e = endings[rng.int(endings.count)]
        return "\(s) \(v) \(o)\(e)."
    }

    /// Paragraphs of sentences, newline-separated.
    static func generate(sentences: Int, seed: UInt64) -> String {
        var rng = SplitMix64(seed: seed)
        var out = ""
        for i in 0..<sentences {
            out += sentence(&rng)
            out += (i % 4 == 3) ? "\n" : " "
        }
        return out
    }

    static let training = generate(sentences: 3000, seed: 7)
    static let validation = generate(sentences: 300, seed: 99)
}
