import Foundation

/// Exact operation counts and published formulas for the Inference chapter.
enum InferenceMaths {
    struct Shape: Hashable {
        let name: String
        let width: Double      // d
        let layers: Double
        let vocab: Double

        static func tiny(_ c: GPTConfig) -> Shape { Shape(name: "Your tiny model", width: Double(c.dModel), layers: Double(c.nLayers), vocab: Double(c.vocab)) }
        /// GPT-3 175B's published shape (Brown et al., 2020).
        static let gpt3 = Shape(name: "GPT-3 sized (175B)", width: 12_288, layers: 96, vocab: 50_257)
    }

    /// Multiply-adds to produce the token at position t (1-based), using a KV cache.
    /// Per layer: Q, K, V and output projections for one token (4d²), the MLP (8d²),
    /// and attention against t cached keys and values (2td). Plus the output layer (dV).
    static func withCache(_ s: Shape, position t: Double) -> Double {
        s.layers * (12 * s.width * s.width + 2 * t * s.width) + s.width * s.vocab
    }

    /// Without a cache, every earlier token is pushed through every layer again:
    /// t tokens of projections and MLP, and attention for each position i ≤ t.
    static func withoutCache(_ s: Shape, position t: Double) -> Double {
        s.layers * (12 * s.width * s.width * t + t * (t + 1) * s.width) + s.width * s.vocab
    }

    /// Numbers stored in the cache after t tokens: a key and a value vector per layer per token.
    static func cacheSize(_ s: Shape, tokens t: Double) -> Double { 2 * s.layers * s.width * t }

    /// Speculative decoding (Leviathan, Kalman and Matias, 2023): a small draft model proposes k tokens,
    /// the big model checks them in one pass. With acceptance rate α, the expected number of tokens
    /// produced per big-model pass is (1 − α^(k+1)) / (1 − α).
    static func tokensPerPass(acceptance a: Double, draft k: Int) -> Double {
        a >= 0.999 ? Double(k + 1) : (1 - pow(a, Double(k + 1))) / (1 - a)
    }

    /// Expected speed-up when one draft step costs `c` of a big-model pass: tokensPerPass ÷ (k·c + 1).
    static func speedup(acceptance a: Double, draft k: Int, draftCost c: Double) -> Double {
        tokensPerPass(acceptance: a, draft: k) / (Double(k) * c + 1)
    }

    /// Nucleus (top-p) filtering: keep the most likely tokens until their probabilities add up to p.
    /// Returns the kept token ids, most likely first (always at least one).
    static func nucleus(_ probs: [Double], p: Double) -> [Int] {
        let order = probs.indices.sorted { probs[$0] > probs[$1] }
        var kept: [Int] = [], total = 0.0
        for i in order {
            kept.append(i)
            total += probs[i]
            if total >= p { break }
        }
        return kept
    }

    /// Sample from the kept set, renormalised.
    static func sampleNucleus(_ probs: [Double], p: Double, rng: inout SplitMix64) -> Int {
        let kept = nucleus(probs, p: p)
        let ps = kept.map { probs[$0] }
        let sum = ps.reduce(0, +)
        return kept[Num.sample(ps.map { $0 / sum }, rng: &rng)]
    }
}
