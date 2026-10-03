import Foundation

struct GPTConfig: Equatable {
    var vocab: Int
    var context: Int = 32
    var dModel: Int = 32
    var nHeads: Int = 2
    var nLayers: Int = 2

    var headDim: Int { dModel / nHeads }
}

/// What happened inside the model on one forward pass — kept so the app can show it.
struct ForwardTrace {
    var tokens: [Int] = []
    var embeddings: [[Double]] = []           // token + position, T × d
    /// [layer][head] → T×T attention weights (rows = query position)
    var attention: [[[[Double]]]] = []
    /// [layer][head] → T×T raw scaled scores, before mask + softmax
    var scores: [[[[Double]]]] = []
    /// size of the residual stream (root-mean-square) after each layer, per position
    var residualRMS: [[Double]] = []
    var logits: [[Double]] = []                // T × vocab
}

/// A real (tiny) GPT: token + position embeddings, N pre-norm transformer blocks
/// with multi-head causal self-attention and a GELU MLP, final layer norm, linear head.
final class TinyGPT {
    final class Block {
        let ln1g, ln1b, wq, wk, wv, wo, ln2g, ln2b, w1, b1, w2, b2: Tensor
        init(_ c: GPTConfig, layer: Int, rng: inout SplitMix64) {
            let d = c.dModel
            let s = 1 / Double(d).squareRoot()
            let resid = s / Double(2 * c.nLayers).squareRoot()
            ln1g = .filled(1, d, 1, label: "L\(layer).ln1.gain"); ln1b = .filled(1, d, 0, label: "L\(layer).ln1.bias")
            wq = .randn(d, d, std: s, rng: &rng, label: "L\(layer).attn.Wq")
            wk = .randn(d, d, std: s, rng: &rng, label: "L\(layer).attn.Wk")
            wv = .randn(d, d, std: s, rng: &rng, label: "L\(layer).attn.Wv")
            wo = .randn(d, d, std: resid, rng: &rng, label: "L\(layer).attn.Wo")
            ln2g = .filled(1, d, 1, label: "L\(layer).ln2.gain"); ln2b = .filled(1, d, 0, label: "L\(layer).ln2.bias")
            w1 = .randn(d, 4 * d, std: s, rng: &rng, label: "L\(layer).mlp.W1")
            b1 = .filled(1, 4 * d, 0, label: "L\(layer).mlp.b1")
            w2 = .randn(4 * d, d, std: resid / 2, rng: &rng, label: "L\(layer).mlp.W2")
            b2 = .filled(1, d, 0, label: "L\(layer).mlp.b2")
        }
        init(copying o: Block) {
            ln1g = o.ln1g.copy(); ln1b = o.ln1b.copy(); wq = o.wq.copy(); wk = o.wk.copy(); wv = o.wv.copy()
            wo = o.wo.copy(); ln2g = o.ln2g.copy(); ln2b = o.ln2b.copy(); w1 = o.w1.copy(); b1 = o.b1.copy()
            w2 = o.w2.copy(); b2 = o.b2.copy()
        }
        var parameters: [Tensor] { [ln1g, ln1b, wq, wk, wv, wo, ln2g, ln2b, w1, b1, w2, b2] }
    }

    let config: GPTConfig
    let wte: Tensor   // vocab × d   token embeddings
    let wpe: Tensor   // context × d position embeddings
    let blocks: [Block]
    let lnfg, lnfb: Tensor
    let head: Tensor  // d × vocab
    let headB: Tensor

    init(config: GPTConfig, seed: UInt64 = 1) {
        self.config = config
        var rng = SplitMix64(seed: seed)
        let d = config.dModel
        wte = .randn(config.vocab, d, std: 0.3, rng: &rng, label: "token embeddings")
        wpe = .randn(config.context, d, std: 0.1, rng: &rng, label: "position embeddings")
        var bs: [Block] = []
        for l in 0..<config.nLayers { bs.append(Block(config, layer: l, rng: &rng)) }
        blocks = bs
        lnfg = .filled(1, d, 1, label: "final.ln.gain"); lnfb = .filled(1, d, 0, label: "final.ln.bias")
        head = .randn(d, config.vocab, std: 0.02, rng: &rng, label: "output head")
        headB = .filled(1, config.vocab, 0, label: "output bias")
    }

    private init(copying o: TinyGPT) {
        config = o.config
        wte = o.wte.copy(); wpe = o.wpe.copy()
        blocks = o.blocks.map { Block(copying: $0) }
        lnfg = o.lnfg.copy(); lnfb = o.lnfb.copy(); head = o.head.copy(); headB = o.headB.copy()
    }

    func copy() -> TinyGPT { TinyGPT(copying: self) }

    var parameters: [Tensor] { [wte, wpe] + blocks.flatMap(\.parameters) + [lnfg, lnfb, head, headB] }
    var parameterCount: Int { parameters.reduce(0) { $0 + $1.count } }

    /// Forward pass over one sequence (length ≤ context). Returns T×vocab logits.
    func forward(_ ids: [Int], trace: UnsafeMutablePointer<ForwardTrace>? = nil) -> Tensor {
        let T = ids.count
        precondition(T >= 1 && T <= config.context)
        let hd = config.headDim
        let scale = 1 / Double(hd).squareRoot()

        var x = Ops.add(Ops.embedding(wte, ids), Ops.firstRows(wpe, T))
        trace?.pointee.tokens = ids
        trace?.pointee.embeddings = (0..<T).map { x.row($0) }

        for blk in blocks {
            // --- multi-head causal self-attention ---
            let h = Ops.layerNorm(x, blk.ln1g, blk.ln1b)
            let q = Ops.matmul(h, blk.wq), k = Ops.matmul(h, blk.wk), v = Ops.matmul(h, blk.wv)
            var heads: [Tensor] = []
            var layerAtt: [[[Double]]] = [], layerScores: [[[Double]]] = []
            for hh in 0..<config.nHeads {
                let qh = Ops.sliceCols(q, hh * hd, hd)
                let kh = Ops.sliceCols(k, hh * hd, hd)
                let vh = Ops.sliceCols(v, hh * hd, hd)
                let s = Ops.scale(Ops.matmul(qh, Ops.transpose(kh)), scale)
                let a = Ops.causalSoftmax(s)
                heads.append(Ops.matmul(a, vh))
                if trace != nil {
                    layerAtt.append((0..<T).map { a.row($0) })
                    layerScores.append((0..<T).map { s.row($0) })
                }
            }
            let att = Ops.matmul(Ops.concatCols(heads), blk.wo)
            x = Ops.add(x, att)                                   // residual
            // --- MLP ---
            let h2 = Ops.layerNorm(x, blk.ln2g, blk.ln2b)
            let m = Ops.addRow(Ops.matmul(Ops.gelu(Ops.addRow(Ops.matmul(h2, blk.w1), blk.b1)), blk.w2), blk.b2)
            x = Ops.add(x, m)                                     // residual
            if trace != nil {
                trace!.pointee.attention.append(layerAtt)
                trace!.pointee.scores.append(layerScores)
                trace!.pointee.residualRMS.append((0..<T).map { r in
                    let row = x.row(r)
                    return (row.reduce(0) { $0 + $1 * $1 } / Double(row.count)).squareRoot()
                })
            }
        }
        let logits = Ops.addRow(Ops.matmul(Ops.layerNorm(x, lnfg, lnfb), head), headB)
        trace?.pointee.logits = (0..<T).map { logits.row($0) }
        return logits
    }

    /// Run the model and record what happened inside.
    func trace(_ ids: [Int]) -> ForwardTrace {
        var t = ForwardTrace()
        _ = withUnsafeMutablePointer(to: &t) { forward(Array(ids.suffix(config.context)), trace: $0) }
        return t
    }

    /// Mean next-token loss over a batch of (input, target) sequences.
    func loss(batch: [([Int], [Int])]) -> Tensor {
        Ops.mean(batch.map { Ops.crossEntropy(forward($0.0), $0.1) })
    }

    /// Probability of each next token after `context`.
    func nextTokenProbs(_ context: [Int], temperature: Double = 1) -> [Double] {
        let ids = Array(context.suffix(config.context))
        guard !ids.isEmpty else { return Array(repeating: 1 / Double(config.vocab), count: config.vocab) }
        let logits = forward(ids)
        return Num.softmax(logits.row(ids.count - 1), temperature: temperature)
    }

    func generate(prompt: [Int], count: Int, temperature: Double, rng: inout SplitMix64) -> [Int] {
        var ids = prompt
        for _ in 0..<count {
            let p = nextTokenProbs(ids, temperature: temperature)
            ids.append(Num.sample(p, rng: &rng))
        }
        return Array(ids.dropFirst(prompt.count))
    }
}
