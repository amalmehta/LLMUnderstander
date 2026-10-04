import { describe, expect, it } from 'vitest'
import { Ops, Rng, Tensor, Num } from '../src/engine/tensor'
import { BPE, TinyGPT, tokenizer, TRAINING_TEXT, scheduledLR } from '../src/engine/model'
import { Trainer } from '../src/engine/trainer'
import { PostTraining, RewardModel, ScalingLaw, Land } from '../src/engine/toys'
import { AgentRL, AgentScript, Bandit, ContextBudget, Sampling } from '../src/engine/agents'

/** Compare autograd gradients with finite differences on a few entries of each parameter. */
function gradCheck(params: Tensor[], f: () => Tensor, samples = 4, tol = 1e-4) {
  params.forEach((p) => p.zeroGrad())
  f().backward()
  const rng = new Rng(9)
  for (const p of params) {
    const analytic = Float64Array.from(p.grad)
    for (let s = 0; s < samples; s++) {
      const i = rng.int(p.count), old = p.data[i], h = 1e-5
      p.data[i] = old + h; const up = f().item
      p.data[i] = old - h; const down = f().item
      p.data[i] = old
      const numeric = (up - down) / (2 * h)
      const rel = Math.abs(numeric - analytic[i]) / Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic[i]))
      expect(rel, `${p.label}[${i}] numeric ${numeric} vs analytic ${analytic[i]}`).toBeLessThan(tol)
    }
  }
}

describe('autograd and model', () => {
  it('every op has correct gradients', () => {
    const rng = new Rng(1)
    const a = Tensor.randn(4, 3, 1, rng, 'a'), b = Tensor.randn(3, 4, 1, rng, 'b')
    const g = Tensor.randn(1, 4, 1, rng, 'gamma'), bb = Tensor.randn(1, 4, 1, rng, 'beta')
    gradCheck([a, b, g, bb], () => {
      const m = Ops.matmul(a, b)
      const s = Ops.causalSoftmax(Ops.scale(m, 0.7))
      const ln = Ops.layerNorm(Ops.gelu(Ops.matmul(s, Ops.transpose(m))), g, bb)
      const cat = Ops.concatCols([Ops.sliceCols(ln, 0, 2), Ops.sliceCols(ln, 2, 2)])
      return Ops.crossEntropy(Ops.addRow(cat, bb), [0, 2, 1, 3], [1, 0, 1, 1])
    })
  })

  it('the whole model has correct gradients', () => {
    const m = new TinyGPT({ vocab: 7, context: 6, dModel: 8, nHeads: 2, nLayers: 2 }, 3)
    gradCheck(m.parameters, () => m.loss([[[1, 4, 2, 0, 6, 3], [4, 2, 0, 6, 3, 5]]]))
  })

  it('attention rows sum to one and are causal', () => {
    const m = new TinyGPT({ vocab: tokenizer.size, context: 32, dModel: 32, nHeads: 2, nLayers: 2 })
    const t = m.trace(tokenizer.encode('the cat sees'))
    for (const head of t.attention[0]) head.forEach((row, i) => {
      expect(Num.sum(row)).toBeCloseTo(1, 9)
      for (let j = i + 1; j < row.length; j++) expect(row[j]).toBe(0)
    })
  })

  it('training reduces validation loss', () => {
    const tr = new Trainer(false)
    tr.settings.totalSteps = 300; tr.settings.warmup = 20
    const before = tr.validationLoss()
    const t0 = performance.now()
    tr.trainNow(300)
    const after = tr.validationLoss()
    console.log(`val loss ${before.toFixed(3)} → ${after.toFixed(3)} in ${((performance.now() - t0) / 1000).toFixed(1)}s; sample: ${tr.sampleText()}`)
    expect(after).toBeLessThan(before - 1)
  }, 120_000)

  it('BPE learns common chunks and round-trips', () => {
    const bpe = BPE.train(TRAINING_TEXT, 40)
    expect(bpe.merges.length).toBe(40)
    const s = 'the cat sees the fish.'
    const toks = bpe.encode(s)
    expect(toks.length).toBeLessThan(s.length)
    expect(toks.join('').replaceAll('␣', ' ')).toBe(s)
  })

  it('learning-rate schedule warms up then decays', () => {
    expect(scheduledLR(1, 100, 1000, 0)).toBeCloseTo(0.01)
    expect(scheduledLR(1, 100, 1000, 99)).toBeCloseTo(1)
    expect(scheduledLR(1, 100, 1000, 1000)).toBeCloseTo(0.1)
  })
})

describe('post-training toys', () => {
  it('SFT moves probability onto the demonstrations', () => {
    let l = PostTraining.baseLogits
    const before = PostTraining.sftLoss(l)
    for (let i = 0; i < 100; i++) l = PostTraining.sftStep(l, 0.5)
    expect(PostTraining.sftLoss(l)).toBeLessThan(before)
    const p = Num.softmax(l)
    expect(p[1] + p[2]).toBeGreaterThan(0.8)
  })

  it('the reward model over-rates the essay, and a weak KL leash reward-hacks', () => {
    const data = PostTraining.collectComparisons(), rm = RewardModel.trained(), r = rm.rewards
    console.log('reward model', rm.w.map((x) => x.toFixed(2)), 'acc', rm.accuracy(data).toFixed(2), 'rewards', r.map((x) => x.toFixed(2)))
    expect(rm.accuracy(data)).toBeGreaterThan(0.6)
    expect(Num.argmax(r)).toBe(5)
    const run = (beta: number) => { let l = PostTraining.sftLogits; for (let i = 0; i < 600; i++) l = PostTraining.rlhfStep(l, r, beta); return Num.softmax(l) }
    const low = run(0.05), mid = run(0.5)
    expect(low[5]).toBeGreaterThan(0.8)
    expect(PostTraining.expected(low, PostTraining.quality)).toBeLessThan(PostTraining.expected(mid, PostTraining.quality))
  })

  it('DPO reduces its loss', () => {
    const data = PostTraining.collectComparisons()
    let l = PostTraining.sftLogits
    const before = PostTraining.dpoLoss(l, data, 0.5)
    for (let i = 0; i < 200; i++) l = PostTraining.dpoStep(l, data, 0.5, 1)
    expect(PostTraining.dpoLoss(l, data, 0.5)).toBeLessThan(before)
  })

  it('scaling-law optimum is about 20 tokens per parameter', () => {
    const o = ScalingLaw.optimum(1e23)
    expect(o.tokens / o.params).toBeGreaterThan(5)
    expect(o.tokens / o.params).toBeLessThan(60)
  })

  it('Adam reaches the ravine minimum', () => {
    const p = Land.path('Narrow ravine', 'AdamW', 0.5, 300).at(-1)!
    expect(Land.value('Narrow ravine', p[0], p[1])).toBeLessThan(0.05)
  })
})

describe('agents and bandits', () => {
  it('agent RL hacks a naive reward but not a robust one', () => {
    const train = (robust: boolean) => {
      const st = AgentRL.newState(), rng = new Rng(21)
      for (let i = 0; i < 150; i++) AgentRL.step(st, robust, rng)
      return Num.softmax(st.logits)
    }
    const naive = train(false), robust = train(true)
    expect(naive[2]).toBeGreaterThan(0.6)
    expect(robust[0]).toBeGreaterThan(0.6)
    expect(AgentRL.genuineSuccess(robust)).toBeGreaterThan(AgentRL.genuineSuccess(naive))
  })

  it('the scripted run ends with an answer, its context grows, and tool output is never trained on', () => {
    const ev = AgentScript.events
    expect(ev.at(-1)!.kind).toBe('final answer')
    expect(AgentScript.contextTokens(ev.length - 1)).toBeGreaterThan(AgentScript.contextTokens(2))
    expect(ev.filter((e) => e.kind === 'tool result').some(AgentScript.trainedOn)).toBe(false)
  })

  it('compaction keeps the context under the window', () => {
    expect(ContextBudget.segments(80, false).overflow).toBe(true)
    const on = ContextBudget.segments(80, true)
    expect(on.overflow).toBe(false)
    expect(on.compactions).toBeGreaterThan(0)
  })

  it('Thompson sampling beats greedy and has the fewest bad runs', () => {
    const runs = (s: Parameters<typeof Bandit.run>[0]) => Array.from({ length: 150 }, (_, i) => Bandit.run(s, 600, i + 1).regret)
    const mean = (r: number[]) => r.reduce((a, b) => a + b) / r.length
    const bad = (r: number[]) => r.filter((x) => x > 60).length / r.length
    const g = runs('Greedy'), e = runs('ε-greedy (10%)'), t = runs('Thompson sampling')
    console.log(`bandit mean regret: greedy ${mean(g).toFixed(1)} ε ${mean(e).toFixed(1)} thompson ${mean(t).toFixed(1)}; bad runs ${bad(g)} ${bad(e)} ${bad(t)}`)
    expect(mean(t)).toBeLessThan(mean(g))
    expect(bad(t)).toBeLessThan(bad(g))
    expect(bad(t)).toBeLessThanOrEqual(bad(e))
  })

  it('voting maths and Beta sampling', () => {
    expect(Sampling.majority(0.6, 1)).toBeCloseTo(0.6, 12)
    expect(Sampling.majority(0.6, 3)).toBeCloseTo(0.648, 12)
    expect(Sampling.majority(0.4, 9)).toBeLessThan(0.4)
    expect(Sampling.anyCorrect(0.5, 3)).toBeCloseTo(0.875, 12)
    const rng = new Rng(3)
    const draws = Array.from({ length: 4000 }, () => Bandit.betaSample(3, 7, rng))
    expect(Num.sum(draws) / 4000).toBeCloseTo(0.3, 1)
  })
})
