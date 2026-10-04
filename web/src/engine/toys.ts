import { Num, Rng } from './tensor'
import type { Point } from './trainer'
import { Optimizer, OptimizerKind } from './model'
import { Tensor } from './tensor'

// =====================================================================
// Post-training toy: one prompt, eight whole answers, real update maths.
// =====================================================================

export interface Answer { id: number; short: string; text: string; features: number[]; hiddenPenalty: number }

export const PostTraining = {
  prompt: 'how do i boil an egg?',
  answers: [
    { id: 0, short: 'Keeps listing questions', text: 'how do i fry an egg? how do i poach an egg? how do i…', features: [0.0, 0.0, 0.35], hiddenPenalty: 0 },
    { id: 1, short: 'Short & correct', text: 'Put the egg in boiling water for 9 minutes, then cool it in cold water.', features: [1.0, 0.3, 0.3], hiddenPenalty: 0 },
    { id: 2, short: 'Friendly & correct', text: 'Happy to help! Lower the egg into boiling water, cook 9 minutes for hard-boiled (6 for soft), then cool it in cold water.', features: [0.95, 1.0, 0.45], hiddenPenalty: 0 },
    { id: 3, short: 'Off-topic fact', text: 'Eggs are laid by hens.', features: [0.1, 0.2, 0.15], hiddenPenalty: 0 },
    { id: 4, short: 'Needless refusal', text: "Sorry, I can't help with cooking.", features: [0.0, 0.6, 0.15], hiddenPenalty: 0 },
    { id: 5, short: 'Long rambling essay', text: 'What a wonderful question! Eggs have fascinated cooks for millennia… (400 more words on the history of eggs) …and that is why, eventually, you boil it.', features: [0.65, 1.0, 1.0], hiddenPenalty: 0 },
    { id: 6, short: 'Wrong & unsafe', text: 'Microwave it in its shell for 5 minutes.', features: [0.3, 0.3, 0.2], hiddenPenalty: 2.5 },
    { id: 7, short: 'Curt', text: 'Boil it.', features: [0.4, 0.0, 0.05], hiddenPenalty: 0 },
  ] as Answer[],

  /** How good each answer really is to a careful person (hidden from the model). */
  trueQuality(a: Answer) {
    const [h, p, l] = a.features
    return 3 * h + 0.8 * p + 2 * Math.min(l, 0.45) - 6 * Math.max(l - 0.55, 0) - a.hiddenPenalty
  },
  get quality() { return PostTraining.answers.map(PostTraining.trueQuality) },

  baseLogits: [1.6, -0.6, -0.9, 1.0, -0.2, -1.4, 0.3, 0.2],
  demonstrations: [1, 2, 2, 1, 2],
  sftLogits: [0.5, 1.2, 1.0, 0.0, 0.3, -0.5, 0.0, 0.3],

  sftStep(logits: number[], lr: number) {
    const p = Num.softmax(logits), d = PostTraining.demonstrations
    const g = logits.map(() => 0)
    for (const k of d) for (let j = 0; j < g.length; j++) g[j] += (p[j] - (j === k ? 1 : 0)) / d.length
    return logits.map((l, j) => l - lr * g[j])
  },
  sftLoss(logits: number[]) {
    const p = Num.softmax(logits), d = PostTraining.demonstrations
    return d.reduce((s, k) => s - Math.log(p[k]), 0) / d.length
  },

  /** Labellers compare pairs and slightly favour longer answers; answer 5 never appears. */
  collectComparisons(count = 240, seed = 3) {
    const rng = new Rng(seed), seen = [0, 1, 2, 3, 4, 6, 7], q = PostTraining.quality, A = PostTraining.answers
    return Array.from({ length: count }, (_, i) => {
      const a = seen[rng.int(seen.length)]
      let b = seen[rng.int(seen.length)]
      while (b === a) b = seen[rng.int(seen.length)]
      const pa = Num.sigmoid(q[a] + 3 * A[a].features[2] - (q[b] + 3 * A[b].features[2]))
      return rng.uniform() < pa ? { id: i, chosen: a, rejected: b } : { id: i, chosen: b, rejected: a }
    })
  },

  kl(p: number[], q: number[]) { return p.reduce((s, pi, i) => s + (pi > 0 ? pi * Math.log(pi / q[i]) : 0), 0) },
  expected(p: number[], v: number[]) { return p.reduce((s, pi, i) => s + pi * v[i], 0) },
  logProbs(l: number[]) { return Num.softmax(l).map(Math.log) },

  /** E_π[r] − β·KL(π‖π_sft); natural-gradient step whose fixed point is π ∝ π_sft·exp(r/β). */
  rlhfStep(logits: number[], rewards: number[], beta: number, lr = 0.3) {
    return logits.map((l, i) => l + lr * (rewards[i] - beta * (l - PostTraining.sftLogits[i])))
  },

  dpoStep(logits: number[], data: Comparison[], beta: number, lr: number) {
    const lp = PostTraining.logProbs(logits), lr0 = PostTraining.logProbs(PostTraining.sftLogits)
    const g = logits.map(() => 0)
    for (const c of data) {
      const z = beta * ((lp[c.chosen] - lr0[c.chosen]) - (lp[c.rejected] - lr0[c.rejected]))
      const coef = 1 - Num.sigmoid(z)
      g[c.chosen] += coef * beta; g[c.rejected] -= coef * beta
    }
    return logits.map((l, i) => l + (lr * g[i]) / data.length)
  },
  dpoLoss(logits: number[], data: Comparison[], beta: number) {
    const lp = PostTraining.logProbs(logits), lr0 = PostTraining.logProbs(PostTraining.sftLogits)
    return data.reduce((s, c) => s - Math.log(Num.sigmoid(beta * ((lp[c.chosen] - lr0[c.chosen]) - (lp[c.rejected] - lr0[c.rejected])))), 0) / data.length
  },
}
export interface Comparison { id: number; chosen: number; rejected: number }

/** Linear reward model r = w·features, trained with the Bradley–Terry loss. */
export class RewardModel {
  w = [0, 0, 0]
  reward(a: Answer) { return this.w.reduce((s, wi, k) => s + wi * a.features[k], 0) }
  get rewards() { return PostTraining.answers.map((a) => this.reward(a)) }
  step(data: Comparison[], lr: number) {
    const A = PostTraining.answers, g = [0, 0, 0]
    for (const c of data) {
      const d = A[c.chosen].features.map((f, k) => f - A[c.rejected].features[k])
      const s = Num.sigmoid(this.w.reduce((acc, wi, k) => acc + wi * d[k], 0))
      for (let k = 0; k < 3; k++) g[k] += (1 - s) * d[k]
    }
    this.w = this.w.map((wi, k) => wi + (lr * g[k]) / data.length)
  }
  accuracy(data: Comparison[]) {
    const A = PostTraining.answers
    return data.filter((c) => this.reward(A[c.chosen]) > this.reward(A[c.rejected])).length / data.length
  }
  static trained(steps = 400) {
    const data = PostTraining.collectComparisons(), rm = new RewardModel()
    for (let i = 0; i < steps; i++) rm.step(data, 0.5)
    return rm
  }
}

// =====================================================================
// Loss landscapes, scaling law, parameter counts, number formatting
// =====================================================================

export type Landscape = 'Narrow ravine' | 'Banana valley' | 'Bumpy bowl'
export const landscapes: Landscape[] = ['Narrow ravine', 'Banana valley', 'Bumpy bowl']

export const Land = {
  range(l: Landscape): { x: [number, number]; y: [number, number] } {
    return l === 'Narrow ravine' ? { x: [-3, 3], y: [-2, 2] } : l === 'Banana valley' ? { x: [-2, 2], y: [-1, 3] } : { x: [-3, 3], y: [-3, 3] }
  },
  start(l: Landscape): [number, number] {
    return l === 'Narrow ravine' ? [-2.6, 1.5] : l === 'Banana valley' ? [-1.6, 2.6] : [-2.5, 2.4]
  },
  value(l: Landscape, x: number, y: number) {
    if (l === 'Narrow ravine') return 0.02 * x * x + 2.5 * y * y
    if (l === 'Banana valley') return 0.05 * ((1 - x) ** 2 + 10 * (y - x * x) ** 2)
    return 0.15 * (x * x + y * y) + 0.6 * Math.sin(2 * x) * Math.sin(2 * y) + 0.6
  },
  path(l: Landscape, kind: OptimizerKind, lr: number, steps: number): [number, number][] {
    const p = new Tensor(1, 2, Land.start(l))
    const opt = new Optimizer(kind)
    opt.weightDecay = 0; opt.beta2 = 0.999
    const out: [number, number][] = [[p.data[0], p.data[1]]]
    const eff = kind === 'AdamW' ? lr * 0.15 : lr, h = 1e-5
    for (let i = 0; i < steps; i++) {
      const [x, y] = [p.data[0], p.data[1]]
      p.grad[0] = (Land.value(l, x + h, y) - Land.value(l, x - h, y)) / (2 * h)
      p.grad[1] = (Land.value(l, x, y + h) - Land.value(l, x, y - h)) / (2 * h)
      opt.step([p], eff)
      p.data[0] = Math.min(Math.max(p.data[0], -50), 50); p.data[1] = Math.min(Math.max(p.data[1], -50), 50)
      out.push([p.data[0], p.data[1]])
    }
    return out
  },
}

/** Chinchilla-form scaling law with constants from the Epoch AI (2024) replication. */
export const ScalingLaw = {
  E: 1.82, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658,
  loss(n: number, d: number) { return this.E + this.A / n ** this.alpha + this.B / d ** this.beta },
  tokens(c: number, n: number) { return c / (6 * n) },
  sweep(c: number): [number, number][] {
    const out: [number, number][] = []
    for (let lg = 6; lg <= 13.0001; lg += 0.05) { const n = 10 ** lg; out.push([lg, this.loss(n, this.tokens(c, n))]) }
    return out
  },
  optimum(c: number) {
    const best = this.sweep(c).reduce((a, b) => (b[1] < a[1] ? b : a))
    const n = 10 ** best[0]
    return { params: n, tokens: this.tokens(c, n), loss: best[1] }
  },
}

export const paramEstimate = (layers: number, d: number, vocab: number) => 12 * layers * d * d + vocab * d

export const fmt = {
  big(x: number) {
    const a = Math.abs(x)
    if (a >= 1e12) return (x / 1e12).toFixed(1) + 'T'
    if (a >= 1e9) return (x / 1e9).toFixed(1) + 'B'
    if (a >= 1e6) return (x / 1e6).toFixed(1) + 'M'
    if (a >= 1e3) return (x / 1e3).toFixed(1) + 'K'
    return x.toFixed(0)
  },
  sci(x: number) {
    const e = Math.floor(Math.log10(x)), m = x / 10 ** e
    return `${m.toFixed(1)}×10^${e}`
  },
  pct(x: number, digits = 0) { return (x * 100).toFixed(digits) + '%' },
  n(x: number, digits = 2) { return x.toFixed(digits) },
}

export type { Point }
