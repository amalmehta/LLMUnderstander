// A toy reasoning model trained with GRPO (same as Reasoning.swift).
// Sums of 2–6 digits; the policy picks how many additions to write out before answering.
import { Num, Rng } from './tensor'

export const MAX_THINKING = 6
export const DIFFICULTIES = [1, 2, 3, 4, 5]
export const HEAD_ACCURACY = 0.35
export const STEP_ACCURACY = 0.98

export interface GroupAnswer { thinking: number; correct: boolean; reward: number; advantage: number }
export interface Group { problem: number[]; answers: GroupAnswer[]; silent: boolean }

export function pCorrect(k: number, d: number) {
  const written = Math.min(k, d)
  return Math.pow(STEP_ACCURACY, written) * Math.pow(HEAD_ACCURACY, d - written)
}

export function makeProblem(rng: Rng): number[] {
  const n = 2 + rng.int(DIFFICULTIES.length)
  return Array.from({ length: n }, () => 1 + rng.int(9))
}

export class ReasoningToy {
  logits: number[][]
  lengthCost = 0
  groupSize = 8
  learningRate = 0.5
  constructor() {
    const start = Array.from({ length: MAX_THINKING + 1 }, (_, k) => -0.6 * k)
    this.logits = DIFFICULTIES.map(() => [...start])
  }
  policy(d: number) { return Num.softmax(this.logits[d - 1]) }
  get expectedAccuracy() {
    return DIFFICULTIES.reduce((s, d) => s + this.policy(d).reduce((a, p, k) => a + p * pCorrect(k, d), 0), 0) / DIFFICULTIES.length
  }
  get expectedLength() {
    return DIFFICULTIES.reduce((s, d) => s + this.policy(d).reduce((a, p, k) => a + p * k, 0), 0) / DIFFICULTIES.length
  }

  step(rng: Rng): Group {
    const problem = makeProblem(rng), d = problem.length - 1
    const pi = this.policy(d)
    const lengths: number[] = [], rewards: number[] = [], correct: boolean[] = []
    for (let i = 0; i < this.groupSize; i++) {
      const k = Num.sample(pi, rng)
      const ok = rng.uniform() < pCorrect(k, d)
      lengths.push(k); correct.push(ok); rewards.push((ok ? 1 : 0) - this.lengthCost * k)
    }
    const mean = rewards.reduce((a, b) => a + b, 0) / this.groupSize
    const sd = Math.sqrt(rewards.reduce((a, r) => a + (r - mean) ** 2, 0) / this.groupSize)
    const adv = rewards.map((r) => (sd < 1e-9 ? 0 : (r - mean) / sd))
    const grad = new Array(pi.length).fill(0)
    lengths.forEach((k, i) => { for (let j = 0; j < pi.length; j++) grad[j] += adv[i] * ((j === k ? 1 : 0) - pi[j]) })
    for (let j = 0; j < pi.length; j++) this.logits[d - 1][j] += (this.learningRate * grad[j]) / this.groupSize
    const answers = lengths.map((k, i) => ({ thinking: k, correct: correct[i], reward: rewards[i], advantage: adv[i] }))
    return { problem, answers, silent: new Set(rewards).size === 1 }
  }
}

/** The written-out working for `k` thinking steps (for display). */
export function working(problem: number[], k: number, rng: Rng) {
  let total = problem[0]
  const lines: string[] = []
  for (let i = 1; i < problem.length; i++) {
    if (i <= k) { const next = total + problem[i]; lines.push(`${total} + ${problem[i]} = ${next}`); total = next }
    else { total += problem[i]; if (rng.uniform() >= HEAD_ACCURACY) total += [-2, -1, 1, 2][rng.int(4)] }
  }
  const truth = problem.reduce((a, b) => a + b, 0)
  return { lines, answer: total, correct: total === truth }
}
