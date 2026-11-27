// Exact operation counts and published formulas for the Inference chapter (same as InferenceMaths.swift).
import { Num, Rng } from './tensor'
import { GPTConfig } from './model'

export interface Shape { name: string; width: number; layers: number; vocab: number }
export const tinyShape = (c: GPTConfig): Shape => ({ name: 'Your tiny model', width: c.dModel, layers: c.nLayers, vocab: c.vocab })
/** GPT-3 175B's published shape (Brown et al., 2020). */
export const gpt3Shape: Shape = { name: 'GPT-3 sized (175B)', width: 12288, layers: 96, vocab: 50257 }

export const withCache = (s: Shape, t: number) => s.layers * (12 * s.width * s.width + 2 * t * s.width) + s.width * s.vocab
export const withoutCache = (s: Shape, t: number) => s.layers * (12 * s.width * s.width * t + t * (t + 1) * s.width) + s.width * s.vocab
export const cacheSize = (s: Shape, t: number) => 2 * s.layers * s.width * t

/** Speculative decoding (Leviathan, Kalman and Matias, 2023): expected tokens per big-model pass. */
export const tokensPerPass = (a: number, k: number) => (a >= 0.999 ? k + 1 : (1 - Math.pow(a, k + 1)) / (1 - a))
export const speedup = (a: number, k: number, c: number) => tokensPerPass(a, k) / (k * c + 1)

/** Top-p: the likeliest tokens until their probabilities reach p (always at least one). */
export function nucleus(probs: number[], p: number): number[] {
  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a])
  const kept: number[] = []
  let total = 0
  for (const i of order) { kept.push(i); total += probs[i]; if (total >= p) break }
  return kept
}

export function sampleNucleus(probs: number[], p: number, rng: Rng): number {
  const kept = nucleus(probs, p)
  const ps = kept.map((i) => probs[i]), sum = ps.reduce((a, b) => a + b, 0)
  return kept[Num.sample(ps.map((x) => x / sum), rng)]
}
