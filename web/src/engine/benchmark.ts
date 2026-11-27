// A fixed 40-item word-completion benchmark, scored per character and by exact match.
// Same items and scoring as Benchmark.swift (same seed, same generator).
import { Rng } from './tensor'
import { Corpus, TinyGPT, tokenizer } from './model'

export interface BenchItem { prompt: string; answer: string }
export interface BenchScore { exact: number; perCharacter: number }

export const benchmarkItems: BenchItem[] = (() => {
  const rng = new Rng(4040)
  const out: BenchItem[] = [], seen = new Set<string>()
  while (out.length < 40) {
    const s = Corpus.sentence(rng)
    const words = s.slice(0, -1).split(' ')
    if (words.length < 3) continue
    const w = 1 + rng.int(words.length - 1)
    const word = words[w]
    if (word.length < 4) continue
    const cut = 1 + rng.int(2)
    const prompt = words.slice(0, w).join(' ') + ' ' + word.slice(0, cut)
    if (!seen.has(prompt)) { seen.add(prompt); out.push({ prompt, answer: word.slice(cut) }) }
  }
  return out
})()

const argmax = (p: number[]) => p.reduce((b, v, i) => (v > p[b] ? i : b), 0)

/** Greedy completion of the word (stops at a space, full stop or newline). */
export function completeWord(model: TinyGPT, item: BenchItem): string {
  const ids = tokenizer.encode(item.prompt)
  let out = ''
  for (let i = 0; i < item.answer.length + 2; i++) {
    const next = argmax(model.nextTokenProbs(ids.slice(-model.config.context)))
    const c = tokenizer.vocab[next]
    if (c === ' ' || c === '.' || c === '\n') break
    out += c; ids.push(next)
  }
  return out
}

export function scoreBenchmark(model: TinyGPT): BenchScore {
  let exact = 0, chars = 0, total = 0
  for (const item of benchmarkItems) {
    if (completeWord(model, item) === item.answer) exact++
    const ids = tokenizer.encode(item.prompt)
    for (const c of item.answer) {
      const best = argmax(model.nextTokenProbs(ids.slice(-model.config.context)))
      const truth = tokenizer.encode(c)[0]
      if (best === truth) chars++
      total++
      ids.push(truth)
    }
  }
  return { exact: exact / benchmarkItems.length, perCharacter: chars / Math.max(total, 1) }
}

/** Standard error of a score p on n independent items: √(p(1 − p) / n). */
export const standardError = (p: number, n: number) => Math.sqrt((p * (1 - p)) / Math.max(n, 1))
