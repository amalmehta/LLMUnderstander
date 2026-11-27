// Supervised fine-tuning of the reader's own tiny model (same as ChatTuning.swift).
// The pretraining text has no questions; demonstrations teach "what does <subject> <verb>"
// → "<subject> <verb>s <object>." with the loss counted only on the answer.
import { Ops, Rng } from './tensor'
import { clipGradients, Corpus, Optimizer, TinyGPT, tokenizer } from './model'

export interface Demo { question: string; answer: string }
const pairs: [string, string][] = [
  ['see', 'sees'], ['like', 'likes'], ['find', 'finds'], ['eat', 'eats'], ['want', 'wants'],
  ['hold', 'holds'], ['chase', 'chases'], ['bring', 'brings'], ['love', 'loves'], ['make', 'makes'],
]
export const heldOutQuestions = ['what does the queen want', 'what does a child chase', 'what does the fox find', 'what does my friend bring']
export const demoText = (e: Demo) => `${e.question}\n${e.answer}\n`
export const SUGGESTED_STEPS = 150

export const demonstrations: Demo[] = (() => {
  const rng = new Rng(2024), seen = new Set<string>(), out: Demo[] = []
  while (out.length < 96) {
    const s = Corpus.subjects[rng.int(Corpus.subjects.length)]
    const [verb, third] = pairs[rng.int(pairs.length)]
    const o = Corpus.objects[rng.int(Corpus.objects.length)]
    const e = { question: `what does ${s} ${verb}`, answer: `${s} ${third} ${o}.` }
    if (!seen.has(e.question) && !heldOutQuestions.includes(e.question)) { seen.add(e.question); out.push(e) }
  }
  return out
})()

export function trainingRow(e: Demo, context: number) {
  const all = tokenizer.encode(demoText(e))
  const answerStart = tokenizer.encode(e.question + '\n').length
  const end = Math.min(all.length - 1, context)
  return {
    ids: all.slice(0, end),
    targets: all.slice(1, end + 1),
    mask: Array.from({ length: end }, (_, i) => (i + 1 >= answerStart ? 1 : 0)),
  }
}

const argmax = (p: number[]) => p.reduce((b, v, i) => (v > p[b] ? i : b), 0)

export function answer(model: TinyGPT, question: string, maxLength = 40): string {
  const ids = tokenizer.encode(tokenizer.clean(question) + '\n')
  const newline = tokenizer.encode('\n')[0]
  const out: number[] = []
  for (let i = 0; i < maxLength; i++) {
    const next = argmax(model.nextTokenProbs(ids.slice(-model.config.context)))
    if (next === newline) break
    out.push(next); ids.push(next)
  }
  return tokenizer.decode(out)
}

function parse(q: string) {
  if (!q.startsWith('what does ')) return null
  const words = q.slice('what does '.length).split(' ')
  if (words.length < 2) return null
  return { subject: words.slice(0, -1).join(' '), verb: words[words.length - 1] }
}
export function startsWithSubject(a: string, q: string) { const p = parse(q); return !!p && a.startsWith(p.subject + ' ') }
export function usesVerb(a: string, q: string) {
  const p = parse(q); const pair = p && pairs.find(([v]) => v === p.verb)
  return !!p && !!pair && a.startsWith(`${p.subject} ${pair[1]} `)
}

export class ChatTuner {
  model: TinyGPT
  step = 0
  learningRate = 3e-4
  private optimizer = new Optimizer('AdamW')
  private rng = new Rng(11)
  constructor(base: TinyGPT) { this.model = base.copy() }

  trainStep(): number {
    const params = this.model.parameters
    const rows = Array.from({ length: 8 }, () => trainingRow(demonstrations[this.rng.int(demonstrations.length)], this.model.config.context))
    const loss = Ops.mean(rows.map((r) => Ops.crossEntropy(this.model.forward(r.ids), r.targets, r.mask)))
    for (const p of params) p.zeroGrad()
    loss.backward()
    clipGradients(params, 1)
    this.optimizer.step(params, this.learningRate)
    this.step++
    return loss.item
  }

  demonstrationLoss(): number {
    const rows = demonstrations.map((e) => trainingRow(e, this.model.config.context))
    return rows.reduce((s, r) => s + Ops.crossEntropy(this.model.forward(r.ids), r.targets, r.mask).item, 0) / rows.length
  }
}
