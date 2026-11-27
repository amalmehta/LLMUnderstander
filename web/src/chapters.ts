export type SectionId = 'tour' | 'start' | 'blocks' | 'learning' | 'pretraining' | 'posttraining' | 'wrap' | 'agents'

export const sections: { id: SectionId; title: string }[] = [
  { id: 'tour', title: 'Tour' },
  { id: 'start', title: 'Start Here' },
  { id: 'blocks', title: 'Building Blocks' },
  { id: 'learning', title: 'How It Learns' },
  { id: 'pretraining', title: 'Pretraining' },
  { id: 'posttraining', title: 'Post-training' },
  { id: 'wrap', title: 'Putting It Together' },
  { id: 'agents', title: 'Agents & Harnesses' },
]

export interface ChapterInfo { id: string; number: number; title: string; tagline: string; section: SectionId }

const list: Omit<ChapterInfo, 'number'>[] = [
  { id: 'tour', title: 'Tour', tagline: 'Follow one sentence through the whole machine. Scroll to move it along.', section: 'tour' },
  { id: 'big-picture', title: 'The Big Picture', tagline: 'An LLM is one simple loop, run trillions of times, in three stages.', section: 'start' },
  { id: 'tokens', title: 'Text into Tokens', tagline: "Models can't read letters — text becomes a list of numbers first.", section: 'blocks' },
  { id: 'embeddings', title: 'Embeddings', tagline: 'Each token number becomes a vector the model can do math with.', section: 'blocks' },
  { id: 'attention', title: 'Attention', tagline: 'Each token looks back at earlier tokens and borrows what it needs.', section: 'blocks' },
  { id: 'block', title: 'The Transformer Block', tagline: 'Attention + a small network + shortcuts, stacked many times.', section: 'blocks' },
  { id: 'next-token', title: 'Predicting the Next Token', tagline: 'The whole job: guess the next token. The loss scores the guess.', section: 'learning' },
  { id: 'backprop', title: 'Backpropagation', tagline: 'The chain rule tells every weight which way to move.', section: 'learning' },
  { id: 'optimizer', title: 'The Optimizer', tagline: 'How big a step to take, and in which direction, for every weight.', section: 'learning' },
  { id: 'training-loop', title: 'The Training Loop', tagline: 'Batch → forward → loss → backward → update. Watch a real model learn.', section: 'learning' },
  { id: 'pretraining', title: 'Pretraining at Scale', tagline: 'Same loop, but on trillions of tokens across thousands of GPUs.', section: 'pretraining' },
  { id: 'sft', title: 'Supervised Fine-Tuning', tagline: 'Teach the text predictor to act like an assistant, using examples.', section: 'posttraining' },
  { id: 'preferences', title: 'Learning from Preferences', tagline: 'Teach it what people prefer: reward models, RLHF and DPO.', section: 'posttraining' },
  { id: 'reasoning', title: 'Reasoning Models', tagline: 'Reward right answers, not nice-sounding ones, and the model learns to think longer.', section: 'posttraining' },
  { id: 'evals', title: 'Evals', tagline: 'How do you know training worked? Scores, the noise in them, and the traps.', section: 'wrap' },
  { id: 'pipeline', title: 'The Full Pipeline', tagline: 'Every stage on one page: what changes, and what stays the same.', section: 'wrap' },
  { id: 'inference', title: 'Inference', tagline: 'Using the trained model: sampling, the KV cache, and getting tokens out faster.', section: 'wrap' },
  { id: 'agent-loop', title: 'From Model to Agent', tagline: 'A model only writes text. Put it in a loop with tools and it can act.', section: 'agents' },
  { id: 'harness', title: 'The Harness', tagline: 'Everything around the model: prompt, tools, permissions, memory, context.', section: 'agents' },
  { id: 'tool-use', title: 'Tool Use, Step by Step', tagline: 'Schema in, JSON out, result back in — the anatomy of one tool call.', section: 'agents' },
  { id: 'agent-training', title: 'Training Models to Be Agents', tagline: 'Teach tool use with transcripts, then sharpen it with RL on real tasks.', section: 'agents' },
  { id: 'workflows', title: 'Workflows & Bandits', tagline: "Wiring model calls into patterns — and choosing between options when you can't be sure which is best.", section: 'agents' },
]

/** The Tour is page 0, so every chapter's number is its position in the list. */
export const chapters: ChapterInfo[] = list.map((c, i) => ({ ...c, number: i }))
/** Everything but the Tour. */
export const numbered = chapters.filter((c) => c.id !== 'tour')
export const menuTitle = (c: ChapterInfo) => (c.id === 'tour' ? c.title : `${c.number}. ${c.title}`)
export const chapterById = (id: string) => chapters.find((c) => c.id === id)
export const secVar = (s: SectionId) => `var(--sec-${s})`
