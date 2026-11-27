// The Tour: scroll to carry one sentence through the whole pipeline.
// Captions scroll on the left; a stage stays pinned on the right (on phones, at the top).
// The active step is the caption block covering the middle of the visible area, and
// `local` (0…1) is how far through that block the reader is, for things that draw as you scroll.
import { JSX } from 'preact'
import { useEffect, useMemo, useRef, useState } from 'preact/hooks'
import { chapterById, menuTitle, secVar } from '../chapters'
import { md, tokenColor, useTrainer } from '../ui/kit'
import { tokenizer } from '../engine/model'
import { fmt } from '../engine/toys'
import type { Point, Trainer } from '../engine/trainer'
import { progress } from '../progress'
import './Tour.css'

const SENTENCE = 'the cat sees the moon.'
/** The prediction steps ask what follows this prefix (the answer is "o"). */
const PREFIX = 'the cat sees the m'

interface TourStep { title: string; caption: string; chapter: string }

const STEPS: TourStep[] = [
  { title: 'It starts as text', caption: 'A language model only ever does one thing: read some text and guess what comes next. Follow this one sentence through the whole machine.', chapter: 'big-picture' },
  { title: 'Text becomes tokens', caption: "The model can't read letters. Each piece of text becomes a number from a fixed list. Our tiny model uses one token per character; real models use chunks of words.", chapter: 'tokens' },
  { title: 'Tokens become vectors', caption: "Each number looks up a row of a table: a list of numbers (a vector) the model can compute with. These are your model's real values.", chapter: 'embeddings' },
  { title: 'Tokens look back', caption: '**Attention:** each token compares itself with every earlier token and borrows information from the ones that match. Thicker lines are stronger matches, read from your model.', chapter: 'attention' },
  { title: 'Through the blocks', caption: 'Attention, then a small network on each token, each adding to a running total (the residual stream). Stack the block a few times (twice here, about a hundred times in big models).', chapter: 'block' },
  { title: 'A guess for every next token', caption: 'At the end, the model scores every possible next character. Softmax turns the scores into probabilities. After "the cat sees the m", what comes next?', chapter: 'next-token' },
  { title: 'Score the guess', caption: '**The loss** is how surprised the model was by the real next token: −log of the probability it gave it. Sure and right: near 0. Sure and wrong: large.', chapter: 'next-token' },
  { title: 'Blame flows backwards', caption: '**Backpropagation** uses the chain rule to work out, for every weight, which way to move it to lower the loss. One backward pass covers all of them.', chapter: 'backprop' },
  { title: 'Nudge every weight', caption: "**The optimizer** moves each weight a small step against its gradient. AdamW, the usual choice, scales each step by how noisy that weight's gradient has been.", chapter: 'optimizer' },
  { title: 'Repeat', caption: "Batch, predict, score, backpropagate, update. That's the training loop. Run it thousands of times and the loss falls. This is your model's own curve.", chapter: 'training-loop' },
  { title: 'Now do it at scale', caption: '**Pretraining** is the same loop on trillions of tokens across thousands of GPUs. The result is a base model: a superb autocomplete.', chapter: 'pretraining' },
  { title: 'Teach it to answer', caption: '**Supervised fine-tuning:** the same loop on example conversations, graded only on the answers. It learns the format of an assistant.', chapter: 'sft' },
  { title: 'Teach it what people prefer', caption: '**Preference tuning:** people pick the better of two answers; a reward model learns their taste; RL nudges the model toward higher reward, but not too far.', chapter: 'preferences' },
  { title: 'Reward right answers', caption: '**Reasoning models:** where a checker can mark answers right or wrong, RL rewards correct results, and the model learns to work things out step by step.', chapter: 'reasoning' },
  { title: 'Put it to work', caption: '**Agents:** put the model in a loop with tools. It writes a tool call, the harness runs it, the result goes back into the context, and the loop repeats until the task is done.', chapter: 'agent-loop' },
]
const N = STEPS.length

// ---------------------------------------------------------------- real numbers from the reader's model

interface TourData {
  ids: number[]
  prefixIDs: number[]
  embeddings: number[][] // one row per token (first 8 numbers)
  embedScale: number
  attention: number[] // last prefix token → each earlier one
  probs: number[]
  target: number
  curve: Point[]
  trainedSteps: number
  tokensSeen: number
  vocab: number
  width: number
  parameters: number
}

function tourData(t: Trainer): TourData {
  const model = t.snapshot
  const ids = tokenizer.encode(SENTENCE)
  const prefixIDs = tokenizer.encode(PREFIX)
  const embeddings = ids.map((id) => model.wte.row(id).slice(0, 8))
  const trace = model.trace(prefixIDs)
  const last = trace.attention.length - 1
  return {
    ids, prefixIDs, embeddings,
    embedScale: Math.max(1e-6, ...embeddings.flat().map(Math.abs)),
    attention: trace.attention[last][0][prefixIDs.length - 1],
    probs: model.nextTokenProbs(prefixIDs),
    target: tokenizer.encode('o')[0],
    curve: t.smoothed(),
    trainedSteps: t.step,
    tokensSeen: t.tokensSeen,
    vocab: tokenizer.size,
    width: model.config.dModel,
    parameters: model.parameterCount,
  }
}

// ---------------------------------------------------------------- helpers

type Style = JSX.CSSProperties & Record<string, string | number>

/** Props for a piece that springs in when its scene appears, staggered ~30 ms by index (fades only with reduced motion). */
const pop = (i = 0, cls = '', style: Style = {}) => ({ class: `pop${cls ? ' ' + cls : ''}`, style: { ...style, '--i': String(Math.min(i, 30)) } as Style })

/** Red for positive, blue for negative, the background colour at zero; works in dark mode. */
function diverging(v: number, scale: number) {
  const x = Math.max(-1, Math.min(1, v / (scale || 1)))
  return `color-mix(in srgb, ${x >= 0 ? 'var(--pos)' : 'var(--neg)'} ${Math.round(Math.abs(x) * 100)}%, var(--zero))`
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))
const pct = (p: number) => `${(p * 100).toFixed(0)}%`

/** Measures an element's content box. */
function useSize<T extends HTMLElement>(): [preact.RefObject<T>, { w: number; h: number }] {
  const ref = useRef<T>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, size]
}

function Chip({ text, sub, color, size = 20, i, cls = '' }: { text: string; sub?: string; color: string; size?: number; i?: number; cls?: string }) {
  const style: Style = { '--chip': color, '--size': `${size}px` }
  const props = i === undefined ? { class: cls, style } : pop(i, cls, style)
  return (
    <span {...props} class={`tour-chip ${props.class}`}>
      <span>{text}</span>
      {sub !== undefined && <small>{sub}</small>}
    </span>
  )
}

const SceneTitle = ({ children }: { children: string }) => <div class="ts-title">{children}</div>

// ---------------------------------------------------------------- scenes

interface SceneProps { d: TourData; local: number; compact: boolean }

function TextScene() {
  // One span per word so lines only break between words; every character pops in turn.
  let i = 0
  const words = SENTENCE.split(/(?<= )/)
  return (
    <div class="ts-center">
      <div class="ts-sentence">
        {words.map((w, wi) => <span key={wi} class="ts-word">{Array.from(w).map((c) => <span key={i} {...pop(i++)}>{c}</span>)}</span>)}
      </div>
    </div>
  )
}

function TokenScene({ d }: SceneProps) {
  return (
    <div class="ts-center">
      <SceneTitle>ONE TOKEN PER CHARACTER</SceneTitle>
      <div class="ts-flow">
        {d.ids.map((id, i) => <Chip key={i} text={tokenizer.display(id)} sub={String(id)} color={tokenColor(id)} i={i} />)}
      </div>
      <p {...pop(d.ids.length, 'ts-note')}>{d.ids.length} tokens, from a vocabulary of {d.vocab}</p>
    </div>
  )
}

function EmbeddingScene({ d }: SceneProps) {
  const shown = d.ids.slice(0, 12)
  return (
    <div class="ts-center">
      <SceneTitle>EACH TOKEN LOOKS UP A ROW OF NUMBERS</SceneTitle>
      <div class="ts-embed">
        {shown.map((id, i) => (
          <div key={i} {...pop(i, 'ts-embed-col')}>
            <Chip text={tokenizer.display(id)} color={tokenColor(id)} size={16} />
            <div class="ts-embed-cells">
              {d.embeddings[i].map((v, k) => <i key={k} {...pop(i + k, 'ts-cell', { background: diverging(v, d.embedScale) })} />)}
            </div>
          </div>
        ))}
      </div>
      <p {...pop(20, 'ts-note')}>First 8 of {d.width} numbers per token. Red is positive, blue negative. The same letter always gets the same row.</p>
    </div>
  )
}

function AttentionScene({ d, local }: SceneProps) {
  const [ref, { w, h }] = useSize<HTMLDivElement>()
  const n = d.prefixIDs.length
  const spacing = w / n
  const cy = h - 34 // chip centre
  const base = cy - 17 // where the arcs leave and land
  const draw = Math.min(1, local * 1.6 + 0.15)
  const chipW = Math.max(12, Math.min(28, spacing - 3))
  const font = Math.min(15, chipW * 0.62)
  const x = (i: number) => spacing * (i + 0.5)
  return (
    <div class="ts-col">
      <SceneTitle>"m" LOOKS BACK AT EVERY EARLIER TOKEN</SceneTitle>
      <div class="ts-fill" ref={ref}>
        {w > 0 && h > 60 && (
          <svg class="ts-svg" width={w} height={h}>
            {Array.from({ length: n - 1 }, (_, j) => {
              const wt = d.attention[j]
              const from = x(n - 1), to = x(j)
              const lift = Math.min(base * 0.92, Math.abs(from - to) * 0.55)
              return (
                <path key={j} d={`M${from},${base} Q${(from + to) / 2},${base - lift} ${to},${base}`} pathLength={1}
                  fill="none" stroke-linecap="round" stroke-dasharray="1 1" stroke-dashoffset={1 - draw}
                  style={{ stroke: 'var(--c2)', strokeOpacity: 0.25 + 0.75 * wt, strokeWidth: 1 + 14 * wt }} />
              )
            })}
            {d.prefixIDs.map((id, i) => {
              const color = i === n - 1 ? 'var(--c2)' : tokenColor(id)
              return (
                <g key={i} transform={`translate(${x(i)},${cy})`}>
                  <g {...pop(i)}>
                    <rect x={-chipW / 2} y={-13} width={chipW} height={26} rx={6}
                      style={{ fill: `color-mix(in srgb, ${color} 16%, var(--panel))`, stroke: `color-mix(in srgb, ${color} 40%, var(--line))` }} />
                    <text class="ts-svg-chip" y={font * 0.35} text-anchor="middle" style={{ fontSize: `${font}px` }}>{tokenizer.display(id)}</text>
                    {i < n - 1 && <text class="ts-svg-pct" y={27} text-anchor="middle" style={{ opacity: draw }}>{pct(d.attention[i])}</text>}
                  </g>
                </g>
              )
            })}
          </svg>
        )}
      </div>
      <p class="ts-note">Line thickness = attention weight, from the last block of your model. Keep scrolling to draw them.</p>
    </div>
  )
}

const BLOCK_PARTS = ['embeddings', 'attention', 'MLP', 'attention', 'MLP', 'scores for the next token']
const BLOCK_COLORS = ['var(--c1)', 'var(--c2)', 'var(--c6)', 'var(--c2)', 'var(--c6)', 'var(--c5)']

function BlockScene({ local, compact, backwards }: SceneProps & { backwards: boolean }) {
  const [ref, { w, h }] = useSize<HTMLDivElement>()
  const t = backwards ? 1 - local : local
  const count = BLOCK_PARTS.length
  const slot = h / count
  const lineX = w * (compact ? 0.22 : 0.3)
  const dotY = h - slot * 0.5 - (h - slot) * t
  const tone = backwards ? 'var(--c4)' : 'var(--c1)'
  return (
    <div class="ts-col">
      <SceneTitle>{backwards ? 'GRADIENTS FLOW BACK DOWN' : 'THE VECTORS FLOW UP THROUGH TWO BLOCKS'}</SceneTitle>
      <div class="ts-fill" ref={ref}>
        {h > 0 && (
          <>
            <div class="ts-block-line" style={{ left: `${lineX - 1.5}px`, top: `${slot / 2}px`, height: `${h - slot}px` }} />
            {BLOCK_PARTS.map((p, i) => (
              <div key={i} class="ts-block-slot" style={{ left: `${lineX + 26}px`, top: `${h - slot * (i + 0.5)}px` }}>
                <div {...pop(i, 'ts-block-row')}>
                  <span class="ts-block" style={{ '--tone': BLOCK_COLORS[i] } as Style}>{p}</span>
                  {backwards && i > 0 && i < count - 1 && (
                    <span class="ts-grad" style={{ opacity: t <= i / (count - 1) + 0.05 ? 1 : 0.15 }}>∂loss/∂W</span>
                  )}
                </div>
              </div>
            ))}
            <div class="ts-dot" style={{ '--tone': tone, left: `${lineX - 9}px`, transform: `translateY(${dotY - 9}px)` } as Style} />
            {!backwards && <div class="ts-running" style={{ right: `${w - lineX + 14}px`, top: `${h / 2}px`, maxWidth: `${Math.max(60, lineX - 18)}px` }}>+ added to the running total</div>}
          </>
        )}
      </div>
    </div>
  )
}

function PredictionScene({ d }: SceneProps) {
  const top = d.probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0]).slice(0, 6)
  return (
    <div class="ts-center">
      <div {...pop(0, 'ts-mono ts-big')}>{PREFIX}<b class="ts-q">?</b></div>
      <div class="ts-bars">
        {top.map(([p, i], r) => (
          <div key={i} {...pop(r + 1, 'ts-bar-row')}>
            <span class="ts-mono ts-bar-label">{tokenizer.display(i)}</span>
            <span class="ts-track"><i class="ts-bar" style={{ width: `${Math.max(1.2, p * 100)}%`, background: i === d.target ? 'var(--good)' : 'color-mix(in srgb, var(--c5) 55%, transparent)' }} /></span>
            <span class="ts-num">{pct(p)}</span>
          </div>
        ))}
      </div>
      <p {...pop(8, 'ts-note')}>{d.trainedSteps === 0 ? 'Untrained, so the guesses are nearly even. Train the model and come back.' : 'Green is the real next letter.'}</p>
    </div>
  )
}

function LossScene({ d }: SceneProps) {
  const p = d.probs[d.target]
  const loss = -Math.log(Math.max(p, 1e-12)), worst = Math.log(d.vocab)
  const bar = (label: string, v: number, color: string, i: number) => (
    <div {...pop(i, 'ts-bar-row ts-loss-row')}>
      <span class="ts-loss-label">{label}</span>
      <span class="ts-track"><i class="ts-bar" style={{ width: `${Math.max(1, (100 * v) / worst)}%`, background: `color-mix(in srgb, ${color} 60%, transparent)` }} /></span>
      <span class="ts-num">{v.toFixed(2)}</span>
    </div>
  )
  return (
    <div class="ts-center">
      <SceneTitle>HOW SURPRISED WAS IT?</SceneTitle>
      <div {...pop(0, 'ts-mono ts-formula')}>loss = −log p("o") = −log {p.toFixed(2)} = {loss.toFixed(2)}</div>
      <div class="ts-bars ts-bars-wide">
        {bar('this guess', loss, 'var(--c4)', 1)}
        {bar('guessing evenly', worst, 'var(--c8)', 2)}
        {bar('perfect', 0, 'var(--good)', 3)}
      </div>
      <p {...pop(4, 'ts-note')}>Training's only goal: make this number small, averaged over every token it sees.</p>
    </div>
  )
}

function UpdateScene({ d, local }: SceneProps) {
  const cols = 10, rows = 6
  return (
    <div class="ts-center">
      <SceneTitle>EVERY WEIGHT TAKES A SMALL STEP</SceneTitle>
      <div {...pop(0, 'ts-mono ts-formula')}>w ← w − learning rate × gradient</div>
      <div class="ts-weights">
        {Array.from({ length: rows * cols }, (_, k) => {
          const r = Math.floor(k / cols), c = k % cols
          const base = Math.sin(r * 7 + c * 3) * 0.8
          const grad = Math.cos(r * 5 + c * 11) * 0.6
          return <i key={k} {...pop(k, 'ts-weight', { background: diverging(base - grad * local, 1.2) })} />
        })}
      </div>
      <p {...pop(10, 'ts-note')}>Scroll to apply the step. Each square is one weight; your model has {d.parameters.toLocaleString()} of them, and every one moves at once.</p>
    </div>
  )
}

const LOOP_PHASES = ['batch', 'forward', 'loss', 'backward', 'update']

function LoopScene({ d, local, compact }: SceneProps) {
  const active = Math.floor(local * 5) % 5
  const R = compact ? 78 : 110
  const cw = compact ? 240 : 280, ch = compact ? 90 : 170
  const path = useMemo(() => {
    if (d.curve.length < 2) return ''
    const maxY = Math.max(0.1, ...d.curve.map((p) => p.y))
    const maxX = Math.max(1, d.curve[d.curve.length - 1].x)
    return d.curve.map((q, k) => `${k ? 'L' : 'M'}${((cw * q.x) / maxX).toFixed(1)},${(ch * (1 - q.y / maxY)).toFixed(1)}`).join('')
  }, [d.curve, cw, ch])
  const reveal = Math.min(1, local * 1.4 + 0.1)
  const trainingLoop = chapterById('training-loop')!
  return (
    <div class="ts-center ts-loop">
      <div class="ts-ring" style={{ '--d': `${R * 2}px`, '--box': `${R * 2 + (compact ? 60 : 80)}px` } as Style}>
        <div class="ts-ring-circle" style={{ borderColor: 'color-mix(in srgb, var(--c6) 35%, transparent)' }} />
        {LOOP_PHASES.map((p, i) => {
          const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5
          return (
            <div key={i} class="ts-orbit" style={{ transform: `translate(${(R * Math.cos(a)).toFixed(1)}px, ${(R * Math.sin(a)).toFixed(1)}px)` }}>
              <span {...pop(i, `ts-phase${i === active ? ' on' : ''}`)}>{p}</span>
            </div>
          )
        })}
        <b class="ts-ring-center" style={{ color: 'var(--c6)' }}>×{Math.max(d.trainedSteps, 1500).toLocaleString()}</b>
      </div>
      <div {...pop(6, 'ts-curve')}>
        <div class="ts-curve-label">Your model's loss</div>
        {path ? (
          <svg width={cw} height={ch} class="ts-curve-svg" aria-hidden="true">
            <defs><clipPath id="tour-curve-clip"><rect x={-4} y={-4} width={cw * reveal + 4} height={ch + 8} /></clipPath></defs>
            <path d={path} fill="none" clip-path="url(#tour-curve-clip)" style={{ stroke: 'var(--c2)', strokeWidth: 2.5, strokeLinejoin: 'round' }} />
          </svg>
        ) : (
          <p class="ts-note ts-curve-empty" style={{ width: `${cw}px`, height: `${ch}px` }}>Train the model in Chapter {trainingLoop.number} and your loss curve appears here.</p>
        )}
      </div>
    </div>
  )
}

const SCALE_CELLS = 12 + 260

function ScaleScene({ d, local }: SceneProps) {
  const count = 12 + Math.floor(local * 260)
  return (
    <div class="ts-center">
      <SceneTitle>THE SAME LOOP, A TRILLION TIMES BIGGER</SceneTitle>
      <div class="ts-scale">
        {Array.from({ length: SCALE_CELLS }, (_, i) => (
          <i key={i} class={`ts-scale-cell rv-fade${i < count ? ' on' : ''}`} style={{ '--o': String(0.25 + (0.5 * ((i * 37) % 10)) / 10) } as Style} />
        ))}
      </div>
      <div class="ts-stats">
        <div {...pop(1, 'ts-stat')}><small>Your model saw</small><b>{fmt.big(Math.max(d.tokensSeen, 384_000))} tokens</b></div>
        <div {...pop(2, 'ts-stat')}><small>A recent open 8B model</small><b>~15T tokens</b></div>
        <div {...pop(3, 'ts-stat')}><small>GPUs</small><b>thousands, for months</b></div>
      </div>
    </div>
  )
}

function SFTScene() {
  const prompt = 'what does the cat see ↵'.split(' ')
  const answer = SENTENCE.split(' ')
  return (
    <div class="ts-center ts-left">
      <SceneTitle>A DEMONSTRATION, GRADED ONLY ON THE ANSWER</SceneTitle>
      <div class="ts-words">{prompt.map((w, i) => <span key={i} {...pop(i, 'ts-mono ts-wordbox', { '--tone': 'var(--c8)' })}>{w}</span>)}</div>
      <div class="ts-words">{answer.map((w, i) => <span key={i} {...pop(i + 6, 'ts-mono ts-wordbox graded', { '--tone': 'var(--c4)' })}>{w}</span>)}</div>
      <p {...pop(12, 'ts-note')}>Grey: read, not graded. Pink: the part the loss counts. Same loop as before.</p>
    </div>
  )
}

const THUMB = 'M2 21h4V9H2v12Zm20-11a2 2 0 0 0-2-2h-6.3l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L13.17 1 6.59 7.59C6.22 7.95 6 8.45 6 9v10a2 2 0 0 0 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2Z'

function PreferenceScene({ local, compact }: SceneProps) {
  const chosen = local > 0.3
  const answer = (text: string, good: boolean, i: number) => (
    <div {...pop(i, `ts-answer${chosen && good ? ' picked' : ''}`)}>
      <span class="ts-mono">{text}</span>
      <svg class={`ts-thumb rv${chosen ? ' on' : ''}${good ? ' good' : ''}`} width="34" height="34" viewBox="0 0 24 24">
        <path d={THUMB} transform={good ? undefined : 'rotate(180 12 12)'} />
      </svg>
    </div>
  )
  return (
    <div class="ts-center">
      <div class="ts-answers">
        {answer('the cat sees the moon at night.', true, 0)}
        {answer('cat moon cat moon moon.', false, 1)}
      </div>
      <p class="ts-note ts-swap" key={chosen ? 'a' : 'b'}>
        {chosen ? `Reward model: +2.1 vs −1.4 → push the model toward the ${compact ? 'top' : 'left'} answer` : 'Scroll: a person picks the better answer'}
      </p>
    </div>
  )
}

const REASON_LINES = ['7 + 5 = 12', '12 + 9 = 21', '21 + 3 = 24']

function ReasoningScene({ local }: SceneProps) {
  const shown = Math.min(REASON_LINES.length, Math.floor(local * 4.5))
  return (
    <div class="ts-center ts-left ts-reason">
      <div {...pop(0, 'ts-mono ts-big')}>Q: 7 + 5 + 9 + 3 = ?</div>
      {REASON_LINES.map((l, i) => <div key={i} class={`ts-mono ts-reason-line rv${i < shown ? ' on' : ''}`}>{'  ' + l}</div>)}
      <div class={`ts-reason-answer rv${shown === REASON_LINES.length ? ' on' : ''}`}>
        <b class="ts-mono ts-big">A: 24</b>
        <span class="ts-check">
          <svg width="18" height="18" viewBox="0 0 24 24"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm-1.2 14.2-4-4 1.4-1.4 2.6 2.6 5.6-5.6 1.4 1.4-7 7Z" /></svg>
          checker: correct, reward 1
        </span>
      </div>
      <p class="ts-note">Scroll to let it think. Writing the steps out is what earned reward.</p>
    </div>
  )
}

const AGENT_STAGES = [
  { name: 'model', what: 'writes a tool call', tone: 'var(--c5)' },
  { name: 'harness', what: 'checks and runs it', tone: 'var(--c3)' },
  { name: 'tool', what: 'run_tests → 1 failed', tone: 'var(--c2)' },
  { name: 'context', what: 'result goes back in', tone: 'var(--c1)' },
]

function AgentScene({ local, compact }: SceneProps) {
  const active = Math.floor(local * 8) % AGENT_STAGES.length
  const rx = compact ? 104 : 150, ry = compact ? 112 : 130
  const d = compact ? 200 : 260
  return (
    <div class="ts-center">
      <div class="ts-ring ts-agent" style={{ '--d': `${d}px`, '--box': `${compact ? 330 : 440}px`, '--boxh': `${compact ? 330 : 380}px` } as Style}>
        <div class="ts-ring-circle" style={{ borderColor: 'color-mix(in srgb, var(--c3) 30%, transparent)' }} />
        {AGENT_STAGES.map((s, i) => {
          const a = -Math.PI / 2 + (i * 2 * Math.PI) / AGENT_STAGES.length
          return (
            <div key={i} class="ts-orbit" style={{ transform: `translate(${(rx * Math.cos(a)).toFixed(1)}px, ${(ry * Math.sin(a)).toFixed(1)}px)` }}>
              <div {...pop(i)}>
                <div class={`ts-agent-stage${i === active ? ' on' : ''}`} style={{ '--tone': s.tone } as Style}>
                  <b>{s.name}</b><small>{s.what}</small>
                </div>
              </div>
            </div>
          )
        })}
        <div class="ts-ring-center ts-note">until<br />the task<br />is done</div>
      </div>
    </div>
  )
}

function Scene({ step, ...p }: SceneProps & { step: number }) {
  switch (step) {
    case 0: return <TextScene />
    case 1: return <TokenScene {...p} />
    case 2: return <EmbeddingScene {...p} />
    case 3: return <AttentionScene {...p} />
    case 4: return <BlockScene {...p} backwards={false} />
    case 5: return <PredictionScene {...p} />
    case 6: return <LossScene {...p} />
    case 7: return <BlockScene {...p} backwards />
    case 8: return <UpdateScene {...p} />
    case 9: return <LoopScene {...p} />
    case 10: return <ScaleScene {...p} />
    case 11: return <SFTScene />
    case 12: return <PreferenceScene {...p} />
    case 13: return <ReasoningScene {...p} />
    default: return <AgentScene {...p} />
  }
}

// ---------------------------------------------------------------- page

interface Leaving { step: number; key: number; local: number }
interface ViewState { pos: number; key: number; leaving: Leaving | null }

function Captions({ active }: { active: number }) {
  return (
    <>
      {STEPS.map((s, i) => {
        const c = chapterById(s.chapter)!
        return (
          <section key={i} class={`tour-step${i === active ? ' active' : ''}`} aria-labelledby={`tour-step-${i}`} style={{ '--sec': secVar(c.section) }}>
            <div class="tour-step-inner">
              <div class="tour-count">{i + 1} of {N}</div>
              <h2 id={`tour-step-${i}`}>{s.title}</h2>
              <p>{md(s.caption)}</p>
              <a href={`#${c.id}`}>Open {menuTitle(c)} →</a>
            </div>
          </section>
        )
      })}
    </>
  )
}

export default function Tour() {
  const t = useTrainer()
  const d = useMemo(() => tourData(t), [t.snapshotVersion])
  const stepsRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const [view, setView] = useState<ViewState>({ pos: 0, key: 0, leaving: null })
  const [compact, setCompact] = useState(false)

  // Title, and counts as read after 20 seconds here.
  useEffect(() => {
    document.title = 'Tour · LLM Understander'
    const id = setTimeout(() => progress.markRead('tour'), 20_000)
    return () => clearTimeout(id)
  }, [])

  // Scroll → position. Reads two rects per frame at most and writes nothing to the DOM itself.
  useEffect(() => {
    const steps = stepsRef.current, stage = stageRef.current
    if (!steps || !stage) return
    const stacked = matchMedia('(max-width: 860px)')
    let stepH = 1, frame = 0, last = -1
    const update = () => {
      frame = 0
      const vh = window.innerHeight
      const top = steps.getBoundingClientRect().top
      let anchor = vh / 2
      if (stacked.matches) { const b = stage.getBoundingClientRect().bottom; anchor = b + (vh - b) / 2 }
      const p = clamp((anchor - top) / stepH, 0, N - 0.001)
      if (Math.floor(p) === Math.floor(last) && Math.abs(p - last) < 0.002) return
      last = p
      setView((v) => {
        const from = Math.floor(v.pos), to = Math.floor(p)
        if (from === to) return { ...v, pos: p }
        return { pos: p, key: v.key + 1, leaving: { step: from, key: v.key, local: to > from ? 1 : 0 } }
      })
    }
    const schedule = () => { if (!frame) frame = requestAnimationFrame(update) }
    const ro = new ResizeObserver(() => {
      stepH = Math.max(1, steps.offsetHeight / N)
      setCompact(stage.clientWidth < 560)
      last = -1
      schedule()
    })
    ro.observe(steps)
    ro.observe(stage)
    window.addEventListener('scroll', schedule, { passive: true })
    window.addEventListener('resize', schedule, { passive: true })
    return () => {
      ro.disconnect()
      window.removeEventListener('scroll', schedule)
      window.removeEventListener('resize', schedule)
      if (frame) cancelAnimationFrame(frame)
    }
  }, [])

  // The outgoing scene fades for 250 ms, then goes.
  useEffect(() => {
    if (!view.leaving) return
    const k = view.key
    const id = setTimeout(() => setView((v) => (v.key === k ? { ...v, leaving: null } : v)), 260)
    return () => clearTimeout(id)
  }, [view.key])

  const step = Math.floor(view.pos)
  const local = view.pos - step
  const captions = useMemo(() => <Captions active={step} />, [step])
  const tour = chapterById('tour')!
  const fill = Math.min(1, (view.pos + 0.5) / N)

  return (
    <article class="page tour" style={{ '--sec': secVar('tour') }}>
      <div class="tour-body">
        <header class="tour-head">
          <div class="eyebrow">Tour</div>
          <h1>One sentence, all the way through</h1>
          <p class="tagline">{tour.tagline}</p>
          <p class="tour-hint"><span aria-hidden="true">↓</span> Scroll down</p>
        </header>

        <div class="tour-captions">
          <div class="tour-steps" ref={stepsRef}>{captions}</div>
        </div>

        <div class="tour-end">
          <h2>That's the whole chain.</h2>
          <p>Every piece has a chapter where you can take it apart and play with it, starting with the model you train yourself.</p>
          <div class="row">
            <a class="btn primary" href="#big-picture">Start with Chapter 1</a>
            <a class="btn" href="#training-loop">Train the model</a>
          </div>
        </div>

        <div class="tour-stage-wrap">
          <div class="tour-stage" ref={stageRef} role="region" aria-label="Tour stage">
            <p class="tour-sr" aria-live="polite" key="sr">Step {step + 1} of {N}: {STEPS[step].title}</p>
            {view.leaving && view.leaving.step !== step && (
              <div class="tour-scene leaving" key={`s${view.leaving.key}`} aria-hidden="true">
                <Scene step={view.leaving.step} d={d} local={view.leaving.local} compact={compact} />
              </div>
            )}
            <div class="tour-scene" key={`s${view.key}`} aria-hidden="true">
              <Scene step={step} d={d} local={local} compact={compact} />
            </div>
          </div>
        </div>
      </div>

      <div class="tour-progress" role="progressbar" aria-label="Tour progress" aria-valuemin={1} aria-valuemax={N} aria-valuenow={step + 1} aria-valuetext={`step ${step + 1} of ${N}`}>
        <div class="tour-progress-fill" style={{ transform: `scaleX(${fill.toFixed(4)})` }} />
      </div>
    </article>
  )
}
