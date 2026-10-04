import { useMemo, useState } from 'preact/hooks'
import { tokenizer } from '../engine/model'
import { fmt, paramEstimate } from '../engine/toys'
import { BarChart } from '../ui/charts'
import { Btn, Card, Formula, GoDeeper, KeyIdea, md, Note, P, Page, Slider, Stat, useTrainer } from '../ui/kit'

type Part = 'input' | 'ln1' | 'attention' | 'add1' | 'ln2' | 'mlp' | 'add2' | 'head'
const parts: Part[] = ['input', 'ln1', 'attention', 'add1', 'ln2', 'mlp', 'add2', 'head']

const info: Record<Part, { title: string; shape: string; color: string; explanation: string }> = {
  input: { title: 'Input vectors', shape: 'T × d', color: 'var(--c8)', explanation: 'One vector per token (embedding + position). This running vector is called the **residual stream** — every layer reads from it and writes back into it.' },
  ln1: { title: 'LayerNorm', shape: 'T × d → T × d', color: 'var(--muted)', explanation: 'Rescales each token\'s vector to mean 0 and spread 1 (then applies a learned gain and shift). It keeps numbers in a healthy range so deep stacks train stably.' },
  attention: { title: 'Multi-head attention', shape: 'T × d → T × d', color: 'var(--c1)', explanation: 'Tokens exchange information (Chapter 4). This is the *only* place where one position can see another.' },
  add1: { title: 'Add (residual)', shape: 'T × d + T × d', color: 'var(--c3)', explanation: 'The layer\'s output is **added** to its input instead of replacing it. Information can flow straight through, and during training gradients can flow straight back. Without these shortcuts, deep networks barely train.' },
  ln2: { title: 'LayerNorm', shape: 'T × d → T × d', color: 'var(--muted)', explanation: 'Rescales each token\'s vector to mean 0 and spread 1 (then applies a learned gain and shift). It keeps numbers in a healthy range so deep stacks train stably.' },
  mlp: { title: 'MLP (feed-forward)', shape: 'T × d → T × 4d → T × d', color: 'var(--c2)', explanation: 'A small two-layer network applied to each token separately: widen to 4×, apply GELU, shrink back. Attention *moves* information between tokens; the MLP *processes* it. Most of a model\'s stored knowledge is thought to live here.' },
  add2: { title: 'Add (residual)', shape: 'T × d + T × d', color: 'var(--c3)', explanation: 'The layer\'s output is **added** to its input instead of replacing it. Information can flow straight through, and during training gradients can flow straight back. Without these shortcuts, deep networks barely train.' },
  head: { title: 'Final LayerNorm + output head', shape: 'T × d → T × vocab', color: 'var(--c5)', explanation: 'After the last block, one matrix turns each vector into a score (a *logit*) for every token in the vocabulary. Softmax turns scores into probabilities (Chapter 6).' },
}

const SENTENCE = 'the cat sees the moon.'

export default function Block() {
  const t = useTrainer()
  const model = t.snapshot
  const [part, setPart] = useState<Part>('attention')
  const [layers, setLayers] = useState(12)
  const [width, setWidth] = useState(768)
  const [vocab, setVocab] = useState(50257)

  const stages = useMemo(() => {
    const trace = model.trace(tokenizer.encode(SENTENCE))
    const rms = (r: number[]) => Math.sqrt(r.reduce((s, v) => s + v * v, 0) / r.length)
    const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
    return [{ label: 'embedding', value: avg(trace.embeddings.map(rms)) },
      ...trace.residualRMS.map((r, i) => ({ label: `after block ${i + 1}`, value: avg(r) }))]
  }, [model])

  const p = paramEstimate(layers, width, vocab)
  const preset = (name: string, l: number, d: number, v: number) => (
    <Btn key={name} onClick={() => { setLayers(l); setWidth(d); setVocab(v) }}>{name}</Btn>
  )
  const cur = info[part]

  return (
    <Page id="block">
      <P>A transformer is a stack of identical **blocks**. Each block does two things: let tokens talk to each other (**attention**), then let each token think on its own (**MLP**). Around both are **shortcuts** that add the result back onto the input.</P>

      <Card title="Inside one block" subtitle="Click a part">
        <div class="row top" style={{ gap: '28px' }}>
          <BlockDiagram part={part} onPick={setPart} />
          <div class="grow" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <h3 style={{ margin: 0, font: '600 20px var(--font-display)' }}>{cur.title}</h3>
            <span class="mono" style={{ color: 'var(--muted)' }}>Shape: {cur.shape}</span>
            <p style={{ margin: 0, fontSize: '15px', lineHeight: 1.6 }}>{md(cur.explanation)}</p>
            {(part === 'add1' || part === 'add2' || part === 'input') && <Formula>{'x = x + Attention(LayerNorm(x))\nx = x + MLP(LayerNorm(x))'}</Formula>}
            {part === 'mlp' && <Formula>{'MLP(x) = GELU(x·W₁ + b₁)·W₂ + b₂'}</Formula>}
          </div>
        </div>
      </Card>

      <Card title="The residual stream in our tiny model" subtitle={`Average size of each token's vector after each block, for the sentence "${SENTENCE}"`}>
        <BarChart height={160} bars={stages.map((s) => ({ ...s, color: 'var(--c3)' }))} />
        <Note>Each block adds its contribution on top of what was already there, so the stream accumulates information as it flows up through the layers.</Note>
      </Card>

      <Card title="How many weights?" subtitle="Standard GPT layout: about 12·d² per block, plus the embedding table">
        <div class="row" style={{ gap: '8px' }}>
          {preset('Our tiny model', model.config.nLayers, model.config.dModel, model.config.vocab)}
          {preset('GPT-2 small', 12, 768, 50257)}
          {preset('GPT-2 XL', 48, 1600, 50257)}
          {preset('GPT-3', 96, 12288, 50257)}
          {preset('8B-class', 32, 4096, 128256)}
        </div>
        <Slider id="block-layers" label={`Blocks (N): ${layers}`} value={layers} min={1} max={128} step={1} onInput={setLayers} />
        <Slider id="block-width" label={`Width (d): ${width}`} value={width} min={16} max={16384} step={1} onInput={setWidth} />
        <Slider id="block-vocab" label={`Vocabulary: ${vocab}`} value={vocab} min={20} max={256000} step={1} onInput={setVocab} />
        <div class="stats">
          <Stat label="Parameters" value={fmt.big(p)} tone="var(--c2)" />
          <Stat label="Memory at 2 bytes each" value={fmt.big(p * 2) + 'B'} tone="var(--c1)" />
          <Stat label="Share in blocks" value={`${((100 * 12 * layers * width * width) / p).toFixed(0)}%`} tone="var(--c3)" />
        </div>
        <Note>{`Our model's exact count is ${model.parameterCount.toLocaleString()} (it also has biases and LayerNorm weights, and a separate output head).`}</Note>
      </Card>

      <KeyIdea>A transformer = embed → (attention + MLP, with shortcuts) × N → scores for the next token. Scaling up mostly means a wider stream (d) and more blocks (N); the recipe stays the same.</KeyIdea>

      <GoDeeper title="LayerNorm and modern tweaks">
        <Formula>{'LayerNorm(x) = γ · (x − mean(x)) / √(var(x) + ε) + β'}</Formula>
        <P>**Pre-norm** (normalise *before* each sub-layer, as here and in GPT-2 onward) trains far more stably than the original 2017 post-norm design.</P>
        <P>Modern models often use **RMSNorm** (skip the mean subtraction), **SwiGLU** (a gated MLP instead of GELU) and **grouped-query attention** (several query heads share one key/value head, to shrink the KV cache). Same skeleton, better parts.</P>
      </GoDeeper>
    </Page>
  )
}

// Fixed geometry so the residual shortcuts can be drawn exactly.
const boxW = 240, arrowH = 18, dividerH = 26, gutter = 34
const boxH = (p: Part) => (p === 'attention' || p === 'mlp' ? 46 : 30)
const layout = (() => {
  let y = 0
  const tops = {} as Record<Part, number>
  for (const p of parts) {
    if (p === 'head') y += dividerH
    tops[p] = y
    y += boxH(p)
    if (p !== 'head') y += arrowH
  }
  return { tops, height: y }
})()

function BlockDiagram({ part, onPick }: { part: Part; onPick: (p: Part) => void }) {
  const { tops, height } = layout
  const x0 = gutter, W = boxW + x0, H = height + 2
  const shortcuts = ([['input', 'add1'], ['add1', 'add2']] as [Part, Part][]).map(([from, to]) => {
    const startY = tops[from] + boxH(from) + arrowH / 2, endY = tops[to] + boxH(to) / 2
    return `M${x0 + boxW / 2} ${startY} L10 ${startY} L10 ${endY} L${x0 - 2} ${endY} M${x0 - 8} ${endY - 5} L${x0 - 2} ${endY} L${x0 - 8} ${endY + 5}`
  })
  const divY = tops.head - dividerH / 2
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block', flex: 'none', maxWidth: '100%', height: 'auto', fontFamily: 'var(--font-display)' }} role="group" aria-label="The parts of one transformer block">
      <path d={shortcuts.join(' ')} fill="none" stroke="var(--c3)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="5 3" />
      {parts.map((p) => {
        const on = p === part, c = info[p].color, y = tops[p], h = boxH(p)
        const isAdd = p === 'add1' || p === 'add2'
        return (
          <g key={p}>
            <g role="button" tabIndex={0} aria-pressed={on} aria-label={info[p].title} style={{ cursor: 'pointer' }}
              onClick={() => onPick(p)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onPick(p) } }}>
              <rect x={x0 + 0.5} y={y + 0.5} width={boxW - 1} height={h - 1} rx="8"
                fill={`color-mix(in srgb, ${c} ${on ? 35 : 13}%, var(--panel))`}
                stroke={`color-mix(in srgb, ${c} ${on ? 100 : 30}%, var(--panel))`} stroke-width={on ? 2 : 1} />
              <text x={x0 + boxW / 2} y={y + h / 2 + 4.5} text-anchor="middle" style={{ fill: 'var(--ink)', fontSize: '13px', fontWeight: 600 }}>
                {isAdd ? '⊕ ' : ''}{info[p].title}
              </text>
            </g>
            {p !== 'head' && <text x={x0 + boxW / 2} y={y + h + arrowH / 2 + 4} text-anchor="middle" style={{ fill: 'var(--muted)', fontSize: '11px' }}>↓</text>}
          </g>
        )
      })}
      <line x1="0" x2={W * 0.27} y1={divY} y2={divY} stroke="var(--muted)" />
      <text x={W / 2} y={divY + 4} text-anchor="middle" style={{ fill: 'var(--muted)', fontSize: '11px' }}>× N blocks, then:</text>
      <line x1={W * 0.73} x2={W} y1={divY} y2={divY} stroke="var(--muted)" />
    </svg>
  )
}
