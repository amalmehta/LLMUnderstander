import { ComponentChildren } from 'preact'
import { useMemo, useState } from 'preact/hooks'
import { tokenizer } from '../engine/model'
import { Tensor } from '../engine/tensor'
import { Heatmap, Legend } from '../ui/charts'
import { Card, Chip, Formula, GoDeeper, KeyIdea, ModelBanner, Note, P, Page, TextInput, tokenColor, useTrainer, useWidth } from '../ui/kit'

interface EmbPoint { id: number; x: number; y: number; label: string; kind: Kind }
type Kind = 'vowel' | 'consonant' | 'space/punctuation'
const kindColor: Record<Kind, string> = { vowel: 'var(--c4)', consonant: 'var(--c1)', 'space/punctuation': 'var(--c8)' }

/** Principal-component projection of each row to 2-D (power iteration with deflation). */
function project(table: Tensor): EmbPoint[] {
  const n = table.rows, d = table.cols
  let rows = Array.from({ length: n }, (_, i) => table.row(i))
  const mean = Array.from({ length: d }, (_, j) => rows.reduce((s, r) => s + r[j], 0) / n)
  rows = rows.map((r) => r.map((v, j) => v - mean[j]))
  const cov = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  for (const r of rows) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i][j] += r[i] * r[j]
  const top = (m: number[][]) => {
    let v = Array.from({ length: d }, (_, i) => (i % 3) - 1 + 0.1)
    for (let it = 0; it < 200; it++) {
      const w = m.map((row) => row.reduce((s, x, j) => s + x * v[j], 0))
      const norm = Math.max(Math.sqrt(w.reduce((s, x) => s + x * x, 0)), 1e-12)
      v = w.map((x) => x / norm)
    }
    return v
  }
  const v1 = top(cov)
  const lambda = v1.reduce((acc, vi, i) => acc + vi * cov[i].reduce((s, x, j) => s + x * v1[j], 0), 0)
  const cov2 = cov.map((row, i) => row.map((x, j) => x - lambda * v1[i] * v1[j]))
  const v2 = top(cov2)
  const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0)
  return rows.map((r, i) => {
    const c = tokenizer.vocab[i]
    const kind: Kind = 'aeiou'.includes(c) ? 'vowel' : /\p{L}/u.test(c) ? 'consonant' : 'space/punctuation'
    return { id: i, x: dot(r, v1), y: dot(r, v2), label: tokenizer.display(i), kind }
  })
}

function neighbours(id: number, table: Tensor): [number, number][] {
  const a = table.row(id)
  const cos = (x: number[], y: number[]) => {
    const dot = x.reduce((s, v, i) => s + v * y[i], 0)
    return dot / Math.max(1e-12, Math.sqrt(x.reduce((s, v) => s + v * v, 0) * y.reduce((s, v) => s + v * v, 0)))
  }
  return Array.from({ length: table.rows }, (_, i) => i).filter((i) => i !== id)
    .map((i) => [i, cos(a, table.row(i))] as [number, number]).sort((p, q) => q[1] - p[1])
}

const Labelled = ({ title, children }: { title: string; children: ComponentChildren }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 'none' }}>
    <span style={{ font: '600 12px var(--font-display)', color: 'var(--muted)' }}>{title}</span>
    {children}
  </div>
)
const Op = ({ children }: { children: string }) => <span style={{ fontSize: '26px', paddingTop: '30px', flex: 'none', fontFamily: 'var(--font-display)' }}>{children}</span>

export default function Embeddings() {
  const t = useTrainer()
  const model = t.snapshot
  const [text, setText] = useState('the cat')
  const [picked, setPicked] = useState<number | null>(null)
  const ids = tokenizer.encode(tokenizer.clean(text)).slice(0, model.config.context)
  const d = model.config.dModel

  return (
    <Page id="embeddings">
      <ModelBanner />
      <P>{`A token ID like 19 is just a name — the number itself means nothing. So the first layer of the model is a **lookup table**: one row of numbers per token. Token 19 → row 19. That row is the token's **embedding**, a list of ${d} numbers here (thousands in real models).`}</P>
      <P>Nobody writes these numbers. They start random and are **learned** by training, like every other weight. Tokens that behave similarly end up with similar rows.</P>

      <Card title="Lookup: token embedding + position embedding" subtitle={`Each row is one token; each column one of the ${d} dimensions. Red = positive, blue = negative.`}>
        <div style={{ maxWidth: '300px' }}>
          <TextInput id="embeddings-text" value={text} onInput={setText} placeholder="Text" />
        </div>
        {ids.length === 0 ? <Note>Type some lowercase letters.</Note> : (() => {
          const labels = ids.map((i) => tokenizer.display(i))
          const tokE = ids.map((i) => model.wte.row(i))
          const posE = ids.map((_, p) => model.wpe.row(p))
          const sum = tokE.map((r, i) => r.map((v, j) => v + posE[i][j]))
          return (
            <div class="scroll-x">
              <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start', width: 'max-content' }}>
                <Labelled title="what token is it?"><Heatmap values={tokE} rowLabels={labels} diverging cell={11} /></Labelled>
                <Op>+</Op>
                <Labelled title="where is it?"><Heatmap values={posE} rowLabels={ids.map((_, p) => String(p))} diverging cell={11} /></Labelled>
                <Op>=</Op>
                <Labelled title="what the transformer receives"><Heatmap values={sum} rowLabels={labels} diverging cell={11} /></Labelled>
              </div>
            </div>
          )
        })()}
        <Note>Attention (next chapter) has no built-in sense of order, so a **position embedding** is added. Without it, "dog bites man" and "man bites dog" would look identical.</Note>
      </Card>

      <Card title="The embedding map" subtitle={`All ${tokenizer.size} token embeddings squashed from ${d} dimensions to 2 (PCA). Click a token to find its nearest neighbours.`}>
        <div class="row top" style={{ gap: '20px' }}>
          <div class="grow">
            <EmbeddingMap wte={model.wte} picked={picked} onPick={setPicked} />
            <Legend items={(Object.keys(kindColor) as Kind[]).map((k) => ({ name: k, color: kindColor[k] }))} />
          </div>
          <div style={{ flex: '0 1 210px', minWidth: '0', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {picked !== null ? (
              <>
                <b style={{ fontFamily: 'var(--font-display)' }}>Closest to {tokenizer.display(picked)}</b>
                {neighbours(picked, model.wte).slice(0, 6).map(([id, sim]) => (
                  <div key={id} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Chip text={tokenizer.display(id)} color={tokenColor(id)} />
                    <div style={{ width: '90px', height: '6px', borderRadius: '3px', background: 'var(--line)', overflow: 'hidden' }}>
                      <div style={{ width: `${Math.max(0, sim) * 100}%`, height: '100%', background: 'var(--accent)' }} />
                    </div>
                    <span style={{ fontSize: '12px', fontVariantNumeric: 'tabular-nums', fontFamily: 'var(--font-display)' }}>{sim.toFixed(2)}</span>
                  </div>
                ))}
                <span style={{ fontSize: '11px', color: 'var(--muted)', fontFamily: 'var(--font-display)' }}>cosine similarity of the full {d}-number rows</span>
              </>
            ) : <span class="note">Click a dot</span>}
          </div>
        </div>
        <Note>{t.isTrained
          ? 'After training, letters that play similar roles tend to drift together — vowels often cluster, and space and full stop sit apart.'
          : 'Untrained: the dots are randomly scattered. Train the model (Chapter 9) and come back — the map reorganises itself.'}</Note>
      </Card>

      <KeyIdea>An embedding turns a meaningless ID into a point in space. Training moves the points so that their positions carry meaning the rest of the network can use.</KeyIdea>

      <GoDeeper title="sizes and modern variants">
        <P>The table has **vocabulary × width** numbers. GPT-2 small: 50,257 × 768 ≈ 39M weights — about a third of the whole model. GPT-3: 12,288 dimensions per token.</P>
        <P>Many models **tie** the input table to the output layer (the same matrix turns vectors back into token scores at the end).</P>
        <P>Modern models mostly replace learned position embeddings with **RoPE** (rotary position embeddings), which rotates the query and key vectors by an angle proportional to position. Relative distance then appears directly in the attention scores.</P>
        <Formula>{'x₀[t] = TokenEmbedding[id_t] + PositionEmbedding[t]      shape: (T × d)'}</Formula>
      </GoDeeper>
    </Page>
  )
}

function EmbeddingMap({ wte, picked, onPick }: { wte: Tensor; picked: number | null; onPick: (id: number) => void }) {
  const pts = useMemo(() => project(wte), [wte])
  const [ref, width] = useWidth<HTMLDivElement>(400)
  const W = width, H = 340, pad = 18
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y)
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
  const sx = (x: number) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * pad)
  const sy = (y: number) => H - pad - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * pad)
  // Draw the picked dot last so it sits on top.
  const order = picked === null ? pts : [...pts.filter((p) => p.id !== picked), pts[picked]]
  return (
    <div ref={ref}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', fontFamily: 'var(--font-mono)', cursor: 'pointer' }} role="img"
        onClick={(e) => {
          // Pick the dot nearest the click, like the Mac app.
          const r = (e.currentTarget as SVGElement).getBoundingClientRect(), cx = e.clientX - r.left, cy = e.clientY - r.top
          const best = pts.reduce((b, p) => (Math.hypot(sx(p.x) - cx, sy(p.y) - cy) < Math.hypot(sx(b.x) - cx, sy(b.y) - cy) ? p : b), pts[0])
          if (best) onPick(best.id)
        }} aria-label="Token embeddings projected to two dimensions">
        {order.map((p) => {
          const on = p.id === picked
          return (
            <g key={p.id}>
              <circle cx={sx(p.x)} cy={sy(p.y)} r={on ? 15 : 11} fill={kindColor[p.kind]} stroke={on ? 'var(--ink)' : 'none'} stroke-width="2" />
              <text x={sx(p.x)} y={sy(p.y) + 4} text-anchor="middle" style={{ fill: 'var(--panel)', fontSize: '12px', fontWeight: 700, pointerEvents: 'none' }}>{p.label}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
