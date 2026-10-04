import { useMemo, useState } from 'preact/hooks'
import { Corpus, TRAINING_TEXT } from '../engine/model'
import { Rng } from '../engine/tensor'
import { fmt, ScalingLaw } from '../engine/toys'
import { LineChart } from '../ui/charts'
import { Btn, Card, KeyIdea, P, Page, Segmented, Slider, Stat, tokenColor, useTrainer } from '../ui/kit'

const tabs = [0, 1, 2, 3, 4] as const
const tabLabels = ['1 · The data', '2 · Packing batches', '3 · Scaling laws', '4 · Thousands of GPUs', '5 · Tiny vs real']

export default function Pretraining() {
  const [tab, setTab] = useState<number>(0)
  return (
    <Page id="pretraining">
      <P>Pretraining is the loop from Chapter 9, unchanged, scaled up by a factor of about a trillion. The model reads a large slice of the written internet and learns to predict the next token. That's where nearly all of its knowledge, grammar and reasoning ability comes from — and nearly all of the cost.</P>

      <Segmented options={tabs} value={tab as (typeof tabs)[number]} onChange={setTab} labels={(i) => tabLabels[i]} />

      {tab === 0 ? <DataTab /> : tab === 1 ? <PackingTab /> : tab === 2 ? <ScalingTab /> : tab === 3 ? <ParallelTab /> : <ComparisonTab />}

      <KeyIdea>Pretraining = next-token prediction on trillions of tokens. The result is a **base model**: a superb autocomplete that knows a lot, but doesn't yet know it's supposed to be an assistant. Ask it a question and it may just continue with more questions.</KeyIdea>
    </Page>
  )
}

// ---------------------------------------------------------------- 1 · data

const stages: [string, string, number, string][] = [
  ['Crawl', 'Billions of web pages (e.g. Common Crawl), plus books, code, papers', 1.0, 'var(--c8)'],
  ['Extract text', 'Strip HTML, menus, ads, boilerplate', 0.7, 'var(--c1)'],
  ['Language & quality filters', "Keep fluent, informative text; classifiers score 'educational value'", 0.35, 'var(--c6)'],
  ['Deduplicate', 'Near-identical pages removed; repeats make models memorise', 0.2, 'var(--c3)'],
  ['Safety & privacy', 'Remove personal data and toxic content', 0.18, 'var(--c2)'],
  ['Mix', 'Blend web + code + math + books + multilingual in chosen proportions', 0.18, 'var(--c4)'],
  ['Tokenize & shuffle', 'Trillions of tokens, e.g. ~15T for recent open 8–400B models', 0.16, 'var(--c5)'],
]

function DataTab() {
  return (
    <Card title="From the raw web to training tokens" subtitle="A typical pipeline. Widths are illustrative of how much each step keeps.">
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontFamily: 'var(--font-display)' }}>
        {stages.map(([name, detail, frac, tone]) => (
          <div key={name} style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 12px', alignItems: 'center' }}>
            <b style={{ flex: '0 0 190px', fontSize: '14.5px' }}>{name}</b>
            <div style={{ flex: '1 1 160px', minWidth: 0 }}>
              <div style={{ width: `${frac * 100}%`, height: '22px', borderRadius: '6px', background: `color-mix(in srgb, ${tone} 55%, var(--panel))` }} />
            </div>
            <small style={{ flex: '1 1 240px', color: 'var(--muted)', fontSize: '12.5px', lineHeight: 1.35 }}>{detail}</small>
          </div>
        ))}
      </div>
      <P>{`Data quality matters as much as quantity: filtering and deduplication often improve a model more than adding raw tokens. Our tiny model's "internet" is ${TRAINING_TEXT.length.toLocaleString()} characters of generated sentences.`}</P>
    </Card>
  )
}

// ---------------------------------------------------------------- 2 · packing

interface Segment { doc: number; text: string; length: number }

function pack(context: number): Segment[][] {
  const rng = new Rng(5)
  const stream: [number, string][] = []
  for (let d = 0; d < 10; d++) {
    const n = 1 + rng.int(3)
    const text = Array.from({ length: n }, () => Corpus.sentence(rng)).join(' ')
    for (const c of text) stream.push([d, c])
    stream.push([-1, '·'])
  }
  const rows: Segment[][] = []
  for (let i = 0; i < stream.length && rows.length < 6; i += context) {
    const segs: Segment[] = []
    let cur: [number, string] | null = null
    for (const [d, c] of stream.slice(i, i + context)) {
      if (cur && cur[0] === d) cur[1] += c
      else {
        if (cur) segs.push({ doc: cur[0], text: cur[1], length: cur[1].length })
        cur = [d, c]
      }
    }
    if (cur) segs.push({ doc: cur[0], text: cur[1], length: cur[1].length })
    rows.push(segs)
  }
  return rows
}

function PackingTab() {
  const [context, setContext] = useState(48)
  const rows = useMemo(() => pack(context), [context])
  return (
    <Card title="Packing documents into fixed-length rows" subtitle="GPUs want rectangular batches: (batch size × context length)">
      <Slider id="pretraining-context" label={`Context length ${context} tokens`} value={context} min={16} max={96} step={8} onInput={setContext} />
      <div class="scroll-x">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', minWidth: 'max-content' }}>
          {rows.map((row, r) => (
            <div key={r} style={{ display: 'flex' }}>
              <span style={{ width: '44px', flex: 'none', fontSize: '11px', color: 'var(--muted)', fontFamily: 'var(--font-display)', alignSelf: 'center' }}>row {r}</span>
              {row.map((seg, k) => (
                <span key={k} style={{
                  width: `${seg.length * 9.5}px`, height: '20px', flex: 'none', overflow: 'hidden', whiteSpace: 'pre',
                  font: '9px/20px var(--font-mono)', textAlign: 'center',
                  background: seg.doc < 0 ? 'var(--ink)' : `color-mix(in srgb, ${tokenColor(seg.doc)} 45%, var(--panel))`,
                  color: seg.doc < 0 ? 'var(--panel)' : 'var(--ink)',
                }}>{seg.doc < 0 ? 'eot' : seg.text}</span>
              ))}
            </div>
          ))}
        </div>
      </div>
      <P>Documents are joined end to end with an **end-of-text** token between them, then cut into rows. One row can hold the end of one document and the start of the next. The end-of-text token teaches the model that what follows is unrelated. A batch of 1,024 rows × 8,192 tokens is ~8 million tokens per step.</P>
    </Card>
  )
}

// ---------------------------------------------------------------- 3 · scaling laws

function ScalingTab() {
  const [logC, setLogC] = useState(23)
  const c = 10 ** logC
  const opt = ScalingLaw.optimum(c)
  const curves = [logC - 2, logC, logC + 2].map((lc) => ({ lc, pts: ScalingLaw.sweep(10 ** lc).filter((p) => p[1] <= 4.5).map(([x, y]) => ({ x, y })) }))
  return (
    <Card title="Scaling laws: how big a model, for how much compute?" subtitle="Predicted final loss = E + A/N^α + B/D^β (Chinchilla form; constants from the 2024 Epoch AI replication)">
      <Slider id="pretraining-compute" label={`Compute budget ${fmt.sci(c)} FLOPs`} value={logC} min={18} max={26} onInput={setLogC} />
      <LineChart height={260} xDomain={[6, 13]} yDomain={[1.8, 4.5]} xTicks={[6, 7, 8, 9, 10, 11, 12, 13]} xFormat={(v) => fmt.big(10 ** v)}
        xLabel="model size (parameters)" yLabel="predicted loss"
        series={[
          ...curves.map(({ lc, pts }) => lc === logC
            ? { name: `budget ${fmt.sci(10 ** lc)} FLOPs`, color: 'var(--c2)', points: pts, width: 3 }
            : { name: `budget ${fmt.sci(10 ** lc)} FLOPs`, color: 'var(--c8)', points: pts, width: 1.5, opacity: 0.35 }),
          { name: 'best', color: 'var(--bad)', points: [{ x: Math.log10(opt.params), y: opt.loss }], dots: true, width: 0 },
        ]} />
      <div class="stats">
        <Stat label="Best model size" value={fmt.big(opt.params)} tone="var(--c2)" />
        <Stat label="Train it on" value={fmt.big(opt.tokens) + ' tokens'} tone="var(--c1)" />
        <Stat label="Tokens per parameter" value={(opt.tokens / opt.params).toFixed(0)} tone="var(--c3)" />
        <Stat label="Predicted loss" value={opt.loss.toFixed(2)} tone="var(--bad)" />
        <Stat label="≈ GPU-days (H100 @ 40%)" value={fmt.big(c / (0.4 * 9.9e14) / 86400)} tone="var(--c8)" />
      </div>
      <P>For a fixed budget, a model that's **too big** sees too little data, and one that's **too small** can't absorb what it sees. The sweet spot is about **20 tokens per parameter**. In practice, labs often train smaller models on far *more* tokens than this, because a smaller model is cheaper to serve to millions of users. Training compute ≈ **6 × parameters × tokens**.</P>
    </Card>
  )
}

// ---------------------------------------------------------------- 4 · parallelism

const dpTexts = [
  '① Each GPU holds a full copy of the weights and gets a different slice of the batch.',
  '② Each runs forward + backward on its own slice → its own gradient (different on each GPU).',
  '③ All-reduce: the GPUs average their gradients over fast links. Now every GPU has the same averaged gradient.',
  '④ Every GPU applies the same optimizer step, so all copies stay identical. Repeat.',
]

function GradientBars({ g, color }: { g: number[]; color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: '3px', height: '50px' }}>
      {g.map((v, i) => (
        <div key={i} style={{ width: '9px', height: `${Math.max(2, Math.abs(v) * 40)}px`, borderRadius: '2px', background: `color-mix(in srgb, ${color} 70%, transparent)`, transition: 'height 0.35s ease, background 0.35s ease' }} />
      ))}
    </div>
  )
}

function ParallelTab() {
  const [workers, setWorkers] = useState(4)
  const [dpStage, setDpStage] = useState(0)
  const grads = Array.from({ length: workers }, (_, w) => { const r = new Rng(w + 1); return Array.from({ length: 4 }, () => r.normal() * 0.5 + 0.3) })
  const avg = [0, 1, 2, 3].map((j) => grads.reduce((s, g) => s + g[j], 0) / workers)
  return (
    <Card title="Data parallelism: many GPUs, one model" subtitle={`Click through the steps of one training step on ${workers} GPUs`}>
      <Slider id="pretraining-gpus" label={`GPUs: ${workers}`} value={workers} min={2} max={8} step={1} onInput={setWorkers} />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
        {grads.map((g, w) => (
          <div key={w} style={{ flex: '1 1 76px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px', padding: '8px', borderRadius: '10px', background: 'color-mix(in srgb, var(--c2) 6%, var(--panel))', border: '1px solid var(--line)', fontFamily: 'var(--font-display)', textAlign: 'center' }}>
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="var(--c2)" stroke-width="1.8" aria-hidden="true">
              <rect x="6" y="6" width="12" height="12" rx="2" /><rect x="9.5" y="9.5" width="5" height="5" rx="0.5" />
              <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" />
            </svg>
            <b style={{ fontSize: '12.5px' }}>GPU {w}</b>
            <small style={{ fontSize: '9.5px', color: 'var(--muted)' }}>full copy of model</small>
            <span style={{ fontSize: '10.5px', padding: '2px 6px', borderRadius: '99px', background: `color-mix(in srgb, ${tokenColor(w)} 30%, var(--panel))` }}>batch slice {w}</span>
            <GradientBars g={dpStage >= 1 ? (dpStage >= 2 ? avg : g) : [0, 0, 0, 0]} color={dpStage >= 2 ? 'var(--c3)' : 'var(--c7)'} />
          </div>
        ))}
      </div>
      <p style={{ margin: 0, minHeight: '2.4em', fontSize: '15px' }}>{dpTexts[dpStage]}</p>
      <div class="row">
        <Btn primary onClick={() => setDpStage((dpStage + 1) % dpTexts.length)}>Next step</Btn>
        <small style={{ color: 'var(--muted)', fontFamily: 'var(--font-display)' }}>{dpStage + 1} / {dpTexts.length}</small>
      </div>
      <hr style={{ border: 0, borderTop: '1px solid var(--line)', margin: 0, width: '100%' }} />
      <P>When the model itself is too large for one GPU, it gets split up too: **tensor parallelism** cuts each matrix across GPUs; **pipeline parallelism** puts different blocks on different GPUs; **FSDP/ZeRO** shards weights, gradients and optimizer state and gathers them just in time. Big runs combine all of these across tens of thousands of GPUs, train in **bf16** (16-bit) arithmetic to save memory and time, and save checkpoints often, because hardware fails mid-run.</P>
    </Card>
  )
}

// ---------------------------------------------------------------- 5 · comparison

function ComparisonTab() {
  const t = useTrainer()
  const ours = t.parameterCount, ourTokens = t.settings.totalSteps * t.settings.batchSize * t.config.context
  const rows: [string, number, number, string][] = [
    ['Our tiny model', ours, ourTokens, 'your browser, ~1 minute'],
    ['GPT-2 (2019)', 1.5e9, 1e10, 'token count and hardware estimated'],
    ['GPT-3 (2020)', 175e9, 300e9, 'thousands of GPUs, weeks'],
    ['Llama 3 405B (2024)', 405e9, 15.6e12, '~16,000 H100 GPUs, months'],
  ]
  return (
    <Card title="Same loop, different universe" subtitle="Published or widely reported figures; compute ≈ 6·N·D">
      <div class="scroll-x">
        <table class="grid-table">
          <thead><tr><th>Model</th><th>Parameters</th><th>Training tokens</th><th>Compute (FLOPs)</th><th>Hardware</th></tr></thead>
          <tbody>
            {rows.map(([name, n, d, hw]) => {
              const isOurs = name.startsWith('Our')
              return (
                <tr key={name}>
                  <td style={isOurs ? { fontWeight: 700, color: 'var(--c6)' } : undefined}>{name}</td>
                  <td class="num">{fmt.big(n)}</td>
                  <td class="num">{fmt.big(d)}</td>
                  <td class="num">{fmt.sci(6 * n * d)}</td>
                  <td style={{ color: 'var(--muted)' }}>{hw}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <P>From our model to a frontier model is roughly **10¹⁴× more compute**. The algorithm is the one you watched in Chapter 9.</P>
    </Card>
  )
}
