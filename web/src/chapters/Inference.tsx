import { useEffect, useState } from 'preact/hooks'
import { cacheSize, gpt3Shape, nucleus, sampleNucleus, speedup, tinyShape, tokensPerPass, withCache, withoutCache } from '../engine/inference'
import { TinyGPT, tokenizer } from '../engine/model'
import { Rng } from '../engine/tensor'
import { fmt } from '../engine/toys'
import { BarChart, Legend, LineChart } from '../ui/charts'
import { Btn, Card, GoDeeper, KeyIdea, ModelBanner, Note, P, Page, Segmented, Slider, Stat, TextInput, useTrainer, useWidth } from '../ui/kit'

const tok = tokenizer
const PURPLE = 'var(--c5)'
const GREY = 'var(--c8)'
const faded = (c: string, pct: number) => `color-mix(in srgb, ${c} ${pct}%, var(--panel))`

/** Run a generator in slices of at most about `budget` ms (always at least one unit of work),
 *  yielding to the page in between. Returns a cancel function. */
function runSliced(job: Generator<unknown, void, unknown>, budget = 12): () => void {
  let cancelled = false
  let timer: ReturnType<typeof setTimeout> | null = null
  const tick = () => {
    if (cancelled) return
    const t0 = performance.now()
    for (;;) {
      const s = performance.now()
      if (job.next().done) return
      const now = performance.now()
      if (now - t0 + (now - s) > budget) break   // another unit this size wouldn't fit
    }
    timer = setTimeout(tick, 0)
  }
  timer = setTimeout(tick, 0)
  return () => { cancelled = true; if (timer) clearTimeout(timer) }
}

export default function Inference() {
  return (
    <Page id="inference">
      <ModelBanner />
      <P>**Inference** is running the trained model to get text out. It's the same forward pass as in training, minus the backward pass and the update, done one token at a time. Training happens once; inference happens every time anyone uses the model, so most of a popular model's lifetime compute goes here.</P>
      <P>Three questions decide what you get and how fast: *which* token to pick from the probabilities, how to avoid redoing work for every new token, and how to produce more than one token per step.</P>

      <SamplingCard />
      <CacheCard />
      <SpeculativeCard />

      <KeyIdea>Inference picks from the model's probabilities, one token at a time. The tricks that make it practical, like caches and draft models, never change what the model computes, only how much work it takes.</KeyIdea>

      <GoDeeper title="what real serving systems add">
        <P>**Batching.** A GPU running one conversation is mostly waiting on memory. Servers process many conversations in the same forward pass, adding and removing them as they start and finish (*continuous batching*).</P>
        <P>**The two phases.** Reading the prompt (*prefill*) processes all its tokens at once and is limited by arithmetic; generating (*decode*) adds one token at a time and is limited by how fast weights and the cache can be read from memory. That's why long answers cost more per token than long prompts.</P>
        <P>**Smaller caches.** Grouped-query attention shares each key and value among several query heads, shrinking the cache several times over, and many models store weights and the cache in 8 or 4 bits (*quantisation*).</P>
        <P>**Where our numbers come from.** The work counts are exact multiply-adds from the model's shape, not timings; on a model this small, timings would mostly measure overhead. The speculative-decoding numbers come from the formula, not a real draft model.</P>
      </GoDeeper>
    </Page>
  )
}

// ---------------------------------------------------------------- sampling

function SamplingCard() {
  const t = useTrainer()
  const model = t.snapshot
  const [prompt, setPrompt] = useState('the old man ')
  const [temperature, setTemperature] = useState(0.8)
  const [topP, setTopP] = useState(0.9)
  const [seed, setSeed] = useState(1)

  const ids = tok.encode(tok.clean(prompt)).slice(-model.config.context)
  const probs = ids.length ? model.nextTokenProbs(ids, temperature) : []
  const kept = new Set(nucleus(probs, topP))
  const top = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a]).slice(0, 12)

  return (
    <Card title="Live: choosing the next token" subtitle="Temperature reshapes the probabilities; top-p then keeps only the likeliest tokens that together reach p">
      <div class="row">
        <div class="grow"><TextInput id="inference-prompt" value={prompt} onInput={setPrompt} placeholder="Prompt" /></div>
        <Slider id="inference-temperature" label={`Temperature ${temperature.toFixed(2)}`} value={temperature} min={0.1} max={2} onInput={setTemperature} />
        <Slider id="inference-top-p" label={`Top-p ${topP.toFixed(2)}`} value={topP} min={0.05} max={1} onInput={setTopP} />
      </div>
      {probs.length > 0 && (
        <>
          <BarChart height={170} yDomain={[0, 1]} yFormat={(v) => `${Math.round(v * 100)}%`} valueFormat={(v) => `${Math.round(v * 100)}%`}
            bars={top.map((i) => ({ label: tok.display(i), value: probs[i], color: kept.has(i) ? PURPLE : faded(GREY, 30) }))} />
          <Note>{`${kept.size} of ${probs.length} characters kept (purple). The rest can't be picked, however lucky the draw.`}</Note>
          <hr style={{ border: 0, borderTop: '1px solid var(--line)', margin: 0, width: '100%' }} />
          <Continuations model={model} version={t.snapshotVersion} prompt={tok.clean(prompt)} ids={ids}
            temperature={temperature} topP={topP} seed={seed} again={() => setSeed(seed + 3)} />
          <Note>Three continuations with these settings. Low temperature or low top-p: safe and repetitive, often the same three times. High: varied, then nonsense.</Note>
        </>
      )}
    </Card>
  )
}

/** Three sampled continuations. Generating them is ~150 forward passes, so it waits for the sliders
 *  to settle and runs in short time slices; the old text stays until the new is ready. */
function Continuations({ model, version, prompt, ids, temperature, topP, seed, again }: {
  model: TinyGPT; version: number; prompt: string; ids: number[]; temperature: number; topP: number; seed: number; again: () => void
}) {
  const [texts, setTexts] = useState<string[]>([])
  const key = `${version}|${ids.join(',')}|${temperature}|${topP}|${seed}`

  useEffect(() => {
    let cancel: (() => void) | null = null
    const wait = setTimeout(() => {
      function* job() {
        const out: string[] = []
        for (let r = 0; r < 3; r++) {
          const rng = new Rng(seed + r)
          const all = [...ids], gen: number[] = []
          for (let i = 0; i < 50; i++) {
            const probs = model.nextTokenProbs(all.slice(-model.config.context), temperature)
            const next = sampleNucleus(probs, topP, rng)
            gen.push(next); all.push(next)
            yield
          }
          out.push(prompt + tok.decode(gen).replaceAll('\n', ' ↵ '))
        }
        setTexts(out)
      }
      cancel = runSliced(job())
    }, 120)
    return () => { clearTimeout(wait); cancel?.() }
  }, [key])

  return (
    <div class="row top" style={{ flexWrap: 'nowrap' }}>
      <div class="grow" aria-live="polite" style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {texts.map((s, i) => <div key={i} class="mono" style={{ fontSize: '13px', wordBreak: 'break-word' }}>{s}</div>)}
        {texts.length === 0 && <span class="note">…</span>}
      </div>
      <Btn onClick={again}>Sample again</Btn>
    </div>
  )
}

// ---------------------------------------------------------------- KV cache

type Size = 'tiny' | 'gpt3'
const sizes: Size[] = ['tiny', 'gpt3']

function CacheCard() {
  const t = useTrainer()
  const [size, setSize] = useState<Size>('tiny')
  const big = size === 'gpt3'
  const shape = big ? gpt3Shape : tinyShape(t.config)
  const maxT = big ? 4096 : t.config.context
  const stepBy = Math.max(1, maxT / 64)
  const points: [number, number, number][] = []
  for (let x = 1; x < maxT; x += stepBy) points.push([x, withCache(shape, x), withoutCache(shape, x)])
  points.push([maxT, withCache(shape, maxT), withoutCache(shape, maxT)])   // end exactly at maxT, as the tiles say
  const last = points[points.length - 1]

  return (
    <Card title="The KV cache: don't redo old work" subtitle="Work needed to produce each new token, counted from the model's shape">
      <div><Segmented options={sizes} value={size} onChange={setSize} labels={(s) => (s === 'tiny' ? 'Your tiny model' : 'GPT-3 sized (175B)')} /></div>
      <LineChart height={200} xLabel="position of the new token" yLabel="multiply-adds" legend={false}
        xDomain={[1, maxT]} yDomain={[0, Math.max(...points.map((p) => p[2]))]} yFormat={fmt.big} xFormat={(v) => String(Math.round(v))}
        series={[
          { name: 'without a cache', color: GREY, points: points.map(([x, , wo]) => ({ x, y: wo })) },
          { name: 'with a cache', color: PURPLE, points: points.map(([x, w]) => ({ x, y: w })) },
        ]} />
      <Legend items={[
        { name: 'without a cache: recompute every earlier token', color: GREY },
        { name: 'with a cache: only the new token, plus attention over the stored keys', color: PURPLE },
      ]} />
      <div class="stats">
        <Stat label={`Token ${maxT}, without`} value={fmt.big(last[2])} tone={GREY} />
        <Stat label={`Token ${maxT}, with`} value={fmt.big(last[1])} tone={PURPLE} />
        <Stat label="Saving" value={`${(last[2] / last[1]).toFixed(0)}×`} tone="var(--good)" />
        <Stat label={`Cache after ${maxT} tokens`} value={`${fmt.big(cacheSize(shape, maxT))} numbers`} tone="var(--c1)" />
      </div>
      <Note>Keys and values of earlier tokens never change once computed (the causal mask means later tokens can't affect them), so they're stored and reused. The cost moves from computing to memory: the cache grows with every token, in every layer.</Note>
    </Card>
  )
}

// ---------------------------------------------------------------- speculative decoding

function SpeculativeCard() {
  const [acceptance, setAcceptance] = useState(0.7)
  const [draft, setDraft] = useState(4)
  const [draftCost, setDraftCost] = useState(0.05)
  const k = Math.round(draft)
  const perPass = tokensPerPass(acceptance, k)
  const speed = speedup(acceptance, k, draftCost)
  const curve = Array.from({ length: 8 }, (_, i) => ({ k: i + 1, s: speedup(acceptance, i + 1, draftCost) }))

  return (
    <Card title="Speculative decoding: guess ahead, check in one go" subtitle="A small draft model writes k tokens; the big model checks them all in one pass and keeps the ones it agrees with">
      <div class="cols">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', minWidth: 0 }}>
          <Slider id="inference-acceptance" label={`Draft accepted: ${Math.round(acceptance * 100)}%`} value={acceptance} min={0.1} max={0.95} onInput={setAcceptance} />
          <Slider id="inference-draft" label={`Draft length k: ${k}`} value={draft} min={1} max={8} step={1} onInput={setDraft} />
          <Slider id="inference-draft-cost" label={`Draft step cost: ${Math.round(draftCost * 100)}% of big`} value={draftCost} min={0.01} max={0.5} onInput={setDraftCost} />
          <div class="stats">
            <Stat label="Tokens per big pass" value={perPass.toFixed(2)} tone={PURPLE} />
            <Stat label="Expected speed-up" value={`${speed.toFixed(2)}×`} tone={speed >= 1 ? 'var(--good)' : 'var(--c4)'} />
          </div>
          <Note>The output is exactly what the big model alone would write: a rejected token is replaced by one sampled from the big model, so only speed changes. Formula from Leviathan, Kalman and Matias (2023).</Note>
        </div>
        <SpeedupChart curve={curve} selected={k} />
      </div>
    </Card>
  )
}

/** Bars of the expected speed-up for k = 1…8, with a dashed break-even line at 1×. */
function SpeedupChart({ curve, selected }: { curve: { k: number; s: number }[]; selected: number }) {
  const [ref, width] = useWidth<HTMLDivElement>(400)
  const H = 220, m = { l: 40, r: 8, t: 14, b: 40 }, iw = width - m.l - m.r, ih = H - m.t - m.b
  const top = Math.max(1.2, ...curve.map((c) => c.s)) * 1.08
  const sy = (v: number) => m.t + ih - (v / top) * ih
  const bw = iw / curve.length
  const step = top > 4 ? 1 : top > 2 ? 0.5 : 0.25
  const ticks: number[] = []
  for (let v = 0; v <= top; v += step) ticks.push(+v.toFixed(2))
  const label = `Expected speed-up by draft length k: ${curve.map((c) => `k ${c.k}: ${c.s.toFixed(2)}×`).join(', ')}. Break-even at 1×. Selected k = ${selected}.`
  return (
    <div ref={ref} style={{ minWidth: 0 }}>
      <svg class="chart" width={width} height={H} viewBox={`0 0 ${width} ${H}`} role="img" aria-label={label}>
        {ticks.map((v) => <g key={v}><line class="grid" x1={m.l} x2={width - m.r} y1={sy(v)} y2={sy(v)} /><text x={m.l - 6} y={sy(v) + 4} text-anchor="end">{`${v}×`}</text></g>)}
        {curve.map((c, i) => {
          const x = m.l + i * bw + bw * 0.15, w = bw * 0.7
          return (
            <g key={c.k}>
              <rect x={x} y={sy(c.s)} width={w} height={Math.max(1, sy(0) - sy(c.s))} rx="3" fill={c.k === selected ? PURPLE : faded(PURPLE, 30)} />
              <text x={x + w / 2} y={H - m.b + 15} text-anchor="middle" style={{ fill: 'var(--ink)' }}>{c.k}</text>
            </g>
          )
        })}
        <line x1={m.l} x2={width - m.r} y1={sy(1)} y2={sy(1)} stroke="var(--muted)" stroke-dasharray="4 3" />
        <text x={width - m.r} y={sy(1) - 4} text-anchor="end">break-even</text>
        <text x={m.l + iw / 2} y={H - 6} text-anchor="middle">draft length k</text>
        <text x={m.l} y={m.t - 2} text-anchor="start">speed-up</text>
      </svg>
    </div>
  )
}
