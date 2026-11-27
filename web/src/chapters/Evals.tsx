import { useEffect, useState } from 'preact/hooks'
import { BenchItem, benchmarkItems, standardError } from '../engine/benchmark'
import { TinyGPT, tokenizer } from '../engine/model'
import { Legend, LineChart } from '../ui/charts'
import { Card, Formula, GoDeeper, KeyIdea, ModelBanner, Note, P, Page, Slider, Stat, useTrainer, useWidth } from '../ui/kit'

const TEAL = 'var(--c6)', PINK = 'var(--c4)', BLUE = 'var(--c1)'
const pct = (v: number) => `${Math.round(v * 100)}%`

export default function Evals() {
  return (
    <Page id="evals">
      <P>Training lowers the loss, but nobody ships a loss. You want to know: *can it do the job?* An **eval** (evaluation) is a fixed set of tasks with a way to score each answer. Benchmarks are evals many people share, so models can be compared.</P>
      <P>Two things make evals harder than they look: the **scoring rule** changes the story the numbers tell, and a score from a few dozen questions is **noisier** than it looks.</P>

      <ModelBanner />
      <CurveCard />
      <BenchmarkItems />
      <NoiseCard />

      <KeyIdea>A score is a measurement, with a rule and an error bar. Before trusting "it suddenly learned X" or "model A beats B", ask how it was scored and how many questions it rests on.</KeyIdea>

      <GoDeeper title="the traps">
        <P>**"Emergence" can be the metric.** Schaeffer, Miranda and Koyejo (2023) showed that many abilities that seem to appear suddenly in bigger models turn smooth when scored per token instead of all-or-nothing. The curve above is the same effect in miniature. Some jumps may still be real; the point is to check the metric first.</P>
        <P>**Contamination.** If test questions leak into the training data (they're on the web, after all), the model can score well by remembering. Labs search training data for benchmark text, and new benchmarks hide their answers or are written after the model's training cut-off. Our benchmark comes from sentences generated with a different seed than the training text, so it's held out, though the tiny grammar means similar sentences do appear in training.</P>
        <P>**Goodhart's law.** "When a measure becomes a target, it ceases to be a good measure." Tune on a benchmark long enough and you'll improve the benchmark more than the skill. Keep a test set you never tune on.</P>
        <P>**LLM-as-judge.** For open-ended answers, another model often does the grading. Judges have known biases, such as preferring longer answers, the first of two options, or their own style; good setups swap answer order and check the judge against people.</P>
        <Formula>{'standard error = √( p · (1 − p) / n )\n95% interval ≈ p ± 1.96 · standard error'}</Formula>
        <P>**p** = the score as a fraction, **n** = number of independent questions. To compare two models on the same questions, a *paired* test (which questions each got right) is more sensitive than comparing two intervals.</P>
      </GoDeeper>
    </Page>
  )
}

// ---------------------------------------------------------------- two scores

function CurveCard() {
  const t = useTrainer()
  const n = benchmarkItems.length
  return (
    <Card title="Two scores, one model" subtitle={`Your model's score on ${n} word endings it never trained on, measured every 100 training steps`}>
      {t.benchmark.length < 2 ? (
        <p class="note" style={{ minHeight: '120px', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
          Train the model (Chapter 9, or the button above) and the two curves fill in as it learns.
        </p>
      ) : (
        <>
          <LineChart height={220} xDomain={[0, t.settings.totalSteps]} yDomain={[0, 1]} xLabel="training step" yFormat={pct} legend={false}
            series={[
              { name: 'per character', color: TEAL, points: t.benchmarkPerChar, dots: true },
              { name: 'exact match', color: PINK, points: t.benchmark, dots: true },
            ]} />
          <Legend items={[
            { name: 'per character: share of letters right, each given the right letters before it', color: TEAL },
            { name: 'exact match: the whole word right, or nothing', color: PINK },
          ]} />
          <Note>{`Same answers, two rules. Per-character accuracy climbs steadily from the first hundred steps; exact match stays near zero, then shoots up once most letters are right, because a word needs every letter. Late in training exact match also wobbles by several points between checks: with ${n} items, one word is ${(100 / n).toFixed(1)}%.`}</Note>
        </>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------- items

const SHOWN = benchmarkItems.slice(0, 8)
const argmax = (p: number[]) => p.reduce((b, v, i) => (v > p[b] ? i : b), 0)

/** The same greedy word completion as `completeWord`, one token per `yield`, so it can be spread across frames. */
function* completions(model: TinyGPT, items: BenchItem[], out: string[]): Generator<void> {
  for (const item of items) {
    const ids = tokenizer.encode(item.prompt)
    let guess = ''
    for (let i = 0; i < item.answer.length + 2; i++) {
      const next = argmax(model.nextTokenProbs(ids.slice(-model.config.context)))
      yield
      const c = tokenizer.vocab[next]
      if (c === ' ' || c === '.' || c === '\n') break
      guess += c; ids.push(next)
    }
    out.push(guess)
  }
}

let cached: { version: number; guesses: string[] } | null = null

/**
 * The model's answers to the shown items, recomputed only when the snapshot changes (not on slider drags).
 * Each token is a full forward pass (several ms), so the work runs in slices of about 8 ms between frames.
 */
function useGuesses(): (string | undefined)[] {
  const t = useTrainer()
  const version = t.snapshotVersion
  const [guesses, setGuesses] = useState<string[]>(cached?.version === version ? cached.guesses : [])
  useEffect(() => {
    if (cached?.version === version) { setGuesses(cached.guesses); return }
    const out: string[] = [], job = completions(t.snapshot, SHOWN, out)
    let shown = 0, id = 0, cancelled = false
    setGuesses([])
    const slice = () => {
      if (cancelled) return
      const start = performance.now()
      let done = false
      while (performance.now() - start < 8) if (job.next().done) { done = true; break }
      if (out.length !== shown) { shown = out.length; setGuesses([...out]) }
      if (done) cached = { version, guesses: [...out] }
      else id = window.setTimeout(slice, 0)
    }
    id = window.setTimeout(slice, 0)
    return () => { cancelled = true; clearTimeout(id) }
  }, [version])
  return SHOWN.map((_, i) => guesses[i])
}

/** Kept separate so dragging the sliders below doesn't re-run the model on these items. */
function BenchmarkItems() {
  const guesses = useGuesses()
  const rows = SHOWN.map((item, i) => ({ item, guess: guesses[i] }))
  return (
    <Card title="Some of the questions, answered by your model now" subtitle="The model writes the rest of the word; the grader compares letters">
      <div class="scroll-x">
        <table class="grid-table">
          <thead><tr><th>Prompt</th><th>Right ending</th><th>Model wrote</th><th aria-label="Result" /></tr></thead>
          <tbody>
            {rows.map(({ item, guess }) => {
              const right = guess === item.answer, pending = guess === undefined
              return (
                <tr key={item.prompt}>
                  <td class="num">{item.prompt}…</td>
                  <td class="num" style={{ color: 'var(--muted)' }}>{item.answer}</td>
                  <td class="num" style={pending ? { color: 'var(--muted)' } : undefined}>{pending ? 'thinking…' : guess === '' ? '(nothing)' : guess}</td>
                  <td>{!pending && <span role="img" aria-label={right ? 'right' : 'wrong'} style={{ color: right ? 'var(--good)' : 'var(--muted)', fontWeight: 700 }}>{right ? '✓' : '✗'}</span>}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------- noise

function NoiseCard() {
  const [nRaw, setN] = useState(40)
  const [p, setP] = useState(0.6)
  const [gap, setGap] = useState(0.05)
  const n = Math.floor(nRaw)
  const se = standardError(p, n)
  const seDiff = Math.sqrt(2 * se * se)
  const z = gap / Math.max(seDiff, 1e-9)
  const gapPts = Math.round(gap * 100)
  return (
    <Card title="How much can you trust a score?" subtitle="The error bar of a benchmark score, from the number of questions alone">
      <div class="cols">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <Slider id="evals-n" label={`Questions: ${n}`} value={nRaw} min={10} max={2000} onInput={setN} />
          <Slider id="evals-p" label={`Score: ${Math.round(p * 100)}%`} value={p} min={0.02} max={0.98} onInput={setP} />
          <div class="stats">
            <Stat label="Standard error" value={`± ${(se * 100).toFixed(1)} points`} tone={TEAL} />
            <Stat label="95% interval" value={`${(Math.max(0, p - 1.96 * se) * 100).toFixed(0)}% – ${(Math.min(1, p + 1.96 * se) * 100).toFixed(0)}%`} tone={BLUE} />
          </div>
          <hr style={{ border: 0, borderTop: '1px solid var(--line)', margin: '4px 0', width: '100%' }} />
          <Slider id="evals-gap" label={`Model B is better by: ${gapPts} pts`} value={gap} min={0} max={0.2} onInput={setGap} />
          <p class="prose" style={{ fontSize: '15px', color: z >= 1.96 ? 'var(--ink)' : 'var(--muted)' }} aria-live="polite">
            {z >= 1.96
              ? `With ${n} questions each, a ${gapPts}-point gap is about ${z.toFixed(1)} standard errors: unlikely to be luck.`
              : `With ${n} questions each, a ${gapPts}-point gap is only ${z.toFixed(1)} standard errors: it could easily be luck.`}
          </p>
          <Note>Treats the two models' scores as independent; a paired comparison on the same questions can be sharper.</Note>
        </div>
        <ErrorBarChart p={p} n={n} />
      </div>
    </Card>
  )
}

/** ±95% error-bar width against number of questions, on a log x axis (10…2000). */
function ErrorBarChart({ p, n }: { p: number; n: number }) {
  const [ref, width] = useWidth<HTMLDivElement>(420)
  const H = 230, m = { l: 40, r: 12, t: 22, b: 36 }
  const iw = width - m.l - m.r, ih = H - m.t - m.b
  const lo = Math.log10(10), hi = Math.log10(2000)
  const sx = (q: number) => m.l + ((Math.log10(q) - lo) / (hi - lo)) * iw
  const curve: [number, number][] = []
  for (let q = 10; q <= 2000; q += 10) curve.push([q, 1.96 * standardError(p, q) * 100])
  const maxW = Math.max(...curve.map((c) => c[1]))
  const step = maxW > 20 ? 10 : 5
  const yMax = Math.ceil(maxW / step) * step
  const sy = (v: number) => m.t + ih / 2 - (v / yMax) * (ih / 2)
  const yTicks: number[] = []
  for (let v = -yMax; v <= yMax; v += step) yTicks.push(v)
  const xTicks = [10, 30, 100, 300, 1000, 2000]
  const band = curve.map(([q, w]) => `${sx(q).toFixed(1)},${sy(w).toFixed(1)}`).join(' ') + ' ' +
    [...curve].reverse().map(([q, w]) => `${sx(q).toFixed(1)},${sy(-w).toFixed(1)}`).join(' ')
  const here = 1.96 * standardError(p, n) * 100
  return (
    <div ref={ref}>
      <svg class="chart" width={width} height={H} viewBox={`0 0 ${width} ${H}`} role="img"
        aria-label={`Error bar width against number of questions, log scale: plus or minus ${here.toFixed(1)} points at ${n} questions, ${(1.96 * standardError(p, 10) * 100).toFixed(1)} at 10, ${(1.96 * standardError(p, 2000) * 100).toFixed(1)} at 2000`}>
        <text x={m.l} y={m.t - 8} text-anchor="start">± points (95%)</text>
        {yTicks.map((v) => (
          <g key={'y' + v}>
            <line class="grid" x1={m.l} x2={width - m.r} y1={sy(v)} y2={sy(v)} />
            <text x={m.l - 6} y={sy(v) + 4} text-anchor="end">{v}</text>
          </g>
        ))}
        {xTicks.map((q) => <text key={'x' + q} x={sx(q)} y={H - m.b + 15} text-anchor="middle">{q}</text>)}
        <text x={m.l + iw / 2} y={H - 4} text-anchor="middle">questions (log scale)</text>
        <polygon points={band} fill={TEAL} fill-opacity="0.25" />
        <line x1={sx(n)} x2={sx(n)} y1={m.t} y2={m.t + ih} stroke={TEAL} stroke-width="2" />
      </svg>
    </div>
  )
}
