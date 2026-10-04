import { useState } from 'preact/hooks'
import { tokenizer } from '../engine/model'
import { Ops, Rng, Tensor } from '../engine/tensor'
import { LineChart } from '../ui/charts'
import { Btn, Card, Formula, GoDeeper, KeyIdea, ModelBanner, Note, P, Page, md, useTrainer } from '../ui/kit'

const X = 2, TARGET = 7, LR = 0.05

type Key = 'x' | 'w' | 'b' | 'm' | 'ŷ' | 'e' | 'L'
interface Graph { values: Record<Key, number>; grads: Record<Key, number> }

/** Every node of L = (w·x + b − y)², computed with the site's autograd engine. */
function computeGraph(w: number, b: number): Graph {
  const one = (v: number) => new Tensor(1, 1, [v])
  const tx = one(X), tw = one(w), tb = one(b), ty = one(TARGET)
  const m = Ops.matmul(tw, tx)
  const yhat = Ops.add(m, tb)
  const e = Ops.add(yhat, Ops.scale(ty, -1))
  const L = Ops.matmul(e, e)
  L.backward()
  const nodes: [Key, Tensor][] = [['x', tx], ['w', tw], ['b', tb], ['m', m], ['ŷ', yhat], ['e', e], ['L', L]]
  const g = { values: {}, grads: {} } as Graph
  for (const [k, t] of nodes) { g.values[k] = t.item; g.grads[k] = t.grad[0] }
  return g
}

const steps: { reveal: Key[]; grad: boolean; text: string }[] = [
  { reveal: [], grad: false, text: 'Start: we have an input **x = 2**, a correct answer **y = 7**, and two weights **w** and **b**. Press *Next* to run the forward pass.' },
  { reveal: ['m'], grad: false, text: '**Forward 1:** multiply: m = w × x.' },
  { reveal: ['ŷ'], grad: false, text: "**Forward 2:** add the bias: ŷ = m + b. This is the model's prediction." },
  { reveal: ['e'], grad: false, text: '**Forward 3:** error e = ŷ − y.' },
  { reveal: ['L'], grad: false, text: '**Forward 4:** loss L = e². Now go *backwards* to ask: how would L change if each number changed a little?' },
  { reveal: ['L'], grad: true, text: '**Backward 1:** dL/dL = 1 (trivially).' },
  { reveal: ['e'], grad: true, text: '**Backward 2:** L = e², so dL/de = 2e.' },
  { reveal: ['ŷ'], grad: true, text: '**Backward 3:** e = ŷ − y, so dL/dŷ = dL/de × 1. Addition passes the gradient through unchanged.' },
  { reveal: ['m', 'b'], grad: true, text: '**Backward 4:** ŷ = m + b, so both get the same gradient: dL/dm = dL/db = dL/dŷ.' },
  { reveal: ['w', 'x'], grad: true, text: "**Backward 5:** m = w × x, so dL/dw = dL/dm × **x**, and dL/dx = dL/dm × **w**. Each multiply swaps in the *other* input. That's the chain rule." },
  { reveal: ['w', 'b'], grad: true, text: '**Update:** now we know which way to push. Press *Take a gradient step*: w ← w − lr·dL/dw, b ← b − lr·dL/db. The loss drops.' },
]

interface CheckRow { name: string; analytic: number; numeric: number; ok: boolean }

export default function Backprop() {
  const t = useTrainer()
  const [w, setW] = useState(1)
  const [b, setB] = useState(0.5)
  const [step, setStep] = useState(0)
  const [history, setHistory] = useState<{ x: number; y: number }[]>([])
  const [checks, setChecks] = useState<CheckRow[]>([])

  const g = computeGraph(w, b)
  const shown = steps.slice(0, step + 1)
  const vals = new Set<Key>(['x', 'w', 'b', ...shown.filter((s) => !s.grad).flatMap((s) => s.reveal)])
  const grads = new Set<Key>(shown.filter((s) => s.grad).flatMap((s) => s.reveal))

  const takeStep = () => {
    const h = history.length ? history : [{ x: 0, y: g.values.L }]
    const nw = w - LR * g.grads.w, nb = b - LR * g.grads.b
    setW(nw); setB(nb)
    setHistory([...h, { x: h.length, y: computeGraph(nw, nb).values.L }])
  }

  const runCheck = () => {
    const model = t.snapshot.copy()
    const s = tokenizer.encode('the fox finds a green leaf near the house.')
    const batch: [number[], number[]][] = [[s.slice(0, 32), s.slice(1, 33)]]
    const params = model.parameters
    for (const p of params) p.zeroGrad()
    model.loss(batch).backward()
    const rng = new Rng(Date.now() >>> 0)
    const rows: CheckRow[] = []
    for (let n = 0; n < 8; n++) {
      const p = params[rng.int(params.length)]
      const i = rng.int(p.count)
      const old = p.data[i], h = 1e-5
      p.data[i] = old + h; const up = model.loss(batch).item
      p.data[i] = old - h; const down = model.loss(batch).item
      p.data[i] = old
      const analytic = p.grad[i], numeric = (up - down) / (2 * h)
      rows.push({ name: `${p.label}[${i}]`, analytic, numeric, ok: Math.abs(analytic - numeric) <= 1e-4 * Math.max(1, Math.abs(analytic) + Math.abs(numeric)) })
    }
    setChecks(rows)
  }

  const node = (k: Key, caption: string, tone: string) => {
    const hasGrad = grads.has(k)
    return (
      <div style={{
        width: '96px', minHeight: '82px', padding: '6px 4px', borderRadius: '10px', textAlign: 'center', fontFamily: 'var(--font-display)',
        background: `color-mix(in srgb, ${tone} 12%, var(--panel))`,
        border: hasGrad ? '2px solid var(--bad)' : `1px solid color-mix(in srgb, ${tone} 40%, var(--line))`,
        display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '1px',
      }}>
        <b style={{ fontFamily: 'var(--font-mono)', fontSize: '16px' }}>{k}</b>
        <small style={{ fontSize: '10px', color: 'var(--muted)' }}>{caption}</small>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px' }}>{vals.has(k) ? g.values[k].toFixed(2) : '?'}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 600, color: 'var(--bad)', minHeight: '1.3em' }}>{hasGrad ? `grad ${g.grads[k].toFixed(2)}` : ' '}</span>
      </div>
    )
  }
  const arrow = <span aria-hidden="true" style={{ color: 'var(--muted)', alignSelf: 'center' }}>→</span>
  const column = (...kids: preact.JSX.Element[]) => <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', justifyContent: 'center' }}>{kids}</div>

  return (
    <Page id="backprop">
      <ModelBanner />
      <P>We know how wrong the model is (the loss). Now: **which of its weights should change, and in which direction?** For every weight we want its **gradient** — how much the loss would go up if that weight were nudged up a tiny bit.</P>
      <P>Computing that one weight at a time would take forever with billions of weights. **Backpropagation** gets all of them in one backward sweep, costing only about twice a forward pass. It's just the chain rule from calculus, applied from the loss backwards.</P>

      <Card title="Step through backprop on a one-neuron model" subtitle="L = (w·x + b − y)²   — gradients computed by the same autograd engine that trains the tiny GPT">
        <div class="scroll-x">
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', minWidth: 'max-content', padding: '6px 2px' }}>
            {column(node('x', 'input', 'var(--c8)'), node('w', 'weight', 'var(--c1)'))}
            {arrow}
            {node('m', 'w × x', 'var(--c6)')}
            {arrow}
            {column(node('ŷ', 'm + b', 'var(--c6)'), node('b', 'weight', 'var(--c1)'))}
            {arrow}
            {node('e', 'ŷ − y (y = 7)', 'var(--c6)')}
            {arrow}
            {node('L', 'e²', 'var(--c2)')}
          </div>
        </div>
        <p style={{ margin: 0, minHeight: '3em' }}>{md(steps[step].text)}</p>
        <div class="row">
          <Btn onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>Back</Btn>
          <Btn primary onClick={() => setStep(Math.min(steps.length - 1, step + 1))} disabled={step === steps.length - 1}>Next</Btn>
          <span class="grow" />
          <Btn onClick={takeStep} disabled={step < steps.length - 1}>Take a gradient step</Btn>
          <Btn onClick={() => { setW(1); setB(0.5); setStep(0); setHistory([]) }}>Reset</Btn>
        </div>
        {history.length > 0 && (
          <>
            <LineChart height={140} xLabel="gradient steps" yLabel="loss" legend={false}
              series={[{ name: 'loss', color: 'var(--c2)', points: history, dots: true }]} />
            <p class="mono" style={{ margin: 0 }}>{`w = ${w.toFixed(3)}   b = ${b.toFixed(3)}   prediction ŷ = ${g.values['ŷ'].toFixed(3)} (target 7)`}</p>
          </>
        )}
      </Card>

      <Card title="The same thing, on every weight of the real model" subtitle="Gradient check: compare backprop's answer with brute force (nudge the weight, re-run, measure)">
        <div class="row">
          <Btn primary onClick={runCheck}>Run gradient check on 8 random weights</Btn>
          <Note>{`Our model has ${t.parameterCount.toLocaleString()} weights. One backward pass gives *all* their gradients at once.`}</Note>
        </div>
        {checks.length > 0 && (
          <div class="scroll-x">
            <table class="grid-table">
              <thead><tr><th>Weight</th><th>Backprop</th><th>Brute force</th><th></th></tr></thead>
              <tbody>
                {checks.map((c, i) => (
                  <tr key={i}>
                    <td class="mono">{c.name}</td>
                    <td class="num">{signed(c.analytic)}</td>
                    <td class="num">{signed(c.numeric)}</td>
                    <td style={{ color: c.ok ? 'var(--good)' : 'var(--bad)', fontWeight: 700 }} aria-label={c.ok ? 'match' : 'mismatch'}>{c.ok ? '✓' : '✗'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <KeyIdea>A gradient is a direction: "to reduce the loss, move this weight that way, this much." Backprop computes it for every weight in one sweep by passing blame backwards through each operation.</KeyIdea>

      <GoDeeper title="why it works at scale">
        <Formula>Chain rule:  dL/dw = dL/dŷ · dŷ/dm · dm/dw</Formula>
        <P>Every operation (matrix multiply, softmax, LayerNorm, GELU…) knows its own small *local* derivative. Autograd records the operations during the forward pass, then visits them in reverse, multiplying local derivatives together. Frameworks like PyTorch and JAX do exactly what this site's ~300-line engine does.</P>
        <P>**Memory:** the backward pass needs the activations saved from the forward pass. For big models that memory, not the weights, is often the bottleneck. *Activation checkpointing* throws some away and recomputes them when needed.</P>
        <P>**Vanishing/exploding gradients:** multiplying many local derivatives can shrink or blow up the result. Residual connections (Chapter 5), normalisation and careful initialisation keep gradients healthy through hundreds of layers.</P>
      </GoDeeper>
    </Page>
  )
}

const signed = (v: number) => (v >= 0 ? '+' : '') + v.toFixed(6)
