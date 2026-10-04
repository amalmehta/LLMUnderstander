import { useEffect, useMemo, useRef, useState } from 'preact/hooks'
import { optimizerKinds, OptimizerKind, scheduledLR } from '../engine/model'
import { Land, Landscape, landscapes } from '../engine/toys'
import { LineChart } from '../ui/charts'
import { Btn, Card, Formula, GoDeeper, KeyIdea, Note, P, Page, Segmented, Slider, Stat, Toggle, useTicker, useWidth } from '../ui/kit'

const MAX_STEPS = 120
const kindColor: Record<OptimizerKind, string> = { SGD: 'var(--c7)', Momentum: 'var(--c1)', AdamW: 'var(--c3)' }
const kindVar: Record<OptimizerKind, string> = { SGD: '--c7', Momentum: '--c1', AdamW: '--c3' }

export default function Optimizer() {
  const [landscape, setLandscape] = useState<Landscape>('Narrow ravine')
  const [logLR, setLogLR] = useState(Math.log10(0.35))
  const [shown, setShown] = useState<Record<OptimizerKind, boolean>>({ SGD: true, Momentum: true, AdamW: true })
  const [progress, setProgress] = useState(MAX_STEPS)
  const [playing, setPlaying] = useState(false)
  const [warmup, setWarmup] = useState(100)
  const total = 1500

  const lr = 10 ** logLR
  const paths = useMemo(() => {
    const out = {} as Record<OptimizerKind, [number, number][]>
    for (const k of optimizerKinds) out[k] = Land.path(landscape, k, lr, MAX_STEPS)
    return out
  }, [landscape, lr])
  const visible = optimizerKinds.filter((k) => shown[k])

  useTicker(playing, 30, () => {
    if (progress < MAX_STEPS) setProgress((p) => Math.min(MAX_STEPS, p + 1))
    else setPlaying(false)
  })

  const schedule = useMemo(() => {
    const pts: { x: number; y: number }[] = []
    for (let s = 0; s <= total; s += 10) pts.push({ x: s, y: scheduledLR(3e-3, warmup, total, s) })
    return pts
  }, [warmup])

  return (
    <Page id="optimizer">
      <P>Backprop gives every weight a gradient — the uphill direction for the loss. The **optimizer** decides how to step downhill. The simplest rule: move each weight a small amount against its gradient. That amount is the **learning rate**, the single most important number in training.</P>
      <P>Picture the loss as a landscape and the weights as a hiker in fog who can only feel the slope underfoot. Real models have billions of dimensions; here are two, so you can watch.</P>

      <Card title="Optimizer race" subtitle="Same start, same landscape, same number of steps. Brighter = lower loss; the minimum is the brightest spot.">
        <div class="row">
          <Segmented options={landscapes} value={landscape} onChange={(l) => { setLandscape(l); setProgress(0); setPlaying(true) }} />
          {optimizerKinds.map((k) => (
            <span key={k} style={{ color: kindColor[k] }}>
              <Toggle id={`optimizer-show-${k.toLowerCase()}`} label={k} checked={shown[k]} onChange={(v) => setShown({ ...shown, [k]: v })} />
            </span>
          ))}
          <span class="grow" />
          <Btn primary onClick={() => { if (progress >= MAX_STEPS) setProgress(0); setPlaying(!playing) }}>
            {playing ? 'Pause' : progress >= MAX_STEPS ? 'Replay' : 'Play'}
          </Btn>
          <Btn onClick={() => { setProgress(MAX_STEPS); setPlaying(false) }}>Show all</Btn>
        </div>
        <Slider id="optimizer-lr" label={`Learning rate ${lr.toFixed(3)}`} value={logLR} min={-2.5} max={0.2}
          onInput={(v) => { setLogLR(v); setProgress(MAX_STEPS); setPlaying(false) }} />

        <LandscapeCanvas landscape={landscape} paths={visible.map((k) => ({ kind: k, pts: paths[k] }))} progress={progress} />

        <div class="stats">
          {visible.map((k) => {
            const p = paths[k][Math.min(progress, MAX_STEPS)]
            return <Stat key={k} label={`${k} loss after ${progress} steps`} value={Land.value(landscape, p[0], p[1]).toFixed(4)} tone={kindColor[k]} />
          })}
        </div>
        <Note>Try it: in the **ravine**, plain SGD zig-zags across the steep walls and crawls along the floor. Raise the learning rate and it overshoots and blows up. **Momentum** builds up speed along the consistent direction. **Adam** gives each weight its own step size, so steep and flat directions both make progress.</Note>
      </Card>

      <Card title="The learning-rate schedule" subtitle="Real training doesn't use one fixed learning rate">
        <LineChart height={170} xDomain={[0, total]} xLabel="training step" yLabel="learning rate" legend={false} yFormat={(v) => v.toFixed(4)}
          series={[{ name: 'lr', color: 'var(--c6)', points: schedule, width: 2.5 }]} />
        <Slider id="optimizer-warmup" label={`Warm-up ${Math.round(warmup)} steps`} value={warmup} min={0} max={600} step={1} onInput={setWarmup} />
        <Note>**Warm-up:** at the start the weights are random and gradients are wild, so the learning rate ramps up gently. **Cosine decay:** later, smaller steps let the model settle into a good spot instead of bouncing around it. Our tiny model uses exactly this schedule in Chapter 9.</Note>
      </Card>

      <KeyIdea>Gradient = which way. Learning rate = how far. Optimizer = how to combine the current gradient with what it has seen before. Nearly every LLM is trained with AdamW plus warm-up and decay.</KeyIdea>

      <GoDeeper title="the update rules">
        <Formula>{'SGD:       w ← w − lr · g\nMomentum:  v ← β·v + g;          w ← w − lr · v\nAdam:      m ← β₁·m + (1−β₁)·g     (average direction)\n           s ← β₂·s + (1−β₂)·g²    (average size²)\n           w ← w − lr · m̂ / (√ŝ + ε) − lr · λ · w   (AdamW decay)'}</Formula>
        <P>**Adam** divides each weight's step by the typical size of its gradients, so weights with tiny gradients still move and weights with huge ones don't explode. The cost: two extra numbers stored per weight. For a 70B-parameter model that's hundreds of gigabytes of optimizer state.</P>
        <P>**Weight decay** gently pulls weights toward zero, a mild preference for simpler solutions. **Gradient clipping** caps the total gradient size, so one bad batch can't throw training off a cliff.</P>
        <P>**Batch size:** each gradient is averaged over a batch of examples. Bigger batches give less noisy directions and use hardware better. LLMs use batches of millions of tokens.</P>
      </GoDeeper>
    </Page>
  )
}

// ---------------------------------------------------------------- landscape canvas

/** Bumps whenever the colour theme changes (system setting or the site's own toggle). */
function useThemeVersion() {
  const [v, setV] = useState(0)
  useEffect(() => {
    const bump = () => setV((n) => n + 1)
    const mq = matchMedia('(prefers-color-scheme: dark)')
    mq.addEventListener('change', bump)
    const mo = new MutationObserver(bump)
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class', 'style'] })
    return () => { mq.removeEventListener('change', bump); mo.disconnect() }
  }, [])
  return v
}

function rgbOf(cssVar: string): [number, number, number] {
  const probe = document.createElement('span')
  probe.style.color = `var(${cssVar})`
  probe.style.display = 'none'
  document.body.appendChild(probe)
  const m = getComputedStyle(probe).color.match(/[\d.]+/g) ?? ['0', '0', '0']
  probe.remove()
  return [+m[0], +m[1], +m[2]]
}

function LandscapeCanvas({ landscape, paths, progress }: { landscape: Landscape; paths: { kind: OptimizerKind; pts: [number, number][] }[]; progress: number }) {
  const [wrap, width] = useWidth<HTMLDivElement>()
  const canvas = useRef<HTMLCanvasElement>(null)
  const theme = useThemeVersion()
  const height = Math.round(Math.min(360, Math.max(240, width * 0.62)))

  // The heat background depends only on the landscape and theme: render it once into a small bitmap.
  const background = useMemo(() => {
    if (typeof document === 'undefined') return null
    const { x: xr, y: yr } = Land.range(landscape)
    const nx = 180, ny = 110
    const vals: number[] = []
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const x = xr[0] + ((i + 0.5) / nx) * (xr[1] - xr[0])
      const y = yr[1] - ((j + 0.5) / ny) * (yr[1] - yr[0])
      vals.push(Math.log(Land.value(landscape, x, y) + 0.01))
    }
    const lo = Math.min(...vals), hi = Math.max(...vals)
    // Low loss = the bright warm theme colour; high loss = a darkened cool theme colour.
    const warm = rgbOf('--c2'), cool = rgbOf('--c1').map((c) => c * 0.4)
    const off = document.createElement('canvas')
    off.width = nx; off.height = ny
    const ctx = off.getContext('2d')!
    const img = ctx.createImageData(nx, ny)
    vals.forEach((v, k) => {
      const t = (v - lo) / (hi - lo || 1)
      const band = Math.floor(t * 14) % 2 === 0 ? 1 : 0.93 // contour banding
      for (let c = 0; c < 3; c++) img.data[k * 4 + c] = (warm[c] * (1 - t) + cool[c] * t) * band
      img.data[k * 4 + 3] = 255
    })
    ctx.putImageData(img, 0, 0)
    return off
  }, [landscape, theme])

  useEffect(() => {
    const el = canvas.current
    if (!el || !background) return
    const dpr = window.devicePixelRatio || 1
    el.width = Math.round(width * dpr); el.height = Math.round(height * dpr)
    const ctx = el.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.imageSmoothingEnabled = true
    ctx.drawImage(background, 0, 0, width, height)

    const { x: xr, y: yr } = Land.range(landscape)
    const toScreen = (x: number, y: number): [number, number] =>
      [((x - xr[0]) / (xr[1] - xr[0])) * width, (1 - (y - yr[0]) / (yr[1] - yr[0])) * height]
    const css = getComputedStyle(document.documentElement)
    const panel = css.getPropertyValue('--panel').trim(), ink = css.getPropertyValue('--ink').trim()

    for (const { kind, pts } of paths) {
      const color = css.getPropertyValue(kindVar[kind]).trim()
      const upto = Math.min(progress, pts.length - 1)
      ctx.beginPath()
      pts.slice(0, upto + 1).forEach((p, n) => { const [sx, sy] = toScreen(p[0], p[1]); if (n === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy) })
      ctx.lineJoin = 'round'
      ctx.strokeStyle = panel; ctx.globalAlpha = 0.55; ctx.lineWidth = 4.5; ctx.stroke()
      ctx.globalAlpha = 1; ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.stroke()
      const [cx, cy] = toScreen(pts[upto][0], pts[upto][1])
      ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2)
      ctx.fillStyle = color; ctx.fill()
      ctx.strokeStyle = panel; ctx.lineWidth = 1.5; ctx.stroke()
    }
    const [stx, sty] = toScreen(...Land.start(landscape))
    ctx.font = `700 12px ${css.getPropertyValue('--font-display')}`
    ctx.textAlign = 'center'
    ctx.lineWidth = 3; ctx.strokeStyle = panel; ctx.strokeText('start', stx, sty - 12)
    ctx.fillStyle = ink; ctx.fillText('start', stx, sty - 12)
  }, [background, paths, progress, width, height, landscape])

  return (
    <div ref={wrap}>
      <canvas ref={canvas} role="img" aria-label={`Optimizer paths on the ${landscape} loss landscape`}
        style={{ width: `${width}px`, height: `${height}px`, display: 'block', borderRadius: '10px' }} />
    </div>
  )
}
