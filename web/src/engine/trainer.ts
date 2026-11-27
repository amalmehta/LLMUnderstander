// The page's view of the training run. Training itself happens in a Web Worker (train.worker.ts),
// so the page stays smooth; where workers aren't available (tests, locked-down frames) the same
// TrainingRun runs here in short time slices instead. Components subscribe and re-render on change.
import { defaultConfig, GPTConfig, TinyGPT } from './model'
import { TrainingRun, defaultSettings, TrainSettings, Point, Sample, LoopPhase, RunState, StepResult } from './run'
import type { FromWorker, ToWorker } from './train.worker'
import TrainWorker from './train.worker?worker&inline'

export type { Point, Sample, LoopPhase, TrainSettings }
export { defaultSettings }

export const loopPhases: { id: LoopPhase; title: string; code: string }[] = [
  { id: 'batch', title: 'Grab a batch of text', code: 'x, y = next_batch(data)' },
  { id: 'forward', title: 'Forward pass: predict every next token', code: 'logits = model(x)' },
  { id: 'loss', title: 'Loss: how surprised was the model?', code: 'loss = cross_entropy(logits, y)' },
  { id: 'backward', title: 'Backward pass: gradients for every weight', code: 'loss.backward()' },
  { id: 'update', title: 'Optimizer step: nudge every weight', code: 'optimizer.step(); optimizer.zero_grad()' },
]

const SAVE_KEY = 'llm-understander.tiny-model.v2'
/** Trained = validation loss at or below 60% of random guessing (DEFINITIONS in the spec). */
export const TRAINED_FRACTION = 0.6

export class Trainer {
  readonly config: GPTConfig = defaultConfig()
  settings = defaultSettings()
  step = 0
  trainLoss: Point[] = []
  valLoss: Point[] = []
  learningRates: Point[] = []
  gradNorms: Point[] = []
  samples: Sample[] = []
  benchmark: Point[] = []
  benchmarkPerChar: Point[] = []
  samplePrompt = 'the '
  isRunning = false
  slowMotion = false
  phase: LoopPhase | null = null
  /** A frozen copy for the other chapters to look inside; replaced every 100 steps. */
  snapshot: TinyGPT
  /** Bumps on every change (for re-rendering) … */
  version = 0
  /** … and this one only when the snapshot changes (for work that reads the model). */
  snapshotVersion = 0

  private listeners = new Set<() => void>()
  private worker: Worker | null = null
  /** Used when there's no worker: the run lives on the page. */
  private local: TrainingRun | null = null
  private timer: ReturnType<typeof setTimeout> | null = null
  private savedState: RunState | null = null

  constructor(private persist = true, useWorker = true) {
    this.snapshot = new TinyGPT(this.config)
    const restored = this.restore()
    if (useWorker && typeof Worker !== 'undefined') {
      try {
        this.worker = new TrainWorker()
        this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.receive(e.data)
        this.send({ type: 'init', state: this.savedState, settings: this.settings, prompt: this.samplePrompt })
      } catch { this.worker = null }
    }
    if (!this.worker) {
      this.local = new TrainingRun(this.settings)
      if (this.savedState) this.local.restore(this.savedState)
    }
    if (!restored) this.recordInitial()
  }

  subscribe(fn: () => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn) } }
  private emit() { this.version++; this.listeners.forEach((f) => f()) }
  private send(m: ToWorker) { this.worker?.postMessage(m) }

  get parameterCount() { return this.snapshot.parameterCount }
  get tokensSeen() { return this.step * this.settings.batchSize * this.config.context }
  get uniformLoss() { return Math.log(this.config.vocab) }
  get trainedLoss() { return TRAINED_FRACTION * this.uniformLoss }
  get latestValLoss(): number | null { return this.valLoss.length ? this.valLoss[this.valLoss.length - 1].y : null }
  get isTrained() { return (this.latestValLoss ?? Infinity) <= this.trainedLoss }
  get finished() { return this.step >= this.settings.totalSteps }

  /** Loss, sample and benchmark of the untrained model, so charts start at step 0. */
  private recordInitial() {
    const r = new TrainingRun(this.settings)
    r.prompt = this.samplePrompt
    const c = r.checkpoint()
    this.valLoss = [{ x: 0, y: c.val }]
    this.samples = [{ step: 0, text: c.sample }]
    this.benchmark = [{ x: 0, y: c.bench.exact }]
    this.benchmarkPerChar = [{ x: 0, y: c.bench.perCharacter }]
  }

  // ---------- messages from the run ----------

  private receive(m: FromWorker) {
    switch (m.type) {
      case 'steps': this.addSteps(m.results); break
      case 'slow': this.addSteps([m.result]); this.playPhases(m.phases); return
      case 'checkpoint': this.addCheckpoint(m); break
      case 'stopped': this.onStopped(m.state); break
    }
    this.emit()
  }

  private addSteps(results: StepResult[]) {
    for (const r of results) {
      if (this.trainLoss.length && r.step <= this.trainLoss[this.trainLoss.length - 1].x) continue
      this.trainLoss.push({ x: r.step, y: r.loss })
      this.learningRates.push({ x: r.step, y: r.lr })
      this.gradNorms.push({ x: r.step, y: r.gradNorm })
      this.step = r.step
    }
  }

  private addCheckpoint(c: { step: number; val: number; sample: string; bench: { exact: number; perCharacter: number }; weights: number[][] }) {
    this.valLoss.push({ x: c.step, y: c.val })
    this.samples.push({ step: c.step, text: c.sample })
    this.benchmark.push({ x: c.step, y: c.bench.exact })
    this.benchmarkPerChar.push({ x: c.step, y: c.bench.perCharacter })
    const snap = new TinyGPT(this.config)
    snap.parameters.forEach((p, i) => p.data.set(c.weights[i]))
    this.snapshot = snap
    this.snapshotVersion++
  }

  private onStopped(state: RunState) {
    this.isRunning = false
    this.phase = null
    this.savedState = state
    const snap = new TinyGPT(this.config)
    snap.parameters.forEach((p, i) => p.data.set(state.weights[i]))
    this.snapshot = snap
    this.snapshotVersion++
    this.save()
  }

  // ---------- controls ----------

  start() {
    if (this.isRunning || this.finished) return
    this.isRunning = true
    this.emit()
    if (this.slowMotion) { this.slowStep(); return }
    if (this.worker) this.send({ type: 'start' })
    else this.localLoop()
  }

  pause() {
    if (!this.isRunning) return
    if (this.timer) clearTimeout(this.timer)
    if (this.worker) this.send({ type: 'pause' })
    else if (this.local) { this.receive({ type: 'stopped', state: this.local.state() }) }
    if (this.slowMotion || !this.worker) { this.isRunning = false; this.phase = null; this.emit() }
  }

  /** Slow motion: one step, then its five phases revealed one by one, then the next step. */
  private slowStep() {
    if (!this.isRunning) return
    if (this.worker) { this.send({ type: 'slowStep' }); return }
    const phases: LoopPhase[] = []
    const r = this.local!.trainStep((p) => phases.push(p))
    this.addSteps([r])
    if (this.local!.isCheckpoint) this.addCheckpoint(this.local!.checkpoint())
    this.playPhases(phases)
  }

  private playPhases(phases: LoopPhase[]) {
    let i = 0
    const show = () => {
      if (!this.isRunning) { this.phase = null; this.emit(); return }
      if (i < phases.length) { this.phase = phases[i++]; this.emit(); this.timer = setTimeout(show, 450); return }
      if (this.finished) { this.pauseAfterSlow(); return }
      if (this.slowMotion) this.slowStep()
      else { this.phase = null; if (this.worker) this.send({ type: 'start' }); else this.localLoop() }
    }
    show()
  }

  private pauseAfterSlow() {
    this.isRunning = false
    this.phase = null
    if (this.worker) this.send({ type: 'pause' })
    else if (this.local) this.onStopped(this.local.state())
    this.emit()
  }

  /** No worker: train in ~20 ms slices between frames. */
  private localLoop() {
    const run = this.local!
    if (!this.isRunning) return
    const t0 = performance.now()
    const results: StepResult[] = []
    while (performance.now() - t0 < 20 && !run.finished) {
      results.push(run.trainStep())
      if (run.isCheckpoint) { this.addSteps(results.splice(0)); this.addCheckpoint(run.checkpoint()); break }
    }
    this.addSteps(results)
    this.emit()
    if (run.finished) { this.onStopped(run.state()); this.emit(); return }
    this.timer = setTimeout(() => this.localLoop(), 0)
  }

  /** Train synchronously on the page (tests and screenshots). */
  trainNow(steps: number) {
    const run = this.local ?? (this.local = new TrainingRun(this.settings))
    for (let i = 0; i < steps && !run.finished; i++) {
      this.addSteps([run.trainStep()])
      if (run.isCheckpoint) this.addCheckpoint(run.checkpoint())
    }
    this.savedState = run.state()
    this.snapshot = run.model.copy()
    this.snapshotVersion++
    this.emit()
  }

  reset() {
    if (this.timer) clearTimeout(this.timer)
    this.isRunning = false
    this.phase = null
    this.step = 0
    this.trainLoss = []; this.valLoss = []; this.learningRates = []; this.gradNorms = []; this.samples = []
    this.benchmark = []; this.benchmarkPerChar = []
    this.savedState = null
    this.snapshot = new TinyGPT(this.config)
    this.snapshotVersion++
    if (this.worker) this.send({ type: 'reset', settings: this.settings })
    else { this.local = new TrainingRun(this.settings); this.local.prompt = this.samplePrompt }
    this.recordInitial()
    try { if (this.persist) localStorage.removeItem(SAVE_KEY) } catch { /* storage unavailable */ }
    this.emit()
  }

  setSettings(s: Partial<TrainSettings>) {
    this.settings = { ...this.settings, ...s }
    if (this.worker) this.send({ type: 'settings', settings: this.settings })
    else if (this.local && this.step === 0) { this.local = new TrainingRun(this.settings); this.local.prompt = this.samplePrompt }
    this.emit()
  }

  setSlowMotion(on: boolean) { this.slowMotion = on; this.emit() }

  setSamplePrompt(p: string) {
    this.samplePrompt = p
    if (this.worker) this.send({ type: 'prompt', prompt: p })
    if (this.local) this.local.prompt = p
    this.emit()
  }

  // ---------- keep the trained model in this browser ----------

  save() {
    if (!this.persist || this.step === 0 || !this.savedState) return
    const round = (p: Point[]) => p.map((q) => ({ x: q.x, y: +q.y.toPrecision(5) }))
    const saved = {
      config: this.config, run: this.savedState,
      trainLoss: round(this.trainLoss), valLoss: round(this.valLoss), learningRates: round(this.learningRates), gradNorms: round(this.gradNorms),
      samples: this.samples, benchmark: round(this.benchmark), benchmarkPerChar: round(this.benchmarkPerChar),
    }
    try { localStorage.setItem(SAVE_KEY, JSON.stringify(saved)) } catch { /* storage full or blocked: training still works */ }
  }

  private restore(): boolean {
    if (!this.persist) return false
    try {
      const raw = localStorage.getItem(SAVE_KEY)
      if (!raw) return false
      const s = JSON.parse(raw)
      if (JSON.stringify(s.config) !== JSON.stringify(this.config)) return false
      const snap = new TinyGPT(this.config)
      const params = snap.parameters
      if (s.run.weights.length !== params.length) return false
      params.forEach((p, i) => p.data.set(s.run.weights[i]))
      this.snapshot = snap
      this.savedState = s.run
      this.step = s.run.step
      this.settings = s.run.settings
      Object.assign(this, {
        trainLoss: s.trainLoss, valLoss: s.valLoss, learningRates: s.learningRates, gradNorms: s.gradNorms,
        samples: s.samples, benchmark: s.benchmark ?? [], benchmarkPerChar: s.benchmarkPerChar ?? [],
      })
      return true
    } catch { return false }
  }

  /** Training loss smoothed for charts (raw per-batch loss is noisy). */
  smoothed(): Point[] {
    let ema: number | null = null
    return this.trainLoss.map((p) => { ema = ema === null ? p.y : 0.95 * ema + 0.05 * p.y; return { x: p.x, y: ema } })
  }
}
