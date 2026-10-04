import { Ops, Rng } from './tensor'
import { clipGradients, defaultConfig, GPTConfig, Optimizer, OptimizerKind, scheduledLR, TinyGPT, tokenizer, TRAINING_TEXT, VALIDATION_TEXT } from './model'

export interface Point { x: number; y: number }
export interface Sample { step: number; text: string }

export type LoopPhase = 'batch' | 'forward' | 'loss' | 'backward' | 'update'
export const loopPhases: { id: LoopPhase; title: string; code: string }[] = [
  { id: 'batch', title: 'Grab a batch of text', code: 'x, y = next_batch(data)' },
  { id: 'forward', title: 'Forward pass: predict every next token', code: 'logits = model(x)' },
  { id: 'loss', title: 'Loss: how surprised was the model?', code: 'loss = cross_entropy(logits, y)' },
  { id: 'backward', title: 'Backward pass: gradients for every weight', code: 'loss.backward()' },
  { id: 'update', title: 'Optimizer step: nudge every weight', code: 'optimizer.step(); optimizer.zero_grad()' },
]

export interface TrainSettings { peakLR: number; batchSize: number; optimizer: OptimizerKind; totalSteps: number; warmup: number; clip: number }
export const defaultSettings = (): TrainSettings => ({ peakLR: 3e-3, batchSize: 8, optimizer: 'AdamW', totalSteps: 1500, warmup: 100, clip: 1 })

const SAVE_KEY = 'llm-understander.tiny-model.v1'

/**
 * Owns the tiny model being trained. In the browser there's no background thread, so
 * training runs in short slices (≈20 ms) between frames; the page stays responsive.
 * Components subscribe and re-render when anything changes.
 */
export class Trainer {
  readonly config: GPTConfig = defaultConfig()
  readonly train = tokenizer.encode(TRAINING_TEXT)
  readonly val = tokenizer.encode(VALIDATION_TEXT)
  settings = defaultSettings()
  model = new TinyGPT(this.config)
  optimizer = new Optimizer(this.settings.optimizer)
  rng = new Rng(42)

  step = 0
  trainLoss: Point[] = []
  valLoss: Point[] = []
  learningRates: Point[] = []
  gradNorms: Point[] = []
  samples: Sample[] = []
  isRunning = false
  slowMotion = false
  phase: LoopPhase | null = null
  /** A frozen copy for the other chapters to look inside; replaced every 100 steps. */
  snapshot: TinyGPT
  version = 0

  private listeners = new Set<() => void>()
  private timer: ReturnType<typeof setTimeout> | null = null

  constructor(private persist = true) {
    this.snapshot = this.model.copy()
    if (!this.restore()) this.recordInitial()
  }

  subscribe(fn: () => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn) } }
  private emit() { this.version++; this.listeners.forEach((f) => f()) }

  get parameterCount() { return this.snapshot.parameterCount }
  get tokensSeen() { return this.step * this.settings.batchSize * this.config.context }
  get isTrained() { return this.step >= 300 }
  get finished() { return this.step >= this.settings.totalSteps }
  get uniformLoss() { return Math.log(this.config.vocab) }

  private batch(data: number[], size: number, rng: Rng): [number[], number[]][] {
    const T = this.config.context
    return Array.from({ length: size }, () => {
      const o = rng.int(data.length - T - 1)
      return [data.slice(o, o + T), data.slice(o + 1, o + T + 1)] as [number[], number[]]
    })
  }

  validationLoss(): number { return this.model.loss(this.batch(this.val, 16, new Rng(1234))).item }

  sampleText(prompt = 'the ', count = 60): string {
    const ids = this.model.generate(tokenizer.encode(prompt), count, 0.8, new Rng(this.step + 5))
    return prompt + tokenizer.decode(ids)
  }

  private recordInitial() {
    this.valLoss = [{ x: 0, y: this.validationLoss() }]
    this.samples = [{ step: 0, text: this.sampleText() }]
  }

  /** One full iteration of the training loop. */
  trainStep(onPhase?: (p: LoopPhase) => void) {
    const params = this.model.parameters
    onPhase?.('batch')
    const b = this.batch(this.train, this.settings.batchSize, this.rng)
    onPhase?.('forward')
    const logits = b.map(([x]) => this.model.forward(x))
    onPhase?.('loss')
    const loss = Ops.mean(logits.map((l, i) => Ops.crossEntropy(l, b[i][1])))
    onPhase?.('backward')
    for (const p of params) p.zeroGrad()
    loss.backward()
    const norm = clipGradients(params, this.settings.clip)
    onPhase?.('update')
    const lr = scheduledLR(this.settings.peakLR, this.settings.warmup, this.settings.totalSteps, this.step)
    this.optimizer.step(params, lr)
    this.step++
    this.trainLoss.push({ x: this.step, y: loss.item })
    this.learningRates.push({ x: this.step, y: lr })
    this.gradNorms.push({ x: this.step, y: norm })
    if (this.step % 100 === 0 || this.step === this.settings.totalSteps) {
      this.valLoss.push({ x: this.step, y: this.validationLoss() })
      this.samples.push({ step: this.step, text: this.sampleText() })
      this.snapshot = this.model.copy()
    }
  }

  start() {
    if (this.isRunning || this.finished) return
    if (this.step === 0) this.optimizer = new Optimizer(this.settings.optimizer)
    this.isRunning = true
    this.emit()
    const PHASE_MS = 450
    const loop = () => {
      if (!this.isRunning) return
      if (this.slowMotion) {
        // One step, revealed phase by phase.
        const phases: LoopPhase[] = []
        this.trainStep((p) => phases.push(p))
        let i = 0
        const show = () => {
          if (!this.isRunning) { this.phase = null; this.emit(); return }
          if (i < phases.length) { this.phase = phases[i++]; this.emit(); this.timer = setTimeout(show, PHASE_MS) } else loop()
        }
        show()
        return
      }
      this.phase = null
      const t0 = performance.now()
      while (performance.now() - t0 < 20 && !this.finished) this.trainStep()
      this.emit()
      if (this.finished) { this.stopped(); return }
      this.timer = setTimeout(loop, 0)
    }
    loop()
  }

  pause() {
    if (!this.isRunning) return
    if (this.timer) clearTimeout(this.timer)
    this.stopped()
  }

  private stopped() {
    this.isRunning = false
    this.phase = null
    this.snapshot = this.model.copy()
    this.save()
    this.emit()
  }

  /** Train synchronously (tests and quick demos). */
  trainNow(steps: number) {
    for (let i = 0; i < steps && !this.finished; i++) this.trainStep()
    this.snapshot = this.model.copy()
    this.emit()
  }

  reset() {
    this.pause()
    this.model = new TinyGPT(this.config)
    this.optimizer = new Optimizer(this.settings.optimizer)
    this.rng = new Rng(42)
    this.step = 0
    this.trainLoss = []; this.valLoss = []; this.learningRates = []; this.gradNorms = []; this.samples = []
    this.snapshot = this.model.copy()
    this.recordInitial()
    try { if (this.persist) localStorage.removeItem(SAVE_KEY) } catch { /* storage unavailable */ }
    this.emit()
  }

  setSettings(s: Partial<TrainSettings>) { this.settings = { ...this.settings, ...s }; this.emit() }
  setSlowMotion(on: boolean) { this.slowMotion = on; this.emit() }

  // ---------- keep the trained model in this browser ----------

  save() {
    if (!this.persist || this.step === 0) return
    const round = (p: Point[]) => p.map((q) => ({ x: q.x, y: +q.y.toPrecision(5) }))
    const saved = {
      config: this.config, settings: this.settings, step: this.step,
      weights: this.model.parameters.map((p) => Array.from(p.data, (v) => +v.toPrecision(7))),
      trainLoss: round(this.trainLoss), valLoss: round(this.valLoss), learningRates: round(this.learningRates), gradNorms: round(this.gradNorms),
      samples: this.samples,
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
      const params = this.model.parameters
      if (s.weights.length !== params.length) return false
      params.forEach((p, i) => p.data.set(s.weights[i]))
      Object.assign(this, { step: s.step, settings: s.settings, trainLoss: s.trainLoss, valLoss: s.valLoss, learningRates: s.learningRates, gradNorms: s.gradNorms, samples: s.samples })
      this.snapshot = this.model.copy()
      return true
    } catch { return false }
  }

  /** Training loss smoothed for charts (raw per-batch loss is noisy). */
  smoothed(): Point[] {
    let ema: number | null = null
    return this.trainLoss.map((p) => { ema = ema === null ? p.y : 0.95 * ema + 0.05 * p.y; return { x: p.x, y: ema } })
  }
}
