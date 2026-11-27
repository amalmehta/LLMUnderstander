// The training loop itself, with no UI and no timers. The Web Worker runs one of these;
// tests and browsers without workers run it on the page. Mirrors TrainingWorker in Trainer.swift.
import { Ops, Rng } from './tensor'
import { clipGradients, defaultConfig, GPTConfig, Optimizer, OptimizerState, scheduledLR, TinyGPT, tokenizer, TRAINING_TEXT, VALIDATION_TEXT } from './model'
import { BenchScore, scoreBenchmark } from './benchmark'

export interface Point { x: number; y: number }
export interface Sample { step: number; text: string }
export type OptimizerKind = 'SGD' | 'Momentum' | 'AdamW'
export type LoopPhase = 'batch' | 'forward' | 'loss' | 'backward' | 'update'

export interface TrainSettings { peakLR: number; batchSize: number; optimizer: OptimizerKind; totalSteps: number; warmup: number; clip: number }
export const defaultSettings = (): TrainSettings => ({ peakLR: 3e-3, batchSize: 8, optimizer: 'AdamW', totalSteps: 1500, warmup: 100, clip: 1 })

export interface StepResult { step: number; loss: number; lr: number; gradNorm: number }
/** What the run reports every 100 steps: validation loss, a sample, the benchmark and a copy of the weights. */
export interface Checkpoint { step: number; val: number; sample: string; bench: BenchScore; weights: number[][] }

/** Everything needed to resume a run exactly (weights, optimizer moments, batch sampler position). */
export interface RunState { step: number; settings: TrainSettings; weights: number[][]; optimizer: OptimizerState; rng: string }

export class TrainingRun {
  readonly config: GPTConfig = defaultConfig()
  readonly train = tokenizer.encode(TRAINING_TEXT)
  readonly val = tokenizer.encode(VALIDATION_TEXT)
  model = new TinyGPT(this.config)
  optimizer: Optimizer
  rng = new Rng(42)
  step = 0
  prompt = 'the '

  constructor(public settings: TrainSettings = defaultSettings()) {
    this.optimizer = new Optimizer(settings.optimizer)
  }

  batch(data: number[], size: number, rng: Rng): [number[], number[]][] {
    const T = this.config.context
    return Array.from({ length: size }, () => {
      const o = rng.int(data.length - T - 1)
      return [data.slice(o, o + T), data.slice(o + 1, o + T + 1)] as [number[], number[]]
    })
  }

  validationLoss(): number { return this.model.loss(this.batch(this.val, 16, new Rng(1234))).item }

  sample(count = 60): string {
    const prompt = tokenizer.clean(this.prompt)
    const ids = this.model.generate(tokenizer.encode(prompt), count, 0.8, new Rng(this.step + 5))
    return prompt + tokenizer.decode(ids)
  }

  /** One full iteration of the training loop. */
  trainStep(onPhase?: (p: LoopPhase) => void): StepResult {
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
    const gradNorm = clipGradients(params, this.settings.clip)
    onPhase?.('update')
    const lr = scheduledLR(this.settings.peakLR, this.settings.warmup, this.settings.totalSteps, this.step)
    this.optimizer.step(params, lr)
    this.step++
    return { step: this.step, loss: loss.item, lr, gradNorm }
  }

  get finished() { return this.step >= this.settings.totalSteps }
  get isCheckpoint() { return this.step % 100 === 0 || this.step === this.settings.totalSteps }

  checkpoint(): Checkpoint {
    return { step: this.step, val: this.validationLoss(), sample: this.sample(), bench: scoreBenchmark(this.model), weights: this.weights() }
  }

  weights(): number[][] { return this.model.parameters.map((p) => Array.from(p.data)) }

  state(): RunState {
    return { step: this.step, settings: this.settings, weights: this.weights(), optimizer: this.optimizer.state(this.model.parameters), rng: this.rng.state.toString() }
  }

  restore(s: RunState): boolean {
    const params = this.model.parameters
    if (s.weights.length !== params.length || !s.weights.every((w, i) => w.length === params[i].count)) return false
    params.forEach((p, i) => p.data.set(s.weights[i]))
    this.step = s.step
    this.settings = s.settings
    this.optimizer = new Optimizer(s.settings.optimizer)
    if (s.optimizer) this.optimizer.restore(s.optimizer, params)
    if (s.rng) this.rng.state = BigInt(s.rng)
    return true
  }
}
