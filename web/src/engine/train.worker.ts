// Runs the training loop off the page's main thread, so scrolling and clicks stay smooth.
// Protocol (page → worker): init, start, pause, reset, settings, prompt, slowStep.
// Worker → page: steps (a batch of results), checkpoint (every 100 steps), state (on stop), phases.
import { TrainingRun, RunState, TrainSettings, StepResult, LoopPhase } from './run'

export type ToWorker =
  | { type: 'init'; state: RunState | null; settings: TrainSettings; prompt: string }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'reset'; settings: TrainSettings }
  | { type: 'settings'; settings: TrainSettings }
  | { type: 'prompt'; prompt: string }
  | { type: 'slowStep' }

export type FromWorker =
  | { type: 'steps'; results: StepResult[] }
  | { type: 'checkpoint'; step: number; val: number; sample: string; bench: { exact: number; perCharacter: number }; weights: number[][] }
  | { type: 'stopped'; state: RunState }
  | { type: 'slow'; result: StepResult; phases: LoopPhase[] }

let run = new TrainingRun()
let running = false

const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m)

function sendCheckpoint() {
  const c = run.checkpoint()
  post({ type: 'checkpoint', ...c })
}

function loop() {
  if (!running) return
  const results: StepResult[] = []
  const t0 = performance.now()
  // Report about every 50 ms, and always right after a checkpoint step.
  while (running && !run.finished && performance.now() - t0 < 50) {
    results.push(run.trainStep())
    if (run.isCheckpoint) break
  }
  if (results.length) post({ type: 'steps', results })
  if (run.isCheckpoint && results.length) sendCheckpoint()
  if (run.finished) { stop(); return }
  setTimeout(loop, 0)   // lets pause messages in between batches
}

function stop() {
  running = false
  post({ type: 'stopped', state: run.state() })
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data
  switch (m.type) {
    case 'init':
      run = new TrainingRun(m.settings)
      run.prompt = m.prompt
      if (m.state) run.restore(m.state)
      break
    case 'start':
      if (running || run.finished) return
      if (run.step === 0) run = Object.assign(new TrainingRun(run.settings), { prompt: run.prompt })
      running = true
      loop()
      break
    case 'pause':
      if (running) stop()
      break
    case 'reset':
      running = false
      run = Object.assign(new TrainingRun(m.settings), { prompt: run.prompt })
      break
    case 'settings':
      if (run.step === 0) run = Object.assign(new TrainingRun(m.settings), { prompt: run.prompt })
      else run.settings = m.settings
      break
    case 'prompt':
      run.prompt = m.prompt
      break
    case 'slowStep': {
      if (run.finished) return
      const phases: LoopPhase[] = []
      const result = run.trainStep((p) => phases.push(p))
      post({ type: 'slow', result, phases })
      if (run.isCheckpoint) sendCheckpoint()
      if (run.finished) stop()
      break
    }
  }
}
