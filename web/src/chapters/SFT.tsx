import { useEffect, useState } from 'preact/hooks'
import { demonstrations, heldOutQuestions, startsWithSubject, SUGGESTED_STEPS, trainingRow, usesVerb } from '../engine/chatTuning'
import { clipGradients, Optimizer, TinyGPT, tokenizer } from '../engine/model'
import { Num, Ops, Rng } from '../engine/tensor'
import { PostTraining, Point } from '../engine/toys'
import { HBarChart, LineChart, Legend } from '../ui/charts'
import { Btn, Card, Formula, GoDeeper, KeyIdea, ModelBanner, Note, P, Page, Slider, Stat, Toggle, useTrainer } from '../ui/kit'
import './SFT.css'

// (text, is special token, counts toward loss)
const template: [string, boolean, boolean][] = [
  ['<|system|>', true, false], ['You', false, false], [' are', false, false], [' a', false, false], [' helpful', false, false], [' assistant.', false, false],
  ['<|user|>', true, false], ['how', false, false], [' do', false, false], [' i', false, false], [' boil', false, false], [' an', false, false], [' egg?', false, false],
  ['<|assistant|>', true, false], ['Put', false, true], [' the', false, true], [' egg', false, true], [' in', false, true], [' boiling', false, true],
  [' water', false, true], [' for', false, true], [' 9', false, true], [' minutes,', false, true], [' then', false, true], [' cool', false, true],
  [' it.', false, true], ['<|end|>', true, true],
]

const chipColor = ([, special, counts]: [string, boolean, boolean]) => (special ? 'var(--c5)' : counts ? 'var(--c4)' : 'var(--c8)')
const startHistory = (): Point[] => [{ x: 0, y: PostTraining.sftLoss(PostTraining.baseLogits) }]

export default function SFT() {
  const [showMask, setShowMask] = useState(true)
  const [logits, setLogits] = useState<number[]>(PostTraining.baseLogits)
  const [history, setHistory] = useState<Point[]>(startHistory)
  const [lr, setLr] = useState(0.5)
  const p = Num.softmax(logits)

  const train = (n: number) => {
    let l = logits
    const add: Point[] = []
    for (let i = 0; i < n; i++) {
      l = PostTraining.sftStep(l, lr)
      add.push({ x: history.length + i, y: PostTraining.sftLoss(l) })
    }
    setLogits(l)
    setHistory([...history, ...add])
  }

  return (
    <Page id="sft">
      <P>A base model has read the internet, but it's an autocomplete. Give it *"how do i boil an egg?"* and a perfectly reasonable continuation is *"how do i fry an egg? how do i poach an egg?…"* — that's what lists of questions on the web look like.</P>
      <P>**Supervised fine-tuning (SFT)** fixes this by continuing training on a much smaller set of example conversations, written or curated by people, showing the behaviour we want. Same loop, same loss — with one twist.</P>

      <Card title="Chat template and loss mask" subtitle="Conversations are flattened into one token stream with special role tokens">
        <Toggle id="sft-mask" label="Show which tokens count toward the loss" checked={showMask} onChange={setShowMask} />
        <div class="chips" style={{ gap: '3px' }}>
          {template.map((t, i) => (
            <span key={i} class="chip" style={{ '--chip': chipColor(t), fontWeight: t[1] ? 700 : 400, fontSize: '13px', opacity: showMask && !t[2] ? 0.45 : 1 }}>
              {t[0].replaceAll(' ', '␣')}
            </span>
          ))}
        </div>
        {showMask && (
          <Legend items={[
            { name: "weight 0: the model reads these but isn't graded on them", color: 'color-mix(in srgb, var(--c8) 45%, var(--panel))' },
            { name: 'weight 1: the model is trained to produce these', color: 'color-mix(in srgb, var(--c4) 60%, var(--panel))' },
          ]} />
        )}
        <Note>{"The twist: the loss is only counted on the **assistant's** tokens. We don't want the model to learn to write user questions — only how to answer them. The `<|end|>` token teaches it when to stop talking."}</Note>
      </Card>

      <OwnModelTuning />

      <Card title="A simpler picture: 8 whole answers" subtitle={`A toy model that chooses between 8 whole answers to "${PostTraining.prompt}". The update is the real cross-entropy gradient.`}>
        <div class="cols">
          <div>
            <HBarChart domain={[0, 1]} format={(v) => `${Math.round(v * 100)}%`} rowHeight={32}
              bars={PostTraining.answers.map((a) => ({ label: a.short, value: p[a.id], color: PostTraining.demonstrations.includes(a.id) ? 'var(--c4)' : 'color-mix(in srgb, var(--c8) 50%, var(--panel))' }))} />
            <Note>Pink = answers that appear in the human-written demonstrations</Note>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', minWidth: 0 }}>
            <div class="row">
              <Btn onClick={() => train(1)}>Train 1 step</Btn>
              <Btn primary onClick={() => train(25)}>Train 25 steps</Btn>
              <Btn onClick={() => { setLogits(PostTraining.baseLogits); setHistory(startHistory()) }}>Reset to base model</Btn>
            </div>
            <Slider id="sft-lr" label={`Learning rate ${lr.toFixed(2)}`} value={lr} min={0.05} max={2} onInput={setLr} />
            <LineChart height={140} xLabel="SFT steps" yLabel="SFT loss" xDomain={[0, Math.max(10, history.length - 1)]}
              series={[{ name: 'SFT loss', color: 'var(--c4)', points: history }]} />
            <div class="stats">
              <Stat label="SFT loss" value={PostTraining.sftLoss(logits).toFixed(2)} tone="var(--c4)" />
              <Stat label="Chance of a good answer" value={`${Math.round((p[1] + p[2]) * 100)}%`} tone="var(--good)" />
            </div>
          </div>
        </div>
        <Note>Before SFT the toy base model mostly "keeps listing questions" or states an off-topic fact. A few dozen steps on the demonstrations move almost all of the probability onto assistant-style answers.</Note>
      </Card>

      <KeyIdea>SFT doesn't teach much new knowledge — the base model already has it. It teaches **format and behaviour**: you are an assistant, this is a conversation, answer the question, then stop. A surprisingly small dataset (thousands to a few million examples) is enough.</KeyIdea>

      <GoDeeper title="practical details">
        <Formula>{'loss = − Σ_t mask_t · log p(x_{t+1} | x_≤t)  /  Σ_t mask_t'}</Formula>
        <P>**Data:** InstructGPT (2022) used about 13,000 human-written demonstrations. Modern pipelines mix human-written, model-generated-then-filtered, and task-specific data: coding, math, tool use, safety refusals and multi-turn chat.</P>
        <P>**Learning rate** is much smaller than in pretraining, and training is short (a few epochs), to avoid *catastrophic forgetting* — overwriting what pretraining learned.</P>
        <P>**LoRA** and other parameter-efficient methods freeze the big weight matrices and train small low-rank add-ons instead. The results are often almost as good, with a fraction of the memory.</P>
        <P>**The limit of SFT:** it can only imitate the demonstrations. It never sees *bad* answers, so it can't learn what to avoid, and the model can never exceed its teachers. That is what preference training is for.</P>
      </GoDeeper>
    </Page>
  )
}

// ---------------------------------------------------------------- fine-tune the reader's own model

/** Run a generator in slices of at most about `budget` ms (always at least one unit of work),
 *  yielding to the page in between. */
function runSliced(job: Generator<unknown, void, unknown>, done: () => void, budget = 12) {
  const tick = () => {
    const t0 = performance.now()
    for (;;) {
      const s = performance.now()
      if (job.next().done) { done(); return }
      const now = performance.now()
      if (now - t0 + (now - s) > budget) break   // another unit this size wouldn't fit
    }
    setTimeout(tick, 0)
  }
  setTimeout(tick, 0)
}

/** ChatTuner (engine/chatTuning.ts) with the same data, seeds, optimizer and learning rate, but each
 *  step is split into its 8 rows: gradients from loss/8 per row add up to the gradient of the batch
 *  mean, so the step is the same while no slice of work is longer than one row's forward or backward pass. */
class SlicedTuner {
  model: TinyGPT
  step = 0
  learningRate = 3e-4
  private optimizer = new Optimizer('AdamW')
  private rng = new Rng(11)
  constructor(base: TinyGPT) { this.model = base.copy() }

  /** One SFT step on a batch of 8 demonstrations; returns the masked loss. Yields after each pass. */
  *trainStep(): Generator<void, number, unknown> {
    const params = this.model.parameters, n = 8
    const rows = Array.from({ length: n }, () => trainingRow(demonstrations[this.rng.int(demonstrations.length)], this.model.config.context))
    for (const p of params) p.zeroGrad()
    let total = 0
    for (const r of rows) {
      const loss = Ops.crossEntropy(this.model.forward(r.ids), r.targets, r.mask)
      total += loss.item
      yield
      Ops.scale(loss, 1 / n).backward()
      yield
    }
    clipGradients(params, 1)
    this.optimizer.step(params, this.learningRate)
    this.step++
    return total / n
  }

  /** Average masked loss over all demonstrations (no update). Yields after each one. */
  *demonstrationLoss(): Generator<void, number, unknown> {
    let sum = 0
    for (const e of demonstrations) {
      const r = trainingRow(e, this.model.config.context)
      sum += Ops.crossEntropy(this.model.forward(r.ids), r.targets, r.mask).item
      yield
    }
    return sum / demonstrations.length
  }
}

/** chatTuning.answer, one forward pass per slice: greedy answer until a newline. */
function* answerSliced(model: TinyGPT, question: string, maxLength = 40): Generator<void, string, unknown> {
  const ids = tokenizer.encode(tokenizer.clean(question) + '\n')
  const newline = tokenizer.encode('\n')[0]
  const out: number[] = []
  for (let i = 0; i < maxLength; i++) {
    const next = Num.argmax(model.nextTokenProbs(ids.slice(-model.config.context)))
    yield
    if (next === newline) break
    out.push(next); ids.push(next)
  }
  return tokenizer.decode(out)
}

interface Row { question: string; before: string; after: string | null }

/** What the card shows while fine-tuning runs (same as ChatTuningSession in Swift). It lives at module
 *  level, so leaving the page and coming back finds it where it was; work runs in short time slices. */
class TuningSession {
  rows: Row[] = []
  loss: Point[] = []
  step = 0
  isRunning = false
  private baseVersion = -1
  private tuner: SlicedTuner | null = null
  private generation = 0
  private running = false
  private loopActive = false
  private queue: Generator<unknown, void, unknown>[] = []
  private busy = false
  private listeners = new Set<() => void>()

  subscribe(fn: () => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn) } }
  private emit() { this.listeners.forEach((f) => f()) }

  get subjectCount() { return this.rows.filter((r) => r.after !== null && startsWithSubject(r.after, r.question)).length }
  get verbCount() { return this.rows.filter((r) => r.after !== null && usesVerb(r.after, r.question)).length }
  get finished() { return this.step >= SUGGESTED_STEPS }

  /** Jobs run one after another, like the serial queue in the Mac app. */
  private enqueue(job: Generator<unknown, void, unknown>) {
    this.queue.push(job)
    this.pump()
  }
  private pump() {
    if (this.busy) return
    const job = this.queue.shift()
    if (!job) return
    this.busy = true
    runSliced(job, () => { this.busy = false; this.pump() })
  }

  /** Start over from the reader's current model (called when the page opens or the model changes). */
  prepare(base: TinyGPT, version: number) {
    if (version === this.baseVersion) return
    this.stop()
    const gen = ++this.generation
    this.queue = []
    this.loopActive = false
    this.baseVersion = version
    const t = new SlicedTuner(base)
    this.tuner = t
    this.step = 0
    this.loss = []
    this.rows = heldOutQuestions.map((q) => ({ question: q, before: '…', after: null }))
    this.emit()
    const self = this
    this.enqueue((function* () {
      const l: number = yield* t.demonstrationLoss()
      const before: string[] = []
      for (const q of heldOutQuestions) { if (self.generation !== gen) return; before.push(yield* answerSliced(t.model, q)) }
      if (self.generation !== gen) return
      self.loss = [{ x: 0, y: l }]
      self.rows = self.rows.map((r, i) => ({ ...r, before: before[i] }))
      self.emit()
    })())
  }

  restart(base: TinyGPT, version: number) {
    this.baseVersion = -1
    this.prepare(base, version)
  }

  start() {
    const t = this.tuner
    if (!t || this.isRunning || this.finished) return
    this.isRunning = true
    this.running = true
    this.emit()
    if (this.loopActive) return   // the loop from before a pause is still finishing; it carries on
    this.loopActive = true
    const gen = this.generation, questions = this.rows.map((r) => r.question), self = this
    this.enqueue((function* () {
      while (self.running && t.step < SUGGESTED_STEPS && self.generation === gen) {
        const losses: Point[] = []
        for (let i = 0; i < 5; i++) { const x = t.step + 1; losses.push({ x, y: yield* t.trainStep() }) }
        const answers: string[] = []
        for (const q of questions) answers.push(yield* answerSliced(t.model, q))
        if (self.generation !== gen) return
        self.step = t.step
        self.loss = [...self.loss, ...losses]
        self.rows = self.rows.map((r, i) => ({ ...r, after: answers[i] }))
        self.emit()
      }
      if (self.generation !== gen) return
      self.loopActive = false
      self.isRunning = false
      self.emit()
    })())
  }

  stop() {
    this.running = false
    if (this.isRunning) { this.isRunning = false; this.emit() }
  }
}

const session = new TuningSession()

function useSession() {
  const [, setTick] = useState(0)
  useEffect(() => session.subscribe(() => setTick((n) => n + 1)), [])
  return session
}

/** SFT on the reader's own tiny model: teach it to answer "what does … <verb>" questions. */
function OwnModelTuning() {
  const t = useTrainer()
  const s = useSession()
  useEffect(() => { if (t.isTrained) s.prepare(t.snapshot, t.snapshotVersion) }, [t.isTrained])
  // A newer model only replaces the session before fine-tuning starts; after that, Start over picks it up.
  useEffect(() => { if (t.isTrained && !t.isRunning && s.step === 0 && !s.isRunning) s.prepare(t.snapshot, t.snapshotVersion) }, [t.snapshotVersion])
  const e = demonstrations[0]
  const lastLoss = s.loss[s.loss.length - 1]

  return (
    <Card title="Live: fine-tune your own model" subtitle={`Your model has only ever seen sentences, never a question. ${demonstrations.length} demonstrations show it how to answer one.`}>
      {!t.isTrained ? (
        <>
          <ModelBanner />
          <Note>Fine-tuning starts from a pretrained model. Train it first (about a minute), then come back.</Note>
        </>
      ) : (
        <>
          <div class="cols">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', minWidth: 0 }}>
              <p class="sft-label">A demonstration</p>
              <div class="sft-demo"><span class="q">{e.question + ' ↵ '}</span><span class="a">{e.answer}</span></div>
              <Note>Grey = the question (read, not graded). Pink = the answer (graded). The newline marks where the answer starts, standing in for the role tokens above.</Note>
              <div class="row" style={{ marginTop: '6px' }}>
                <Btn primary onClick={() => (s.isRunning ? s.stop() : s.start())} disabled={s.finished}>
                  {s.isRunning ? '❚❚ Pause' : s.step === 0 ? `▶ Fine-tune ${SUGGESTED_STEPS} steps` : '▶ Resume'}
                </Btn>
                <Btn onClick={() => s.restart(t.snapshot, t.snapshotVersion)} disabled={s.isRunning || s.step === 0}>Start over</Btn>
              </div>
              <div aria-label={`Fine-tuning loss: ${lastLoss ? `${lastLoss.y.toFixed(2)} after ${lastLoss.x} steps` : 'not started'}`} role="group">
                <LineChart height={130} xDomain={[0, SUGGESTED_STEPS]} xLabel="fine-tuning steps" yLabel="loss on answers" legend={false}
                  series={[{ name: 'loss on answers', color: 'var(--c4)', points: s.loss }]} />
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', minWidth: 0 }}>
              <p class="sft-label">Questions it never saw</p>
              <table class="sft-table">
                <thead>
                  <tr><th scope="col"><span class="sft-sr">Question</span></th><th scope="col">Before</th><th scope="col">{s.step === 0 ? 'After' : `After ${s.step} steps`}</th></tr>
                </thead>
                <tbody>
                  {s.rows.map((r) => (
                    <tr key={r.question}>
                      <td>{r.question}</td>
                      <td class="muted"><span class="sft-clamp">{r.before}</span></td>
                      <td class={r.after !== null && startsWithSubject(r.after, r.question) ? '' : 'muted'}><span class="sft-clamp">{r.after ?? '—'}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div class="stats">
                <Stat label="Answers about the right subject" value={`${s.subjectCount} of ${s.rows.length}`} tone="var(--c4)" />
                <Stat label="Also using the right verb" value={`${s.verbCount} of ${s.rows.length}`} tone="var(--good)" />
              </div>
            </div>
          </div>
          <Note>Expect it to learn the **format** fast (answer on a new line, with a sentence about the question's subject) and the **verb** less reliably. A 28k-weight model has to copy words from 20 characters back, which is hard at its size. Real SFT shows the same split: format and style come quickly, new skills don't.</Note>
        </>
      )}
    </Card>
  )
}
