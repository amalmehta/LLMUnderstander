import { useRef, useState } from 'preact/hooks'
import { DIFFICULTIES, Group, HEAD_ACCURACY, makeProblem, MAX_THINKING, ReasoningToy, working } from '../engine/reasoning'
import { Rng } from '../engine/tensor'
import { LineChart } from '../ui/charts'
import { Btn, Card, Formula, GoDeeper, KeyIdea, Note, P, Page, Stat, useTicker } from '../ui/kit'
import './Reasoning.css'

const MAX_UPDATES = 300
const SEED = 42
const PURPLE = 'var(--c5)'
const steps = (k: number) => `${k} step${k === 1 ? '' : 's'}`
const argmax = (p: number[]) => p.reduce((b, v, i) => (v > p[b] ? i : b), 0)
const signed = (v: number) => (v >= 0 ? '+' : '') + v.toFixed(1)

interface HistPoint { x: number; accuracy: number; length: number }

export default function Reasoning() {
  const toy = useRef<ReasoningToy>(null as unknown as ReasoningToy)
  if (!toy.current) toy.current = new ReasoningToy()
  const rng = useRef(new Rng(SEED))
  const history = useRef<HistPoint[]>([])
  if (history.current.length === 0) history.current.push({ x: 0, accuracy: toy.current.expectedAccuracy, length: toy.current.expectedLength })
  const [updates, setUpdates] = useState(0)
  const [silentCount, setSilentCount] = useState(0)
  const [lastGroup, setLastGroup] = useState<Group | null>(null)
  const [running, setRunning] = useState(false)
  const [lengthCost, setLengthCost] = useState(0)
  const [example, setExample] = useState([7, 5, 9, 3])

  /** Runs `n` GRPO updates (capped at 300) and records each one. */
  const advance = (n: number) => {
    const t = toy.current
    let done = updates, silent = 0, g: Group | null = null
    for (let i = 0; i < n && done < MAX_UPDATES; i++) {
      g = t.step(rng.current)
      if (g.silent) silent++
      done++
      history.current.push({ x: done, accuracy: t.expectedAccuracy, length: t.expectedLength })
    }
    if (!g) return
    setUpdates(done)
    setSilentCount((s) => s + silent)
    setLastGroup(g)
    if (done >= MAX_UPDATES) setRunning(false)
  }

  useTicker(running && updates < MAX_UPDATES, 40, () => advance(3))

  const reset = () => {
    setRunning(false)
    toy.current = new ReasoningToy()
    toy.current.lengthCost = lengthCost
    rng.current = new Rng(SEED)
    history.current = [{ x: 0, accuracy: toy.current.expectedAccuracy, length: toy.current.expectedLength }]
    setUpdates(0); setSilentCount(0); setLastGroup(null)
  }

  const t = toy.current
  const h = history.current

  return (
    <Page id="reasoning">
      <P>Preference tuning rewards answers people *like*. For maths, code and puzzles there's something better: a **checker**. Did the number match? Did the tests pass? Rewarding only answers the checker accepts is called **RL with verifiable rewards**, and it's how today's reasoning models are trained.</P>
      <P>Nobody tells the model to show its working. It discovers that writing out intermediate steps before answering (a **chain of thought**) gets more answers right, so RL makes it do that more, and its answers get longer.</P>

      <Card title="Live: GRPO on a toy reasoning model" subtitle={`Sums of 2–6 digits. Before answering, the model picks how many additions to write out; the rest it does in its head, where each is right only ${Math.round(HEAD_ACCURACY * 100)}% of the time.`}>
        <div class="rsn-controls">
          <Btn primary onClick={() => setRunning(!running)} disabled={updates >= MAX_UPDATES}>
            <span aria-hidden="true">{running ? '❚❚ ' : '▶ '}</span>{running ? 'Pause' : updates === 0 ? 'Train with GRPO' : 'Resume'}
          </Btn>
          <Btn onClick={() => advance(1)} disabled={running || updates >= MAX_UPDATES}>One update</Btn>
          <Btn onClick={reset}><span aria-hidden="true">↺ </span>Reset</Btn>
          <span class="grow" style={{ flexBasis: '20px' }} />
          <label class="rsn-cost" for="reasoning-cost">
            <span>Thinking cost: {lengthCost.toFixed(2)} per step</span>
            <input id="reasoning-cost" type="range" min={0} max={0.15} step="any" value={lengthCost} aria-label="Thinking cost per step"
              onInput={(e) => { const v = +(e.currentTarget as HTMLInputElement).value; toy.current.lengthCost = v; setLengthCost(v) }} />
          </label>
        </div>
        <div class="stats">
          <Stat label="Updates" value={String(updates)} tone="var(--sec)" />
          <Stat label="Accuracy" value={`${Math.round(t.expectedAccuracy * 100)}%`} tone="var(--good)" />
          <Stat label="Average thinking" value={`${t.expectedLength.toFixed(1)} steps`} tone={PURPLE} />
          <Stat label="Groups with no signal" value={updates === 0 ? '—' : `${silentCount} of ${updates}`} tone="var(--c8)" />
        </div>
        <div class="cols">
          <div>
            <p class="rsn-sub">Over training</p>
            <LineChart height={170} xDomain={[0, MAX_UPDATES]} yDomain={[0, 1]} xLabel="GRPO updates" yFormat={(v) => `${Math.round(v * 100)}%`}
              series={[
                { name: 'accuracy', color: 'var(--good)', points: h.map((p) => ({ x: p.x, y: p.accuracy })) },
                { name: `thinking length (share of the ${MAX_THINKING}-step maximum)`, color: PURPLE, points: h.map((p) => ({ x: p.x, y: p.length / MAX_THINKING })) },
              ]} />
          </div>
          <div>
            <p class="rsn-sub">How long it thinks, by difficulty</p>
            <PolicyGrid toy={t} />
            <Note>Darker = more likely. The outlined box is one step per addition, just enough.</Note>
          </div>
        </div>
        {lastGroup && <GroupView g={lastGroup} />}
      </Card>

      <ExampleCard toy={t} updates={updates} example={example} onNew={() => setExample(makeProblem(new Rng(1 + Math.floor(Math.random() * 2 ** 52))))} />

      <KeyIdea>Reward the result, not the method, and let the model find the method. Thinking longer isn't taught; it's what earns reward when problems need several steps.</KeyIdea>

      <GoDeeper title="GRPO and its relatives">
        <Formula>{'advantage_i = (reward_i − mean of the group) ÷ standard deviation of the group\nupdate ∝ Σ_i advantage_i · ∇ log π(answer_i)   (minus a small KL penalty to the starting model)'}</Formula>
        <P>**GRPO** (group relative policy optimisation) was introduced with DeepSeekMath (Shao et al., 2024) and used to train DeepSeek-R1 (2025). Older RLHF used PPO, which needs a second network, a *value model*, to estimate how good each situation is. GRPO replaces it with the group: sample several answers to the same problem, and the group's average *is* the baseline. Cheaper, and simple.</P>
        <P>**No signal from easy or impossible problems.** If all 8 answers are right (or all wrong), every advantage is 0 and nothing is learned. Training sets are filtered toward problems the model gets right *sometimes*. The counter above shows how often this happens here.</P>
        <P>**Outcome vs process rewards.** We reward only the final answer (an *outcome* reward). A *process* reward grades each step, which gives more signal but needs a grader for steps, and models learn to please that grader.</P>
        <P>**Overthinking.** More thinking costs time and money. Raise the thinking cost and the policy settles on the shortest working that still gets the answer: here, one step per addition.</P>
        <P>**What's simplified.** A real model chooses every token, not one number, and its "steps" are free-form text. The update maths is the same: sample a group, score, compare with the group, push up what beat it.</P>
      </GoDeeper>
    </Page>
  )
}

function PolicyGrid({ toy }: { toy: ReasoningToy }) {
  const ks = Array.from({ length: MAX_THINKING + 1 }, (_, k) => k)
  return (
    <div class="scroll-x">
      <div class="rsn-grid">
        <div class="rsn-row" aria-hidden="true">
          <span class="lbl" />
          {ks.map((k) => <span key={k} class="head">{k}</span>)}
        </div>
        {DIFFICULTIES.map((d) => {
          const p = toy.policy(d), best = argmax(p)
          return (
            <div key={d} class="rsn-row" role="img" aria-label={`${d} addition${d === 1 ? '' : 's'}: most likely ${best} thinking steps, ${Math.floor(p[best] * 100)} percent`}>
              <span class="lbl" aria-hidden="true">{d} addition{d === 1 ? '' : 's'}</span>
              {ks.map((k) => (
                <span key={k} aria-hidden="true" class={`rsn-cell${k === d ? ' just' : ''}`}
                  style={{ background: `color-mix(in srgb, ${PURPLE} ${(8 + 90 * p[k]).toFixed(1)}%, transparent)` }} />
              ))}
            </div>
          )
        })}
        <span class="axis" aria-hidden="true">thinking steps →</span>
      </div>
    </div>
  )
}

function GroupView({ g }: { g: Group }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <div class="rsn-group-head">
        <b>Latest group: {g.problem.join(' + ')} = {g.problem.reduce((a, b) => a + b, 0)}</b>
        {g.silent && <span class="rsn-silent">all 8 scored the same → advantage 0 → no update</span>}
      </div>
      <div class="rsn-answers">
        {g.answers.map((a, i) => (
          <div key={i} class="rsn-answer" role="img" aria-label={`${a.thinking} steps, ${a.correct ? 'right' : 'wrong'}, advantage ${a.advantage.toFixed(1)}`}>
            <span class="mark" aria-hidden="true" style={{ color: a.correct ? 'var(--good)' : 'var(--muted)' }}>{a.correct ? '✓' : '✗'}</span>
            <span aria-hidden="true">{steps(a.thinking)}</span>
            <span class="adv" aria-hidden="true" style={{ color: a.advantage > 0 ? 'var(--good)' : a.advantage < 0 ? 'var(--sec)' : 'var(--muted)' }}>{signed(a.advantage)}</span>
          </div>
        ))}
      </div>
      <Note>Bottom number = advantage: how much better than its group's average. Positive pushes that thinking length up; negative pushes it down.</Note>
    </div>
  )
}

function ExampleCard({ toy, updates, example, onNew }: { toy: ReasoningToy; updates: number; example: number[]; onNew: () => void }) {
  const d = example.length - 1
  const k = argmax(toy.policy(d))
  const truth = example.reduce((a, b) => a + b, 0)
  const w = working(example, k, new Rng(Math.floor(updates / 10) + truth))
  return (
    <Card title="What its answer looks like right now" subtitle="The model's most likely thinking length for this problem, written out">
      <div class="rsn-example">
        <div class="rsn-work" aria-live="polite">
          <span class="q">Q: {example.join(' + ')} = ?</span>
          {w.lines.length === 0 && <span style={{ color: 'var(--muted)' }}>(answers straight away)</span>}
          {w.lines.map((l) => <span key={l} class="line">{'  ' + l}</span>)}
          <span class="a" style={{ color: w.correct ? 'var(--good)' : 'var(--sec)' }}>A: {w.answer}  {w.correct ? '✓' : `✗ (true answer ${truth})`}</span>
        </div>
        <Btn onClick={onNew}>New problem</Btn>
      </div>
    </Card>
  )
}
