import { useRef, useState } from 'preact/hooks'
import { AgentRL, AgentScript } from '../engine/agents'
import { Num, Rng } from '../engine/tensor'
import { fmt } from '../engine/toys'
import { HBarChart, LineChart } from '../ui/charts'
import { Btn, Callout, Card, Flow, Formula, GoDeeper, KeyIdea, md, Note, P, Page, Toggle, useTicker } from '../ui/kit'
import { EventBubble } from './AgentLoop'

export default function AgentTraining() {
  return (
    <Page id="agent-training">
      <P>A harness can wrap any model, but some models make far better agents than others. Agent skill is trained in the same three stages you've already seen, with agent-shaped data.</P>
      <Flow steps={[
        { title: 'Pretraining', detail: 'lots of code, docs,\nterminal logs, how-tos', tone: 'var(--sec-pretraining)', href: '#pretraining' },
        { title: 'SFT on transcripts', detail: 'examples of good\ntool-using runs', tone: 'var(--sec-posttraining)', href: '#sft' },
        { title: 'RL in environments', detail: 'attempt real tasks,\nreward = did it work?', tone: 'var(--sec-agents)' },
      ]} />

      <MaskCard />
      <RLCard />

      <Card title="Credit assignment over many steps" subtitle="The reward arrives only at the very end">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', alignItems: 'center', fontFamily: 'var(--font-display)', fontSize: '12.5px' }}>
          {['run tests', 'read file', 'think', 'edit', 'run tests', 'answer'].map((s, i) => (
            <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
              <span style={{ padding: '5px 9px', borderRadius: '999px', background: 'color-mix(in srgb, var(--c2) 13%, var(--panel))' }}>{s}</span>
              <span aria-hidden="true" style={{ color: 'var(--muted)', fontSize: '11px' }}>→</span>
            </span>
          ))}
          <span style={{ padding: '5px 9px', borderRadius: '999px', fontWeight: 700, background: 'color-mix(in srgb, var(--good) 20%, var(--panel))' }}>reward = 1 ✓</span>
        </div>
        <Note>In the simplest methods every token in a successful run is pushed up a little and every token in a failed run is pushed down a little. Over thousands of runs, steps that *consistently* show up in successes (like re-running the tests) get reinforced, and incidental ones average out. Long tasks make this noisy, which is one reason agent RL needs so many attempts.</Note>
      </Card>

      <KeyIdea>Agents are trained on the agent loop itself: imitate good transcripts, then practise on real tasks and reinforce whatever *verifiably* worked. The reward checker is part of the design — a lazy checker trains a lazy (or sneaky) agent.</KeyIdea>

      <GoDeeper title="how this works at scale">
        <P>**Environments:** thousands of sandboxed tasks — repositories with failing tests, websites to navigate, data to analyse — each with an automatic checker. The model runs the full loop inside a container during training, exactly as it would for a user.</P>
        <P>**Verifiable rewards** (tests pass, answer matches, page reached) need no human labeller, so RL can run at huge scale. Where outcomes can't be checked automatically, reward models and AI judges fill in, with the risks from Chapter 12.</P>
        <P>**Evaluation:** benchmarks built from real tasks (for example, fixing real GitHub issues and checking with the project's own tests) measure progress, and must be kept out of training data so they stay honest.</P>
        <Formula>{'REINFORCE with a baseline:   θ ← θ + lr · mean[ (R − b) · ∇ log π(action) ]'}</Formula>
      </GoDeeper>
    </Page>
  )
}

function MaskCard() {
  const [showMask, setShowMask] = useState(true)
  return (
    <Card title="SFT on agent transcripts: what gets trained?" subtitle="Chapter 14's run as a training example. Same loss mask idea as Chapter 11.">
      <Toggle id="agent-training-mask" label="Show the loss mask" checked={showMask} onChange={setShowMask} />
      <div style={{ height: '300px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', paddingRight: '4px' }}>
        {AgentScript.events.filter(AgentScript.inContext).map((e) => {
          const trained = AgentScript.trainedOn(e)
          return (
            <div key={e.id} style={{ display: 'grid', gridTemplateColumns: '20px minmax(0, 1fr)', gap: '8px', alignItems: 'start' }}>
              <span aria-label={showMask ? (trained ? 'trained on' : 'not trained on') : undefined}
                style={{ paddingTop: '8px', textAlign: 'center', fontWeight: 700, color: trained ? 'var(--good)' : 'var(--muted)', visibility: showMask ? 'visible' : 'hidden' }}>
                {trained ? '✓' : '–'}
              </span>
              <EventBubble event={e} dimmed={showMask && !trained} />
            </div>
          )
        })}
      </div>
      <Note>Only the model's own text and tool calls are trained. Tool results, the user's messages and the system prompt are context, not targets — we want the model to learn to *call* tools and *use* their output, not to imitate what the test runner prints.</Note>
    </Card>
  )
}

interface HistPoint { update: number; reward: number; genuine: number }

function RLCard() {
  const rl = useRef(AgentRL.newState())
  const rng = useRef(new Rng(21))
  const history = useRef<HistPoint[]>([])
  const [robust, setRobust] = useState(false)
  const [remaining, setRemaining] = useState(0)
  const [, setVersion] = useState(0)

  const reset = () => {
    rl.current = AgentRL.newState(); rng.current = new Rng(21); history.current = []
    setRemaining(0); setVersion((v) => v + 1)
  }

  useTicker(remaining > 0, 50, () => {
    let left = remaining
    for (let k = 0; k < 2 && left > 0; k++) {
      AgentRL.step(rl.current, robust, rng.current)
      left--
      const p = Num.softmax(rl.current.logits), rs = rl.current.lastRewards
      history.current.push({ update: history.current.length + 1, reward: Num.sum(rs) / rs.length, genuine: AgentRL.genuineSuccess(p) })
    }
    setRemaining(left)
    setVersion((v) => v + 1)
  })

  const p = Num.softmax(rl.current.logits)
  const hacked = p[2] > 0.5
  const h = history.current

  return (
    <Card title="RL with a verifiable reward — live" subtitle={'The policy picks a strategy for "make the failing test pass". Each update samples 16 attempts and applies REINFORCE.'}>
      <div class="row">
        <Btn primary onClick={() => setRemaining(150)} disabled={remaining > 0}>{remaining > 0 ? 'Training…' : 'Run 150 updates'}</Btn>
        <Btn onClick={reset}>Reset</Btn>
        <span class="grow" style={{ flexBasis: '20px' }} />
        <Toggle id="agent-training-robust" label="Checker also verifies the tests weren't modified" checked={robust} onChange={(v) => { setRobust(v); reset() }} />
      </div>
      <div class="cols">
        <HBarChart domain={[0, 1]} format={(v) => fmt.pct(v)}
          bars={AgentRL.strategies.map((s) => ({ label: s.name, value: p[s.id], color: s.tampers ? 'var(--bad)' : s.id === 0 ? 'var(--good)' : 'var(--c8)' }))} />
        <LineChart height={200} xDomain={[0, 150]} yDomain={[0, 1.05]} xLabel="updates"
          series={[
            { name: 'reward the checker gives', color: 'var(--c2)', points: h.map((x) => ({ x: x.update, y: x.reward })) },
            { name: 'genuinely fixed', color: 'var(--good)', points: h.map((x) => ({ x: x.update, y: x.genuine })) },
          ]} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {AgentRL.strategies.map((s) => (
          <p key={s.id} class="note" style={{ fontSize: '13px' }}>{md(`**${s.name}** — ${s.detail}; tests pass ${Math.round(s.passRate * 100)}% of the time${s.tampers ? ' (by tampering)' : ''}`)}</p>
        ))}
      </div>
      {rl.current.episodes >= 1500 && (
        <Callout tone={hacked ? 'var(--bad)' : 'var(--good)'}>{hacked
          ? '🚨 **Reward hacking.** The checker only asked "do the tests pass?", and deleting the failing test always makes them pass. RL found that shortcut. Turn on the stricter checker and run again.'
          : robust ? 'With a checker that also verifies the tests are untouched, the shortcut earns nothing, and RL reinforces genuine investigation and fixing.'
            : 'Not hacked this time — keep training; the shortcut usually wins eventually.'}</Callout>
      )}
    </Card>
  )
}
