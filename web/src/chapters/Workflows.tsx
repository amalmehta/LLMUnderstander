import { Fragment } from 'preact'
import { useRef, useState } from 'preact/hooks'
import { Bandit, banditExplanation, BanditState, BanditStrategy, banditStrategies, Sampling } from '../engine/agents'
import { Rng } from '../engine/tensor'
import { fmt } from '../engine/toys'
import { HBarChart, LineChart } from '../ui/charts'
import { Btn, Card, Formula, GoDeeper, KeyIdea, md, Note, P, Page, Segmented, Slider, Stat, useTicker } from '../ui/kit'

const patterns = ['Chaining', 'Routing', 'Parallel & voting', 'Orchestrator–workers', 'Evaluator–optimizer', 'Autonomous agent'] as const
type Pattern = (typeof patterns)[number]

const steps: Record<Pattern, [string, string][]> = {
  'Chaining': [['Input', 'a topic'], ['LLM call 1', 'write an outline'], ['Check', 'code: outline has 5 parts?'], ['LLM call 2', 'write each part'], ['LLM call 3', 'translate'], ['Output', '']],
  'Routing': [['Input', 'support ticket'], ['Router LLM', 'classify: billing / bug / question'], ['Specialist', 'a prompt + tools for that type'], ['Output', '']],
  'Parallel & voting': [['Input', 'one question'], ['N calls at once', 'same prompt (voting) or\ndifferent sub-parts (sectioning)'], ['Combine', 'majority vote, or merge'], ['Output', '']],
  'Orchestrator–workers': [['Input', 'a big task'], ['Orchestrator LLM', 'plans and splits the work'], ['Workers', 'sub-agents, each with\nits own clean context'], ['Orchestrator', 'merges results'], ['Output', '']],
  'Evaluator–optimizer': [['Input', 'a draft request'], ['Generator LLM', 'writes a draft'], ['Evaluator', 'LLM or tests give feedback'], ['Loop', 'revise until it passes'], ['Output', '']],
  'Autonomous agent': [['Input', 'an open-ended goal'], ['Model', 'decides the next step'], ['Tools', 'act, observe the result'], ['Repeat', "the model decides when it's done"], ['Output', '']],
}

const when: Record<Pattern, string> = {
  'Chaining': '**Use when** a task splits cleanly into fixed steps. Each call is easier than the whole, and code can check the output between steps. **Cost:** more calls in sequence, so slower.',
  'Routing': '**Use when** inputs fall into distinct kinds that need different handling. Easy questions can even go to a smaller, cheaper model. **Cost:** a misrouted input gets the wrong specialist.',
  'Parallel & voting': '**Use when** sub-tasks are independent (sectioning), or when you want more reliability by asking several times and taking a vote. See the calculator below. **Cost:** N× the tokens.',
  'Orchestrator–workers': "**Use when** you can't know the sub-tasks in advance (\"change every file that uses this API\"). The plan is decided by the model at run time. **Cost:** more tokens and more coordination to get right.",
  'Evaluator–optimizer': '**Use when** there are clear criteria and feedback measurably helps, like code that must pass tests or a translation that must keep its meaning. **Cost:** loops can run long; cap the rounds.',
  'Autonomous agent': '**Use when** the path really can\'t be predicted and you trust the model to steer (Chapters 14–17). The most flexible pattern and the hardest to make reliable. Start with the simplest pattern that works.',
}

const stepColors = ['var(--c8)', 'var(--c3)', 'var(--c6)', 'var(--c1)', 'var(--c5)', 'var(--c8)']

const strategyColors: Record<BanditStrategy, string> = {
  'Greedy': 'var(--c7)', 'ε-greedy (10%)': 'var(--c2)', 'UCB1': 'var(--c1)', 'Thompson sampling': 'var(--c3)',
}

export default function Workflows() {
  return (
    <Page id="workflows">
      <P>Not every useful system is a free-running agent. Often the best design is a **workflow**: model calls wired together in a fixed pattern by ordinary code. And whenever a system can choose between options — prompts, models, tools, strategies — it faces the oldest problem in RL: **explore or exploit?**</P>

      <PatternsCard />
      <VotingCard />
      <BanditCard />

      <KeyIdea>Use the simplest pattern that does the job: one call, then a fixed workflow, then an agent. Whenever the system must choose between options it can't fully evaluate in advance, give it a principled way to explore — like UCB or Thompson sampling — rather than greedily trusting early results.</KeyIdea>

      <GoDeeper title="where bandits show up in LLM systems">
        <P>**Choosing prompts, models or tools online:** each user request is a pull; success (a thumbs-up, a passing test) is the reward. A bandit shifts traffic toward what works while still checking the alternatives, which is smarter than a fixed 50/50 A/B test.</P>
        <P>**Exploration in RL training:** in Chapter 17 the policy only discovered the test-deleting shortcut because it sometimes tried it. Sampling at a temperature is how language models explore. Too little and RL never finds better strategies; too much and training is noisy.</P>
        <P>**Search at inference time:** tree search over reasoning steps or code edits must decide which branch to expand next. Monte Carlo tree search does this with the UCB rule from this page.</P>
        <Formula>{'UCB1:      pick argmax_i [ mean_i + √(2 ln t / n_i) ]\nThompson:  draw θ_i ~ Beta(wins_i + 1, losses_i + 1), pick argmax_i θ_i\nRegret:    Σ_t ( best rate − rate of the arm chosen at t )'}</Formula>
      </GoDeeper>
    </Page>
  )
}

function PatternsCard() {
  const [pattern, setPattern] = useState<Pattern>('Chaining')
  const s = steps[pattern]
  const loops = pattern === 'Evaluator–optimizer' || pattern === 'Autonomous agent'
  return (
    <Card title="Six common patterns" subtitle="From most predictable to most autonomous">
      <Segmented options={patterns} value={pattern} onChange={setPattern} />
      <div class="flow">
        {s.map(([title, detail], i) => (
          <Fragment key={i}>
            <div class="step" style={{ '--tone': stepColors[Math.min(i, 5)], minHeight: '70px' }}>
              <b>{title}</b>{detail && <small>{detail}</small>}
            </div>
            {i < s.length - 1 && <span class="arrow" aria-hidden="true">{loops && i === 3 ? '↩' : '→'}</span>}
          </Fragment>
        ))}
      </div>
      <p class="prose">{md(when[pattern])}</p>
    </Card>
  )
}

function VotingCard() {
  const [p, setP] = useState(0.6)
  const [n, setN] = useState(5)
  const ns: number[] = []
  for (let k = 1; k <= 21; k += 2) ns.push(k)
  return (
    <Card title="Why ask more than once?" subtitle="Each attempt is independently right with probability p. Exact binomial maths.">
      <Slider id="workflows-p" label={`Single attempt right: ${Math.round(p * 100)}%`} value={p} min={0.05} max={0.95} onInput={setP} />
      <Slider id="workflows-n" label={`Attempts: ${n}`} value={n} min={1} max={21} step={2} onInput={setN} />
      <LineChart height={180} xDomain={[1, 21]} yDomain={[0, 1]} xLabel="attempts" yFormat={(v) => fmt.pct(v)}
        rules={[{ x: n, color: 'var(--muted)' }]}
        series={[
          { name: 'majority vote', color: 'var(--c1)', points: ns.map((k) => ({ x: k, y: Sampling.majority(p, k) })) },
          { name: 'best-of-n with a checker', color: 'var(--good)', points: ns.map((k) => ({ x: k, y: Sampling.anyCorrect(p, k) })) },
        ]} />
      <div class="stats">
        <Stat label="One attempt" value={fmt.pct(p)} tone="var(--c8)" />
        <Stat label={`Majority of ${n}`} value={fmt.pct(Sampling.majority(p, n))} tone="var(--c1)" />
        <Stat label={`Any of ${n}, checked`} value={fmt.pct(Sampling.anyCorrect(p, n))} tone="var(--good)" />
      </div>
      <Note>Voting helps only if a single attempt is right more often than not; below 50%, the majority is *more* likely to be wrong. With a reliable checker (tests, a verifier), any one success is enough. The catch: this assumes independent attempts, and real attempts from the same model tend to make the same mistakes.</Note>
    </Card>
  )
}

const TOTAL = 600
const freshStates = () => Object.fromEntries(banditStrategies.map((s) => [s, Bandit.newState()])) as Record<BanditStrategy, BanditState>
const freshRngs = (seed: number) => Object.fromEntries(banditStrategies.map((s, i) => [s, new Rng(seed * 1000 + i)])) as Record<BanditStrategy, Rng>

function BanditCard() {
  const seed = useRef(1)
  const states = useRef(freshStates())
  const rngs = useRef(freshRngs(1))
  const [remaining, setRemaining] = useState(0)
  const [focus, setFocus] = useState<BanditStrategy>('Thompson sampling')
  const [, setVersion] = useState(0)

  const resetBandit = (newSeed: boolean) => {
    if (newSeed) seed.current += 1
    states.current = freshStates()
    rngs.current = freshRngs(seed.current)
  }

  useTicker(remaining > 0, 30, () => {
    let left = remaining
    for (let k = 0; k < 6 && left > 0; k++) {
      for (const s of banditStrategies) Bandit.step(s, states.current[s], rngs.current[s])
      left--
    }
    setRemaining(left)
    setVersion((v) => v + 1)
  })

  const tries = states.current[banditStrategies[0]].t
  const st = states.current[focus]
  const maxPulls = Math.max(1, ...st.pulls)

  return (
    <Card title="Multi-armed bandit: explore or exploit?" subtitle={`Five prompts with hidden success rates. Four strategies each get ${TOTAL} tries. Regret = successes lost by not always using the best prompt.`}>
      <div class="row">
        <Btn primary disabled={remaining > 0} onClick={() => { if (tries >= TOTAL) resetBandit(false); setRemaining(TOTAL); setVersion((v) => v + 1) }}>
          {remaining > 0 ? 'Running…' : `Run ${TOTAL} tries`}
        </Btn>
        <Btn onClick={() => { resetBandit(true); setRemaining(TOTAL); setVersion((v) => v + 1) }}>New random run</Btn>
        <span class="grow" style={{ flexBasis: '20px' }} />
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '14px', fontVariantNumeric: 'tabular-nums' }}>Tries: {tries}</span>
      </div>
      <div class="cols">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', minWidth: 0 }}>
          <b style={{ fontFamily: 'var(--font-display)', fontSize: '14.5px' }}>Cumulative regret (lower is better)</b>
          <LineChart height={230} xDomain={[0, TOTAL]} xLabel="tries"
            series={banditStrategies.map((s) => ({ name: s, color: strategyColors[s], points: states.current[s].regretHistory }))} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', minWidth: 0 }}>
          <label class="toggle" for="workflows-focus">Strategy
            <select id="workflows-focus" class="text" style={{ width: 'auto' }} value={focus}
              onChange={(e) => setFocus((e.currentTarget as HTMLSelectElement).value as BanditStrategy)}>
              {banditStrategies.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          <b style={{ fontFamily: 'var(--font-display)', fontSize: '14.5px' }}>How often {focus} tried each prompt</b>
          <HBarChart domain={[0, maxPulls]} format={(v) => String(v)}
            bars={Bandit.arms.map((a) => ({
              label: `${a.name} (${Math.round(a.rate * 100)}%)`, value: st.pulls[a.id],
              note: st.pulls[a.id] > 0 ? `est. ${Math.round(Bandit.mean(st, a.id) * 100)}%` : undefined,
              color: a.rate === Bandit.bestRate ? 'var(--good)' : 'color-mix(in srgb, var(--c8) 60%, var(--panel))',
            }))} />
        </div>
      </div>
      <p class="prose">{md(`**${focus}:** ${banditExplanation[focus]}`)}</p>
      <div class="stats">
        {banditStrategies.map((s) => <Stat key={s} label={`${s} regret`} value={states.current[s].regret.toFixed(1)} tone={strategyColors[s]} />)}
      </div>
      <Note>Try **New random run** a few times. Greedy is sometimes perfect and sometimes stuck on a bad prompt for good. Over many runs, ε-greedy and Thompson sampling end up with similar average regret, but Thompson has the fewest bad runs: its regret curve bends flat as it stops wasting tries. UCB1 is the most cautious: with prompts this close in quality it keeps re-checking them, so over 600 tries its regret is often higher, but it never gets permanently stuck the way greedy can.</Note>
    </Card>
  )
}
