import { useState } from 'preact/hooks'
import { Num } from '../engine/tensor'
import { PostTraining, Point, RewardModel } from '../engine/toys'
import { BarChart, HBarChart, LineChart, Legend } from '../ui/charts'
import { Btn, Callout, Card, Formula, GoDeeper, KeyIdea, Note, P, Page, Segmented, Slider, Stat, useTicker, useWidth } from '../ui/kit'

// Fixed data and a fully trained reward model, computed once for the whole page.
const data = PostTraining.collectComparisons()
const fullyTrainedRewards = RewardModel.trained().rewards
const sftProbs = Num.softmax(PostTraining.sftLogits)
const quality = PostTraining.quality
const rlTotal = 600, dpoTotal = 300
const tabLabels = ['1 · Collect comparisons', '2 · Train a reward model', '3 · RLHF', '4 · DPO']
const pct = (v: number) => `${Math.round(v * 100)}%`

interface RLPoint { step: number; proxy: number; truth: number; kl: number }

const rmWith = (w: number[]) => { const rm = new RewardModel(); rm.w = [...w]; return rm }

export default function Preferences() {
  const [tab, setTab] = useState(0)

  // reward model
  const [rmW, setRmW] = useState([0, 0, 0])
  const [rmSteps, setRmSteps] = useState(0)

  // RLHF
  const [logBeta, setLogBeta] = useState(Math.log10(0.5))
  const [policy, setPolicy] = useState<number[]>(PostTraining.sftLogits)
  const [rlStep, setRlStep] = useState(0)
  const [rlRunning, setRlRunning] = useState(false)
  const [rlHistory, setRlHistory] = useState<RLPoint[]>([])

  // DPO
  const [dpoBeta, setDpoBeta] = useState(0.5)
  const [dpoPolicy, setDpoPolicy] = useState<number[]>(PostTraining.sftLogits)
  const [dpoStep, setDpoStep] = useState(0)
  const [dpoRunning, setDpoRunning] = useState(false)
  const [dpoHistory, setDpoHistory] = useState<Point[]>([])

  const rm = rmWith(rmW)
  /** RLHF uses the reward model from step 2 if you trained it; otherwise a fully trained one. */
  const trainedRewards = rmSteps >= 100 ? rm.rewards : fullyTrainedRewards

  const trainRM = (n: number) => {
    const m = rmWith(rmW)
    for (let i = 0; i < n; i++) m.step(data, 0.5)
    setRmW(m.w); setRmSteps(rmSteps + n)
  }

  const startRL = () => { setPolicy(PostTraining.sftLogits); setRlStep(0); setRlHistory([]); setRlRunning(true) }
  const resetRL = () => { setRlRunning(false); setPolicy(PostTraining.sftLogits); setRlStep(0); setRlHistory([]) }
  const startDPO = () => { setDpoPolicy(PostTraining.sftLogits); setDpoStep(0); setDpoHistory([]); setDpoRunning(true) }
  const resetDPO = () => { setDpoRunning(false); setDpoPolicy(PostTraining.sftLogits); setDpoStep(0); setDpoHistory([]) }

  useTicker(rlRunning || dpoRunning, 30, () => {
    if (rlRunning) {
      const beta = 10 ** logBeta, r = trainedRewards
      let pol = policy, s = rlStep
      const add: RLPoint[] = []
      for (let i = 0; i < 12 && s < rlTotal; i++) {
        pol = PostTraining.rlhfStep(pol, r, beta)
        s++
        if (s % 6 === 0 || s === 1) {
          const p = Num.softmax(pol)
          add.push({ step: s, proxy: PostTraining.expected(p, r), truth: PostTraining.expected(p, quality), kl: PostTraining.kl(p, sftProbs) })
        }
      }
      setPolicy(pol); setRlStep(s); setRlHistory((h) => [...h, ...add])
      if (s >= rlTotal) setRlRunning(false)
    }
    if (dpoRunning) {
      let pol = dpoPolicy, s = dpoStep
      const add: Point[] = []
      for (let i = 0; i < 6 && s < dpoTotal; i++) {
        pol = PostTraining.dpoStep(pol, data, dpoBeta, 1)
        s++
        if (s % 3 === 0) add.push({ x: s, y: PostTraining.dpoLoss(pol, data, dpoBeta) })
      }
      setDpoPolicy(pol); setDpoStep(s); setDpoHistory((h) => [...h, ...add])
      if (s >= dpoTotal) setDpoRunning(false)
    }
  })

  // ------------------------------------------------------------ 1 comparisons
  const comparisons = () => (
    <Card title="Human comparisons" subtitle={`${data.length} judgements: for the same prompt, which of two answers is better?`}>
      {data.slice(0, 4).map((c) => (
        <div key={c.id} class="row top" style={{ gap: '10px' }}>
          <AnswerBox id={c.chosen} chosen />
          <span class="note" style={{ alignSelf: 'center' }}>vs</span>
          <AnswerBox id={c.rejected} chosen={false} />
        </div>
      ))}
      <HBarChart domain={[0, 1]} bars={PostTraining.answers.map((a) => {
        const won = data.filter((c) => c.chosen === a.id).length
        const shown = data.filter((c) => c.chosen === a.id || c.rejected === a.id).length
        return { label: a.short, value: shown > 0 ? won / shown : 0, color: 'color-mix(in srgb, var(--c4) 60%, var(--panel))', note: shown === 0 ? 'never shown to labellers' : `${won}/${shown} wins` }
      })} />
      <P>Two realistic flaws are built in. Labellers **slightly favour longer answers**, a known bias. And the **long rambling essay never appeared** in the labelled data. Watch what the next steps do with that.</P>
    </Card>
  )

  // ------------------------------------------------------------ 2 reward model
  const rewardModel = () => (
    <Card title="Reward model" subtitle="Learns a score r(answer) such that preferred answers score higher. Here: a weighted sum of 3 features.">
      <div class="row">
        <Btn primary onClick={() => trainRM(50)}>Train 50 steps</Btn>
        <Btn onClick={() => trainRM(400 - rmSteps)} disabled={rmSteps >= 400}>Train to convergence</Btn>
        <Btn onClick={() => { setRmW([0, 0, 0]); setRmSteps(0) }}>Reset</Btn>
        <span class="grow" />
        <div class="stats">
          <Stat label="Steps" value={String(rmSteps)} tone="var(--c8)" />
          <Stat label="Agrees with labellers" value={pct(rm.accuracy(data))} tone="var(--good)" />
        </div>
      </div>
      <div class="cols">
        <div>
          <b style={{ fontFamily: 'var(--font-display)', fontSize: '14.5px' }}>Learned weights</b>
          <BarChart height={180} yDomain={[-0.5, 3.5]} valueFormat={(v) => v.toFixed(2)}
            bars={['helpfulness', 'politeness', 'length'].map((n, k) => ({ label: n, value: rm.w[k], color: n === 'length' ? 'var(--c2)' : 'var(--c1)' }))} />
          <Note>It learns to like length — because the labellers did.</Note>
        </div>
        <div>
          <b style={{ fontFamily: 'var(--font-display)', fontSize: '14.5px' }}>Reward model's score vs. what's really good</b>
          <PairedBars a={rm.rewards} b={quality} />
          <Legend items={[{ name: 'reward model', color: 'var(--c4)' }, { name: 'true quality (hidden)', color: 'color-mix(in srgb, var(--good) 60%, var(--panel))' }]} />
        </div>
      </div>
      {rmSteps >= 200 && <Callout tone="var(--warn)">⚠️ The reward model rates the **long rambling essay** highest of all — it extrapolated "longer is better" to an answer no labeller ever saw. It is a good but imperfect proxy for what people want.</Callout>}
    </Card>
  )

  // ------------------------------------------------------------ 3 RLHF
  const rlhf = () => {
    const beta = 10 ** logBeta, p = Num.softmax(policy)
    const presetBeta = (label: string, b: number) => (
      <Btn small disabled={rlRunning} onClick={() => { setLogBeta(Math.log10(b)); startRL() }}>{label}</Btn>
    )
    const hacked = p[5] > 0.5
    return (
      <Card title="RLHF: optimise the reward, but stay close to the SFT model" subtitle="Objective: expected reward − β × KL(policy ‖ SFT model)">
        <div class="row">
          <Btn primary onClick={startRL} disabled={rlRunning}>{rlRunning ? 'Running…' : `Run RL (${rlTotal} steps)`}</Btn>
          <Btn onClick={resetRL}>Reset</Btn>
        </div>
        <Slider id="preferences-rlhf-beta" label={`KL leash β = ${beta.toFixed(2)}`} value={logBeta} min={-1.6} max={0.7} onInput={setLogBeta} />
        <div class="row" style={{ gap: '8px' }}>
          {presetBeta('weak leash 0.05', 0.05)}{presetBeta('balanced 0.5', 0.5)}{presetBeta('tight leash 3', 3)}
        </div>
        <div class="cols">
          <PolicyBars p={p} color="var(--c4)" />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', minWidth: 0 }}>
            <LineChart height={200} xDomain={[0, rlTotal]} xLabel="RL steps" series={[
              { name: 'reward model score', color: 'var(--c4)', points: rlHistory.map((h) => ({ x: h.step, y: h.proxy })) },
              { name: 'true quality', color: 'var(--good)', points: rlHistory.map((h) => ({ x: h.step, y: h.truth })) },
            ]} />
            <div class="stats">
              <Stat label="Reward score" value={PostTraining.expected(p, trainedRewards).toFixed(2)} tone="var(--c4)" />
              <Stat label="True quality" value={PostTraining.expected(p, quality).toFixed(2)} tone="var(--good)" />
              <Stat label="KL from SFT" value={PostTraining.kl(p, sftProbs).toFixed(2)} tone="var(--c8)" />
            </div>
          </div>
        </div>
        <Note>Gray diamonds = the SFT model it started from.</Note>
        {rlStep >= rlTotal && (
          <Callout tone={hacked ? 'var(--bad)' : 'var(--good)'}>
            {hacked
              ? "🚨 **Reward hacking.** With a weak leash the policy piled onto the rambling essay. The reward model's score went *up*, but true quality went *down*. The policy found the reward model's blind spot."
              : beta > 2 ? 'With a very tight leash the policy barely moves from the SFT model: safe, but it leaves much of the improvement on the table.'
                : "A balanced leash: the policy moves toward genuinely good answers, and the KL penalty stops it from collapsing onto the reward model's favourite mistake."}
          </Callout>
        )}
      </Card>
    )
  }

  // ------------------------------------------------------------ 4 DPO
  const dpo = () => {
    const p = Num.softmax(dpoPolicy)
    return (
      <Card title="DPO: Direct Preference Optimisation" subtitle="No reward model, no RL loop — a classification loss on the comparisons themselves">
        <div class="row">
          <Btn primary onClick={startDPO} disabled={dpoRunning}>{dpoRunning ? 'Running…' : `Run DPO (${dpoTotal} steps)`}</Btn>
          <Btn onClick={resetDPO}>Reset</Btn>
        </div>
        <Slider id="preferences-dpo-beta" label={`β = ${dpoBeta.toFixed(2)}`} value={dpoBeta} min={0.05} max={2} onInput={setDpoBeta} />
        <div class="cols">
          <PolicyBars p={p} color="var(--c5)" />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', minWidth: 0 }}>
            <LineChart height={180} xDomain={[0, dpoTotal]} xLabel="DPO steps" yLabel="DPO loss" series={[{ name: 'DPO loss', color: 'var(--c5)', points: dpoHistory }]} />
            <div class="stats">
              <Stat label="True quality" value={PostTraining.expected(p, quality).toFixed(2)} tone="var(--good)" />
              <Stat label="Essay probability" value={`${(p[5] * 100).toFixed(1)}%`} tone="var(--c2)" />
              <Stat label="KL from SFT" value={PostTraining.kl(p, sftProbs).toFixed(2)} tone="var(--c8)" />
            </div>
          </div>
        </div>
        <Note>Gray diamonds = the SFT model it started from.</Note>
        <P>DPO shows the same comparisons directly to the policy: raise the log-probability of each preferred answer and lower the rejected one, both measured relative to the SFT model. It's simpler and more stable than RLHF, and it **can't be tricked by the rambling essay**, because there's no reward model to exploit. The flip side: it only learns about answers that appear in the data, and it inherits the labellers' length bias directly.</P>
      </Card>
    )
  }

  return (
    <Page id="preferences">
      <P>After SFT the model answers like an assistant, but only as well as its demonstrations. And for many questions it's much easier for a person to say **which of two answers is better** than to write the perfect one. Preference training uses those judgements.</P>
      <P>Below is the same toy model as in the SFT chapter (one prompt, 8 possible answers). The steps are the real algorithms.</P>

      <Segmented<number> options={[0, 1, 2, 3]} value={tab} onChange={setTab} labels={(v) => tabLabels[v]} />

      {tab === 0 ? comparisons() : tab === 1 ? rewardModel() : tab === 2 ? rlhf() : dpo()}

      <KeyIdea>Preference training turns "A is better than B" into a training signal. RLHF learns a reward model and then optimises against it, held on a **KL leash** to the SFT model so it can't exploit the reward model's blind spots. DPO skips the reward model and learns from the pairs directly.</KeyIdea>

      <GoDeeper title="how it's done for real">
        <Formula>{'Reward model (Bradley–Terry):  loss = −log σ( r(chosen) − r(rejected) )\nRLHF objective:                 maximise  E_π[ r(x, y) ]  −  β · KL( π ‖ π_SFT )\nDPO:  loss = −log σ( β·[ log π(y_w)/π_SFT(y_w)  −  log π(y_l)/π_SFT(y_l) ] )'}</Formula>
        <P>**Real RLHF** samples whole answers from the model, scores them with the reward model (itself a transformer), and updates with **PPO**, a policy-gradient method that limits how far each update can move. Here, with only 8 answers, the expectation can be computed exactly instead of sampled.</P>
        <P>**RLAIF / Constitutional AI:** the comparisons can come from an AI model judging answers against a written set of principles, rather than from people.</P>
        <P>**Verifiable rewards:** for math and code the reward can be computed directly — did the answer match, did the tests pass? Reinforcement learning on such rewards, with long chains of thought, is how recent *reasoning* models are trained. Same idea: generate, score, push up what scored well.</P>
      </GoDeeper>
    </Page>
  )
}

/** One side of a human comparison. */
function AnswerBox({ id, chosen }: { id: number; chosen: boolean }) {
  const a = PostTraining.answers[id], tone = chosen ? 'var(--good)' : 'var(--bad)'
  return (
    <div class="grow" style={{ flexBasis: '220px', padding: '10px', borderRadius: '8px', background: `color-mix(in srgb, ${tone} 8%, var(--panel))`, border: `1px solid color-mix(in srgb, ${tone} 25%, var(--line))` }}>
      <div style={{ font: '700 12px var(--font-display)', color: tone }}>{chosen ? '👍 Preferred' : '👎 Rejected'}</div>
      <div style={{ fontSize: '15px', marginTop: '3px' }}>{a.text}</div>
    </div>
  )
}

/** The policy's probability for each answer, with the SFT model as grey diamonds. The essay is orange. */
function PolicyBars({ p, color }: { p: number[]; color: string }) {
  return (
    <HBarChart domain={[0, 1]} format={pct} rowHeight={32} markers={sftProbs}
      bars={PostTraining.answers.map((a) => ({ label: a.short, value: p[a.id], color: a.id === 5 ? 'var(--c2)' : color }))} />
  )
}

/** Two thin bars per answer (reward model score and hidden true quality) on one shared axis. */
function PairedBars({ a, b }: { a: number[]; b: number[] }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const all = [...a, ...b, 0]
  const lo = Math.min(...all), hi = Math.max(...all, 1)
  const labelRoom = 40, iw = Math.max(60, width - labelRoom), row = 40
  const sx = (v: number) => ((v - lo) / (hi - lo || 1)) * iw
  const zero = sx(0)
  const H = PostTraining.answers.length * row + 4
  return (
    <div ref={ref}>
      <svg class="chart" width={width} height={H} viewBox={`0 0 ${width} ${H}`} role="img" aria-label="Reward model score and true quality for each answer">
        <line x1={zero} x2={zero} y1="0" y2={H} class="grid" />
        {PostTraining.answers.map((ans, i) => {
          const y = i * row
          const bar = (v: number, dy: number, fill: string) => {
            const x = sx(v)
            return (
              <g>
                <rect x={Math.min(zero, x)} y={y + dy} width={Math.max(1.5, Math.abs(x - zero))} height="8" rx="2" fill={fill} />
                <text x={Math.max(zero, x) + 4} y={y + dy + 8} style={{ fontSize: '10px' }}>{v.toFixed(2)}</text>
              </g>
            )
          }
          return (
            <g key={ans.id}>
              <text x="0" y={y + 12} style={{ fill: 'var(--ink)', fontSize: '12px' }}>{ans.short}</text>
              {bar(a[i], 16, 'var(--c4)')}
              {bar(b[i], 26, 'color-mix(in srgb, var(--good) 60%, var(--panel))')}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
