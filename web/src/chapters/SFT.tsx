import { useState } from 'preact/hooks'
import { Num } from '../engine/tensor'
import { PostTraining, Point } from '../engine/toys'
import { HBarChart, LineChart, Legend } from '../ui/charts'
import { Btn, Card, Formula, GoDeeper, KeyIdea, Note, P, Page, Slider, Stat, Toggle } from '../ui/kit'

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

      <Card title="Live: fine-tune a toy model" subtitle={`A simplified model that chooses between 8 whole answers to "${PostTraining.prompt}". The update is the real cross-entropy gradient.`}>
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
