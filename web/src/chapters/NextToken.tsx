import { useRef, useState } from 'preact/hooks'
import { tokenizer } from '../engine/model'
import { Num, Rng } from '../engine/tensor'
import { BarChart, LineChart } from '../ui/charts'
import { Btn, Card, Chip, Formula, GoDeeper, KeyIdea, ModelBanner, Note, P, Page, Slider, Stat, TextInput, useTrainer } from '../ui/kit'

const tok = tokenizer
const pct = (p: number) => (p * 100).toFixed(1) + '%'

// The −log(p) curve never changes, so build it once.
const lossCurve = Array.from({ length: 200 }, (_, i) => { const p = 0.005 * (i + 1); return { x: p, y: -Math.log(p) } })

export default function NextToken() {
  const t = useTrainer()
  const model = t.snapshot
  const [prompt, setPrompt] = useState('the cat ')
  const [temperature, setTemperature] = useState(1)
  const [actual, setActual] = useState(tok.encode('s')[0] ?? 0)
  const [sentence, setSentence] = useState('the queen chases the old boat at night.')
  const rng = useRef(new Rng(11))

  const ids = tok.encode(tok.clean(prompt))
  const probs = model.nextTokenProbs(ids, temperature)
  const rawProbs = model.nextTokenProbs(ids)
  const top = probs.map((p, i) => ({ i, p })).sort((a, b) => b.p - a.p).slice(0, 14)
  const pActual = rawProbs[actual]

  const sids = tok.encode(tok.clean(sentence)).slice(0, model.config.context + 1)
  let losses: number[] = []
  if (sids.length >= 2) {
    const logits = model.forward(sids.slice(0, -1))
    losses = sids.slice(0, -1).map((_, i) => -Math.log(Num.softmax(logits.row(i))[sids[i + 1]]))
  }

  return (
    <Page id="next-token">
      <ModelBanner />
      <P>At the top of the stack, the model outputs one score (a **logit**) per vocabulary token. **Softmax** turns the scores into probabilities that add up to 100%. That list *is* the model's prediction: not one answer, but a bet spread across every possible next token.</P>

      <Card title="Ask the model what comes next" subtitle="Real probabilities from the tiny model">
        <div class="row">
          <div class="grow"><TextInput id="next-token-prompt" value={prompt} onInput={setPrompt} placeholder="Prompt" /></div>
          <Btn primary onClick={() => setPrompt(prompt + tok.vocab[Num.sample(probs, rng.current)])}>Sample next</Btn>
          <Btn onClick={() => setPrompt(prompt + tok.vocab[Num.argmax(probs)])}>Most likely</Btn>
          <Btn onClick={() => setPrompt('the cat ')}>Reset</Btn>
        </div>
        <BarChart height={210} yDomain={[0, 1]} yFormat={(v) => `${Math.round(v * 100)}%`}
          valueFormat={(v) => (v >= 0.01 ? `${Math.round(v * 100)}%` : '')}
          bars={top.map(({ i, p }) => ({ label: tok.display(i), value: p, color: 'color-mix(in srgb, var(--c6) 75%, var(--panel))' }))} />
        <Slider id="next-token-temperature" label={`Temperature ${temperature.toFixed(2)}`} value={temperature} min={0.1} max={2.5} onInput={setTemperature} />
        <Note>Generation is just this, in a loop: predict, pick one token, append it, predict again. **Temperature** divides the scores before softmax — low = safe and repetitive, high = creative and chaotic.</Note>
      </Card>

      <Card title="Scoring a guess: the loss" subtitle="Training needs one number that says how wrong the prediction was">
        <div class="cols">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <label class="toggle" for="next-token-actual">Suppose the real next character was
              <select id="next-token-actual" class="text mono" style={{ width: 'auto' }} value={String(actual)}
                onChange={(e) => setActual(+(e.currentTarget as HTMLSelectElement).value)}>
                {tok.vocab.map((_, i) => <option key={i} value={String(i)}>{tok.display(i)}</option>)}
              </select>
            </label>
            <div class="stats">
              <Stat label="Model gave it" value={pct(pActual)} tone="var(--c6)" />
              <Stat label="Loss = −log(p)" value={(-Math.log(pActual)).toFixed(2)} tone="var(--c2)" />
            </div>
            <Note>If the model gave the right answer 100%, the loss is 0. At 50% it's 0.69. At 1% it's 4.6. Confident *and* wrong is punished hardest.</Note>
          </div>
          <LineChart height={200} xDomain={[0, 1]} yDomain={[0, 5.5]} xLabel="probability given to the right answer" yLabel="loss" legend={false}
            series={[
              { name: '−log(p)', color: 'var(--c1)', points: lossCurve },
              { name: 'this guess', color: 'var(--c2)', points: [{ x: pActual, y: -Math.log(pActual) }], dots: true, width: 0 },
            ]} />
        </div>
      </Card>

      <Card title="Loss over a whole sentence" subtitle="Every position is a prediction. Redder = more surprised.">
        <TextInput id="next-token-sentence" value={sentence} onInput={setSentence} placeholder="Sentence" />
        {sids.length >= 2 && (
          <>
            <div class="chips">
              <Chip text={tok.display(sids[0])} color="var(--c8)" sub="start" />
              {losses.map((l, i) => (
                <Chip key={i} text={tok.display(sids[i + 1])} sub={l.toFixed(1)}
                  color={`color-mix(in srgb, var(--bad) ${Math.round(Math.min(1, l / 4) * 100)}%, var(--good))`} />
              ))}
            </div>
            <div class="stats">
              <Stat label="Average loss (what training reports)" value={(losses.reduce((a, b) => a + b, 0) / losses.length).toFixed(2)} tone="var(--c2)" />
              <Stat label="Random guessing would be" value={Math.log(tok.size).toFixed(2)} tone="var(--c8)" />
            </div>
            <Note>Notice the pattern once trained: the *first* letter of each word is hard to guess, the rest of the word is easy. A single forward pass scores every position at once — that's why training is so efficient.</Note>
          </>
        )}
      </Card>

      <KeyIdea>Pretraining has one objective: make the average −log(probability of the actual next token) as small as possible across all the training text. Everything the model 'knows' is a side effect of getting better at this.</KeyIdea>

      <GoDeeper title="cross-entropy, perplexity and sampling">
        <Formula>{'p = softmax(logits)          p_i = e^{z_i} / Σ_j e^{z_j}\nloss = −(1/T) Σ_t log p(x_{t+1} | x_1 … x_t)'}</Formula>
        <P>This is the **cross-entropy** between the true next token and the model's distribution. Its exponential, **perplexity**, reads as "how many tokens is the model choosing between": loss 3.33 here = perplexity 28 = pure guessing among 28 characters.</P>
        <P>**Teacher forcing:** during training, each position predicts its next token given the *real* previous tokens (not the model's own guesses), so all T predictions are computed in one parallel pass.</P>
        <P>**Sampling tricks:** besides temperature, **top-k** keeps only the k most likely tokens and **top-p (nucleus)** keeps the smallest set whose probabilities sum to p. Both cut off the long tail of unlikely tokens.</P>
      </GoDeeper>
    </Page>
  )
}
