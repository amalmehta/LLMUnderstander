import { optimizerKinds, OptimizerKind } from '../engine/model'
import { loopPhases } from '../engine/trainer'
import { fmt } from '../engine/toys'
import { LineChart } from '../ui/charts'
import { Btn, Card, Formula, GoDeeper, KeyIdea, Note, P, Page, Segmented, Stat, Toggle, useTrainer } from '../ui/kit'

const lrChoices = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

export default function TrainingLoop() {
  const t = useTrainer()
  const total = t.settings.totalSteps
  const last = <T,>(xs: T[]) => xs[xs.length - 1]
  const smoothed = t.smoothed()
  // Thin the raw series so the chart stays light.
  const raw = t.trainLoss.filter((_, i) => i % 3 === 0)

  return (
    <Page id="training-loop">
      <P>{`Everything comes together here. Below is a real transformer (${t.config.nLayers} blocks, ${t.config.nHeads} heads, width ${t.config.dModel}, ${t.parameterCount.toLocaleString()} weights) and a real training loop, running in your browser. It reads a corpus of simple English sentences, **one character at a time**. Press **Start training** and watch it go from random noise to writing sentences.`}</P>

      <Card>
        <div class="row">
          <Btn primary onClick={() => (t.isRunning ? t.pause() : t.start())} disabled={t.finished}>
            {t.isRunning ? '❚❚ Pause' : t.step === 0 ? '▶ Start training' : '▶ Resume'}
          </Btn>
          <Btn onClick={() => t.reset()}>↺ Reset</Btn>
          <Toggle id="slowmo" label="Slow motion" checked={t.slowMotion} onChange={(v) => t.setSlowMotion(v)} />
          <span class="grow" />
          <label class="toggle" for="opt">Optimizer
            <select id="opt" class="text" style={{ width: 'auto' }} value={t.settings.optimizer} disabled={t.step > 0}
              onChange={(e) => t.setSettings({ optimizer: (e.currentTarget as HTMLSelectElement).value as OptimizerKind })}>
              {optimizerKinds.map((k) => <option key={k}>{k}</option>)}
            </select>
          </label>
          <label class="toggle" for="lr">Peak learning rate
            <select id="lr" class="text" style={{ width: 'auto' }} value={String(t.settings.peakLR)} disabled={t.step > 0}
              onChange={(e) => t.setSettings({ peakLR: +(e.currentTarget as HTMLSelectElement).value })}>
              {lrChoices.map((v) => <option key={v} value={String(v)}>{v}</option>)}
            </select>
          </label>
        </div>
        {t.step > 0 && !t.isRunning && !t.finished && <Note>Settings are locked once training starts. Press Reset to try different ones (e.g. a huge learning rate, or plain SGD).</Note>}
        <div class="stats">
          <Stat label="Step" value={`${t.step.toLocaleString()} / ${total.toLocaleString()}`} tone="var(--sec-learning)" />
          <Stat label="Tokens seen" value={fmt.big(t.tokensSeen)} tone="var(--c1)" />
          <Stat label="Train loss" value={smoothed.length ? last(smoothed).y.toFixed(3) : '—'} tone="var(--c2)" />
          <Stat label="Validation loss" value={t.valLoss.length ? last(t.valLoss).y.toFixed(3) : '—'} tone="var(--c4)" />
          <Stat label="Learning rate" value={t.learningRates.length ? last(t.learningRates).y.toFixed(5) : '—'} tone="var(--c5)" />
          <Stat label="Gradient norm" value={t.gradNorms.length ? last(t.gradNorms).y.toFixed(2) : '—'} tone="var(--c8)" />
        </div>
        <Note>Your trained model is kept in this browser, so the other chapters can use it next time you visit. Reset deletes it.</Note>
      </Card>

      <div class="cols">
        <Card title="The loop" subtitle={t.slowMotion ? 'Following one step at a time' : 'Turn on slow motion to follow each phase'}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
            <div>for step in range({total}):</div>
            {loopPhases.map((p) => {
              const on = t.phase === p.id
              return (
                <div key={p.id} style={{ padding: '5px 8px 5px 24px', borderRadius: '6px', background: on ? 'color-mix(in srgb, var(--sec-learning) 22%, var(--panel))' : 'transparent', fontWeight: on ? 700 : 400, transition: 'background 0.2s' }}>
                  {p.code}
                  <div style={{ fontFamily: 'var(--font-display)', fontSize: '12px', color: on ? 'var(--ink)' : 'var(--muted)' }}>{p.title}</div>
                </div>
              )
            })}
          </div>
        </Card>
        <Card title="What it writes" subtitle={'Prompt "the ", sampled every 100 steps'}>
          <div style={{ maxHeight: '300px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {[...t.samples].reverse().map((s) => (
              <div key={s.step} style={{ display: 'grid', gridTemplateColumns: '70px 1fr', gap: '6px' }}>
                <span class="note" style={{ fontVariantNumeric: 'tabular-nums' }}>step {s.step.toLocaleString()}</span>
                <span class="mono" style={{ wordBreak: 'break-word' }}>{s.text.replaceAll('\n', ' ↵ ')}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Loss" subtitle={`Lower is better. Dashed line = random guessing (log ${t.config.vocab} = ${t.uniformLoss.toFixed(2)}).`}>
        <LineChart height={260} xDomain={[0, total]} yDomain={[0, 4]} xLabel="training step"
          rules={[{ y: t.uniformLoss, label: 'random guessing' }]}
          series={[
            { name: 'raw batch loss', color: 'var(--c2)', points: raw, width: 1, opacity: 0.25 },
            { name: 'training loss (on the batch it just learned from)', color: 'var(--c2)', points: smoothed, width: 2.5 },
            { name: 'validation loss (sentences it never trains on)', color: 'var(--c4)', points: t.valLoss, dots: true },
          ]} />
      </Card>

      <div class="cols">
        <Card title="Learning rate" subtitle="Warm-up, then cosine decay">
          <LineChart height={150} xDomain={[0, total]} yFormat={(v) => v.toExponential(0)} series={[{ name: 'lr', color: 'var(--c5)', points: t.learningRates.filter((_, i) => i % 5 === 0) }]} />
        </Card>
        <Card title="Gradient norm" subtitle={`Size of the whole gradient (clipped at ${t.settings.clip})`}>
          <LineChart height={150} xDomain={[0, total]} series={[{ name: 'norm', color: 'var(--c8)', points: t.gradNorms.filter((_, i) => i % 3 === 0) }]} />
        </Card>
      </div>

      <KeyIdea>That's the entire algorithm behind every LLM: take a batch, predict, score, backprop, step — repeat. GPT-4-class models run this same loop with ~10¹² times more compute.</KeyIdea>

      <GoDeeper title="reading the curves">
        <P>**Fast early drop:** first the model learns which characters are common (spaces, 'e'), then common pairs ('th'), then whole words, then grammar. You can see these phases in the samples.</P>
        <P>**Train vs validation:** if training loss keeps falling while validation loss rises, the model is *memorising* rather than generalising (overfitting). LLM pretraining rarely sees the same text twice, so the two curves stay together, as they do here.</P>
        <P>**Spikes:** a sudden jump usually means the learning rate is too high for that moment. Try Reset with a peak learning rate of 0.1.</P>
        <Formula>{`for step in range(total_steps):
    x, y = get_batch("train")            # (batch, T) token ids, y = x shifted by one
    logits = model(x)                      # (batch, T, vocab)
    loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for g in optimizer.param_groups: g["lr"] = schedule(step)
    optimizer.step()`}</Formula>
      </GoDeeper>
    </Page>
  )
}
