import { useMemo, useState } from 'preact/hooks'
import { tokenizer } from '../engine/model'
import { Num } from '../engine/tensor'
import { BarChart, Heatmap } from '../ui/charts'
import { Btn, Card, Formula, GoDeeper, KeyIdea, ModelBanner, Note, P, Page, Segmented, Slider, TextInput, useTrainer } from '../ui/kit'

const range = (n: number) => Array.from({ length: n }, (_, i) => i)
const subhead = { font: '600 14.5px var(--font-display)', margin: 0 }

export default function Attention() {
  const t = useTrainer()
  const model = t.snapshot
  const [text, setText] = useState('the farmer brings a warm cake.')
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [query, setQuery] = useState(10)
  const [sharpness, setSharpness] = useState(1)

  const ids = useMemo(() => tokenizer.encode(tokenizer.clean(text)).slice(0, model.config.context), [text, model])
  const trace = useMemo(() => (ids.length >= 2 ? model.trace(ids) : null), [ids, model])
  const textInput = <TextInput id="attention-text" value={text} onInput={setText} placeholder="Text" />

  return (
    <Page id="attention">
      <ModelBanner />
      <P>After embedding, each token is a vector that only knows about itself. But to predict what comes after "the farmer brings a warm …", the last position needs information from *earlier* tokens. **Attention** is how tokens share information.</P>
      <P>Every token makes three vectors from its embedding: a **query** (*what am I looking for?*), a **key** (*what do I contain?*) and a **value** (*what will I hand over if picked?*). Each token compares its query with the keys of every earlier token. Strong matches get a bigger share of those tokens' values.</P>

      {!trace ? (
        <Card>
          {textInput}
          <Note>Type at least two characters.</Note>
        </Card>
      ) : (() => {
        const q = Math.min(query, ids.length - 1)
        const labels = ids.map((i) => tokenizer.display(i))
        const L = Math.min(layer, model.config.nLayers - 1), H = Math.min(head, model.config.nHeads - 1)
        const raw = trace.scores[L][H][q]
        const visible = raw.slice(0, q + 1).map((s) => s * sharpness)
        const weights = Num.softmax(visible)
        const top3 = weights.map((w, j) => [j, w] as const).sort((a, b) => b[1] - a[1]).slice(0, 3)
        return (
          <>
            <Card title="Attention patterns inside the tiny model" subtitle="Row = the token doing the looking. Column = the token being looked at. Darker = more attention. Click a row.">
              <div class="row">
                <div class="grow">{textInput}</div>
                <Segmented options={range(model.config.nLayers)} value={L} onChange={setLayer} labels={(v) => `Layer ${v + 1}`} />
                <Segmented options={range(model.config.nHeads)} value={H} onChange={setHead} labels={(v) => `Head ${v + 1}`} />
              </div>
              <Heatmap values={trace.attention[L][H]} rowLabels={labels} colLabels={labels} cell={19} highlightRow={q} onRow={setQuery} />
              <Note>The empty upper-right triangle is the **causal mask**: a token may only look backwards. Otherwise, during training, it could simply read the answer it is supposed to predict.</Note>
            </Card>

            <Card title="One row, step by step" subtitle={`The token "${labels[q]}" at position ${q} decides where to look`}>
              <div class="cols">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', minWidth: 0 }}>
                  <h4 style={subhead}>① Scores: query · key ÷ √{model.config.dModel / model.config.nHeads}</h4>
                  <BarChart height={170} bars={visible.map((s, j) => ({ label: labels[j], value: s, color: s >= 0 ? 'var(--c6)' : 'var(--c8)' }))} />
                  <Note>{`Any number, positive or negative. Future tokens (after position ${q}) are masked out.`}</Note>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', minWidth: 0 }}>
                  <h4 style={subhead}>② Softmax → weights that sum to 1</h4>
                  <BarChart height={170} yDomain={[0, 1]} bars={weights.map((w, j) => ({ label: labels[j], value: w, color: 'var(--c1)' }))} />
                  <Note>{`③ Output = ${top3.map(([j, w]) => `${(w * 100).toFixed(0)}% of "${labels[j]}"`).join(' + ')} + … (their value vectors)`}</Note>
                </div>
              </div>
              <div class="row">
                <div class="grow">
                  <Slider id="attention-sharpness" label={`What if the scores were bigger? ×${sharpness.toFixed(1)}`} value={sharpness} min={0} max={5} onInput={setSharpness} />
                </div>
                <Btn onClick={() => setSharpness(1)}>Reset</Btn>
              </div>
              <Note>Bigger scores make softmax pick one token decisively; scores near zero spread attention evenly. That's why scores are divided by √(head size) — so they don't get too big just because vectors are long.</Note>
              <div class="row" style={{ gap: '6px' }}>
                <span style={{ font: '13px var(--font-display)' }}>Pick a position:</span>
                {labels.map((l, j) => (
                  <Btn key={j} small primary={j === q} onClick={() => setQuery(j)} title={`Position ${j}`}>
                    <span class="mono">{l}</span>
                  </Btn>
                ))}
              </div>
            </Card>
          </>
        )
      })()}

      <KeyIdea>Attention is a soft, learned lookup: each token asks a question, every earlier token advertises what it has, and information flows to wherever the match is strongest. Different heads learn to ask different questions.</KeyIdea>

      <GoDeeper title="the formula and its cost">
        <Formula>{'Q = X·Wq    K = X·Wk    V = X·Wv\nAttention(Q, K, V) = softmax( Q·Kᵀ / √d_head  +  mask ) · V'}</Formula>
        <P>{`**Multi-head:** the vectors are split into *h* smaller pieces, attention runs on each independently, and the results are concatenated and mixed by one more matrix (W_o). This model has ${t.config.nHeads} heads per layer; large models have 32–128.`}</P>
        <P>**Cost:** every token compares with every earlier token, so the work grows with the *square* of the context length. This is why long context is expensive, and why tricks like FlashAttention (computing attention in tiles that fit in fast on-chip memory) matter.</P>
        <P>**KV cache:** when generating text one token at a time, the keys and values of earlier tokens don't change, so they're stored and reused rather than recomputed.</P>
      </GoDeeper>
    </Page>
  )
}
