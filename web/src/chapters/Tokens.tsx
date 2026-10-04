import { useMemo, useState } from 'preact/hooks'
import { BPE, TRAINING_TEXT, tokenizer } from '../engine/model'
import { Card, Chip, GoDeeper, KeyIdea, Note, P, Page, Segmented, Slider, Stat, TextInput, tokenColor } from '../ui/kit'

interface Piece { text: string; id: number }

export default function Tokens() {
  const [text, setText] = useState('the old man finds a warm cake in the garden.')
  const [mode, setMode] = useState<0 | 1>(1)
  const [mergesUsed, setMergesUsed] = useState(60)
  const bpe = useMemo(() => BPE.train(TRAINING_TEXT, 200), [])

  const pieces: Piece[] = useMemo(() => {
    if (mode === 0) return tokenizer.encode(tokenizer.clean(text)).map((id) => ({ text: tokenizer.display(id), id }))
    // Give each distinct piece a stable ID: characters first, then merges in order.
    const base = tokenizer.vocab
    const merged = bpe.merges.map((m) => m.joined)
    return bpe.encode(text.toLowerCase(), mergesUsed).map((p) => {
      const raw = p.replaceAll('␣', ' ').replaceAll('↵', '\n')
      const i = base.indexOf(raw)
      if (i >= 0) return { text: p, id: i }
      const j = merged.indexOf(p)
      if (j >= 0) return { text: p, id: base.length + j }
      return { text: p, id: 999 }
    })
  }, [text, mode, mergesUsed, bpe])

  const chars = Array.from(text).length

  return (
    <Page id="tokens">
      <P>A neural network only does arithmetic, so the first step is turning text into numbers. We chop the text into **tokens** and give each distinct token an ID. The set of all tokens is the **vocabulary**.</P>
      <P>There are three obvious ways to chop. **Characters** give a tiny vocabulary but very long sequences. **Whole words** give short sequences but a huge vocabulary, and new words are impossible. Real LLMs use the middle path: **subwords**, learned by an algorithm called **byte-pair encoding (BPE)**.</P>

      <Card title="Try it" subtitle="Type anything — see how it becomes token IDs">
        <TextInput id="tokens-text" value={text} onInput={setText} placeholder="Type some text" />
        <Segmented options={[0, 1] as const} value={mode} onChange={setMode}
          labels={(v) => (v === 0 ? 'Characters (what our tiny model uses)' : 'BPE subwords (what real LLMs use)')} />
        {mode === 1 && (
          <Slider id="tokens-merges" label={`Merges learned: ${mergesUsed}`} value={mergesUsed} min={0} max={200} step={1} onInput={setMergesUsed} />
        )}
        <div class="chips">
          {pieces.map((p, i) => <Chip key={i} text={p.text} color={tokenColor(p.id)} sub={String(p.id)} />)}
        </div>
        <div class="stats">
          <Stat label="Characters" value={String(chars)} />
          <Stat label="Tokens" value={String(pieces.length)} tone="var(--c2)" />
          <Stat label="Characters per token" value={(chars / Math.max(pieces.length, 1)).toFixed(1)} tone="var(--c3)" />
        </div>
        <Note>{mode === 0
          ? 'Each character is one token. Characters the model never saw in training (capitals, digits, most punctuation) are dropped here.'
          : 'Drag the slider: with 0 merges BPE is just characters. Each merge glues the most common neighbouring pair into a new token, so frequent words become single tokens.'}</Note>
      </Card>

      <Card title="Watch BPE learn its vocabulary" subtitle="Trained live on this site's corpus. Each row is one merge, in the order it was learned.">
        <div style={{ maxHeight: '220px', overflowY: 'auto', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(175px, 1fr))', gap: '6px 8px', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
          {bpe.merges.slice(0, 200).map((m, i) => (
            <div key={i} style={{ display: 'flex', gap: '4px', alignItems: 'baseline', opacity: i < mergesUsed ? 1 : 0.3, whiteSpace: 'nowrap' }}>
              <span style={{ width: '28px', textAlign: 'right', color: 'var(--muted)', fontSize: '10.5px', flex: 'none' }}>{i + 1}.</span>
              <span>{m.left}</span>
              <span style={{ color: 'var(--muted)', fontSize: '10.5px' }}>+</span>
              <span>{m.right}</span>
              <span style={{ color: 'var(--muted)', fontSize: '10.5px' }}>→</span>
              <b>{m.joined}</b>
              <span style={{ color: 'var(--muted)', fontSize: '9px' }}>×{m.count}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Our tiny model's whole vocabulary" subtitle={`${tokenizer.size} tokens — one per character. GPT-2 has 50,257; Llama 3 has 128,256.`}>
        <div class="chips">
          {tokenizer.vocab.map((_, i) => <Chip key={i} text={tokenizer.display(i)} color={tokenColor(i)} sub={String(i)} />)}
        </div>
      </Card>

      <KeyIdea>From here on the model never sees letters — only lists of IDs like [19, 7, 4, 0, …]. Everything it learns is about which IDs tend to follow which.</KeyIdea>

      <GoDeeper title="how real tokenizers work">
        <P>**BPE training:** start with every byte (256 tokens) as the vocabulary. Count every adjacent pair in the training text. Merge the most frequent pair into a new token. Repeat until the vocabulary hits the target size (~50k–200k).</P>
        <P>**Byte-level** means the base alphabet is the 256 possible bytes, so *any* text — emoji, Chinese, code — can be encoded with no 'unknown' token.</P>
        <P>{"**Special tokens** like `<|endoftext|>` or chat-role markers are added by hand. They mark document boundaries and who is speaking, and become important in post-training (Chapter 11)."}</P>
        <P>**Rule of thumb:** in English, one token ≈ 4 characters ≈ ¾ of a word. Token counts are how training data, context windows and prices are measured.</P>
      </GoDeeper>
    </Page>
  )
}
