import { useState } from 'preact/hooks'
import { loopPhases } from '../engine/trainer'
import { Card, Flow, KeyIdea, P, Page, useTicker } from '../ui/kit'

const phaseTargets = ['tokens', 'attention', 'next-token', 'backprop', 'optimizer']

export default function BigPicture() {
  return (
    <Page id="big-picture">
      <P>A large language model is a function that reads some text and outputs **a probability for every possible next piece of text**. That's it. Everything impressive — answering questions, writing code, holding a conversation — comes from doing that one thing extremely well, over and over.</P>
      <P>**Training** is how the function gets good: show it text, measure how wrong its guesses are, and nudge its billions of numbers (its *weights*) so it's a little less wrong next time. Repeat trillions of times.</P>

      <Card title="The three stages" subtitle="Click any stage to jump to its chapter">
        <Flow steps={[
          { title: 'Raw text', detail: 'web pages, books, code', tone: 'var(--c8)', href: '#pretraining' },
          { title: 'Tokens', detail: 'text → numbers', tone: 'var(--sec-blocks)', href: '#tokens' },
          { title: 'Pretraining', detail: 'predict the next token\non trillions of tokens', tone: 'var(--sec-pretraining)', href: '#pretraining' },
          { title: 'Base model', detail: 'a brilliant autocomplete', tone: 'var(--sec-pretraining)', href: '#next-token' },
          { title: 'SFT', detail: 'learn the assistant format\nfrom examples', tone: 'var(--sec-posttraining)', href: '#sft' },
          { title: 'Preference tuning', detail: 'RLHF / DPO: learn\nwhat people prefer', tone: 'var(--sec-posttraining)', href: '#preferences' },
          { title: 'Assistant', detail: 'helpful, honest,\nharmless', tone: 'var(--sec-wrap)', href: '#pipeline' },
        ]} />
      </Card>

      <div class="cols">
        <Card title="The one loop inside every stage" subtitle="Click a step to learn how it works">
          <LoopCircle />
        </Card>
        <Card title="What changes between stages?">
          <div class="scroll-x">
            <table class="grid-table">
              <thead><tr><th></th><th>Data</th><th>What counts as “right”</th></tr></thead>
              <tbody>
                <tr><td style={{ color: 'var(--sec-pretraining)', fontWeight: 650 }}>Pretraining</td><td>Trillions of tokens of raw text</td><td>The actual next token</td></tr>
                <tr><td style={{ color: 'var(--sec-posttraining)', fontWeight: 650 }}>SFT</td><td>~10k–1M written conversations</td><td>The demo answer's tokens (only the answer)</td></tr>
                <tr><td style={{ color: 'var(--sec-posttraining)', fontWeight: 650 }}>RLHF</td><td>Prompts + a reward model</td><td>Whatever scores high reward, without drifting far</td></tr>
                <tr><td style={{ color: 'var(--sec-posttraining)', fontWeight: 650 }}>DPO</td><td>Pairs: better vs worse answer</td><td>Raise the better one relative to the worse</td></tr>
                <tr><td style={{ color: 'var(--sec-agents)', fontWeight: 650 }}>Agent RL</td><td>Tasks in sandboxes</td><td>Did the task verifiably succeed?</td></tr>
              </tbody>
            </table>
          </div>
          <p class="note">The loop — forward, loss, backward, update — is identical in all of them. Only the data and the definition of “right” change.</p>
        </Card>
      </div>

      <KeyIdea>Learn the loop once and you understand every stage. Chapters 2–5 build the model, 6–9 show how it learns, 10–12 show how that loop is used at scale and then reshaped into an assistant, and 14–18 show how the assistant becomes an agent.</KeyIdea>

      <Card title="There's a real model inside this page">
        <P>Most pictures on this site are not drawings — they are computed live, in your browser, by a tiny GPT (about 28,000 weights). It starts out knowing nothing. In **Chapter 9** you can train it and watch it learn English-like sentences in a minute or two. Once trained, the tokens, embeddings, attention and predictions in the other chapters all come from it.</P>
      </Card>
    </Page>
  )
}

/** The five steps of a training step on a circle, with a pulse travelling round. */
function LoopCircle() {
  const [active, setActive] = useState(0)
  useTicker(true, 1100, () => setActive((a) => (a + 1) % 5))
  const W = 460, H = 400, cx = W / 2, cy = H / 2, r = 140
  return (
    <div class="scroll-x">
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: `${W}px`, display: 'block', margin: '0 auto', fontFamily: 'var(--font-display)' }} role="img" aria-label="The training loop: five steps repeated">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--sec-learning)" stroke-opacity="0.35" stroke-width="3" stroke-dasharray="6 6" />
        <text x={cx} y={cy - 4} text-anchor="middle" style={{ fill: 'var(--muted)', fontSize: '12px' }}>repeat</text>
        <text x={cx} y={cy + 13} text-anchor="middle" style={{ fill: 'var(--ink)', fontSize: '12.5px', fontWeight: 600 }}>millions of times</text>
        {loopPhases.map((p, i) => {
          const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5, x = cx + r * Math.cos(a), y = cy + r * Math.sin(a)
          const on = i === active, w = 150, h = 62
          return (
            <a key={p.id} href={`#${phaseTargets[i]}`}>
              <g transform={`translate(${x - w / 2} ${y - h / 2})`} style={{ cursor: 'pointer' }}>
                <rect width={w} height={h} rx="9" fill={on ? 'color-mix(in srgb, var(--sec-learning) 20%, var(--panel))' : 'var(--panel)'} stroke={on ? 'var(--sec-learning)' : 'var(--line)'} stroke-width={on ? 2 : 1} />
                <circle cx="14" cy="14" r="8" fill={on ? 'var(--sec-learning)' : 'var(--line)'} />
                <text x="14" y="18" text-anchor="middle" style={{ fill: on ? 'var(--panel)' : 'var(--ink)', fontSize: '10px', fontWeight: 700 }}>{i + 1}</text>
                <text x={w / 2 + 8} y="21" text-anchor="middle" style={{ fill: 'var(--ink)', fontSize: '11.5px', fontWeight: on ? 700 : 500 }}>{p.title.split(':')[0]}</text>
                <text x={w / 2} y="40" text-anchor="middle" style={{ fill: 'var(--muted)', fontSize: '10px' }}>{p.title.includes(':') ? p.title.split(':')[1].trim() : ''}</text>
                <text x={w / 2} y="54" text-anchor="middle" style={{ fill: 'var(--muted)', fontSize: '9.5px', fontFamily: 'var(--font-mono)' }}>{p.code.length > 26 ? p.code.slice(0, 24) + '…' : p.code}</text>
              </g>
            </a>
          )
        })}
      </svg>
    </div>
  )
}
