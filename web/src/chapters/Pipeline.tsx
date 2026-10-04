import { chapterById, secVar } from '../chapters'
import { Card, GoDeeper, KeyIdea, P, Page } from '../ui/kit'

interface Stage { name: string; color: string; data: string; objective: string; loss: string; learns: string; scale: string; chapters: string[] }

const stages: Stage[] = [
  {
    name: '0 · Tokenizer', color: 'var(--c1)',
    data: 'A sample of the training text',
    objective: 'Find frequent chunks (BPE merges)',
    loss: '— (not gradient-based)',
    learns: 'How to turn any text into token IDs and back',
    scale: 'Minutes on CPUs', chapters: ['tokens'],
  },
  {
    name: '1 · Pretraining', color: 'var(--c2)',
    data: 'Trillions of tokens of web, code, books, papers',
    objective: 'Predict the next token, everywhere',
    loss: 'loss = cross_entropy(logits, next_tokens)',
    learns: 'Language, facts, skills, reasoning patterns → a base model',
    scale: 'The overwhelming majority of total compute: weeks–months on thousands of GPUs',
    chapters: ['embeddings', 'attention', 'block', 'next-token', 'backprop', 'optimizer', 'training-loop', 'pretraining'],
  },
  {
    name: '2 · Supervised fine-tuning', color: 'var(--c4)',
    data: 'Thousands to millions of example conversations',
    objective: 'Imitate good assistant answers',
    loss: 'loss = cross_entropy(logits, next_tokens, mask = assistant_tokens)',
    learns: 'Conversation format, following instructions, when to stop',
    scale: 'Small: hours to days', chapters: ['sft'],
  },
  {
    name: '3 · Preference tuning', color: 'var(--c4)',
    data: 'Comparisons (A better than B), or a reward model, or checkable tasks',
    objective: 'Produce answers people (or tests) prefer, without drifting far from SFT',
    loss: 'RLHF: −(reward − β·KL)    DPO: −log σ(β·Δ log-ratio)',
    learns: 'Helpfulness, honesty, harmlessness, style; with verifiable rewards, better reasoning',
    scale: 'Small to growing: days to weeks', chapters: ['preferences'],
  },
  {
    name: '4 · Assistant', color: 'var(--c5)',
    data: 'Your prompt',
    objective: '— (inference: no learning)',
    loss: 'next = sample(softmax(model(tokens) / temperature))',
    learns: 'Nothing. The weights are frozen; it predicts one token at a time',
    scale: 'Every message you send', chapters: ['next-token'],
  },
]

const stack: [string, string][] = [
  ['tokens', 'Text is chopped into tokens and turned into ID numbers.'],
  ['embeddings', 'Each ID looks up a learned vector; position is added.'],
  ['attention', 'Tokens look back at earlier tokens and borrow information.'],
  ['block', 'Attention + MLP + shortcuts, stacked N times, ending in a score for every possible next token.'],
  ['next-token', 'Softmax turns scores into probabilities; the loss is −log(probability of the real next token).'],
  ['backprop', "Backprop finds every weight's gradient in one backward sweep."],
  ['optimizer', 'AdamW nudges every weight downhill, with warm-up and decay.'],
  ['training-loop', 'Repeat: batch → forward → loss → backward → update.'],
  ['pretraining', 'Do it on trillions of tokens across thousands of GPUs → base model.'],
  ['sft', 'Fine-tune on example conversations, grading only the answers → assistant format.'],
  ['preferences', 'Push toward answers people prefer (RLHF or DPO), on a KL leash → aligned assistant.'],
]

const keyStyle = { font: '600 12px var(--font-display)', color: 'var(--muted)', paddingTop: '2px' }

export default function Pipeline() {
  return (
    <Page id="pipeline">
      <P>Here is the whole journey on one page. Read down the **loss** column: the loop never changes. Only the data and the definition of a good answer do.</P>

      <div role="list" aria-label="Training pipeline stages">
        {stages.map((s, i) => (
          <div key={s.name} role="listitem" style={{ display: 'grid', gridTemplateColumns: '18px minmax(0, 1fr)', columnGap: '14px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }} aria-hidden="true">
              <span style={{ width: '18px', height: '18px', borderRadius: '50%', background: s.color, flex: 'none', marginTop: '4px' }} />
              {i < stages.length - 1 && <span style={{ width: '3px', flex: 1, background: `color-mix(in srgb, ${s.color} 40%, transparent)` }} />}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', paddingBottom: '22px', minWidth: 0 }}>
              <h2 class="section" style={{ color: s.color, margin: 0 }}>{s.name}</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'auto minmax(0, 1fr)', columnGap: '14px', rowGap: '6px', fontSize: '15px', lineHeight: 1.45 }}>
                <span style={keyStyle}>Data</span><span>{s.data}</span>
                <span style={keyStyle}>Goal</span><span>{s.objective}</span>
                <span style={keyStyle}>Loss</span>
                <span><code style={{ background: `color-mix(in srgb, ${s.color} 10%, var(--panel))`, fontSize: '12.5px', wordBreak: 'break-word' }}>{s.loss}</code></span>
                <span style={keyStyle}>Learns</span><span>{s.learns}</span>
                <span style={keyStyle}>Scale</span><span>{s.scale}</span>
              </div>
              <div class="row" style={{ gap: '6px' }}>
                {s.chapters.map((id) => {
                  const c = chapterById(id)!
                  return <a key={id} class="btn small" href={`#${id}`} style={{ textDecoration: 'none', color: 'var(--ink)' }}>{c.number}. {c.title}</a>
                })}
              </div>
            </div>
          </div>
        ))}
      </div>

      <Card title="The whole stack in one sentence each">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {stack.map(([id, text]) => {
            const c = chapterById(id)!
            return (
              <a key={id} href={`#${id}`} class="row top" style={{ gap: '4px 10px', textDecoration: 'none', color: 'var(--ink)', fontFamily: 'var(--font-display)', fontSize: '15px', lineHeight: 1.4 }}>
                <span style={{ flex: '0 0 210px', display: 'flex', gap: '8px', alignItems: 'baseline', fontWeight: 650 }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '3px', background: secVar(c.section), flex: 'none' }} aria-hidden="true" />
                  {c.title}
                </span>
                <span class="grow" style={{ color: 'var(--muted)' }}>{text}</span>
              </a>
            )
          })}
        </div>
      </Card>

      <KeyIdea>If you remember one thing: an LLM is next-token prediction, trained by gradient descent, with the data and the definition of 'good' changed at each stage.</KeyIdea>

      <GoDeeper title="what this site leaves out">
        <P>**Evaluation:** benchmarks, human ratings and red-teaming run throughout every stage to decide what to change next.</P>
        <P>**Mid-training and long context:** many pipelines add stages between pretraining and SFT: high-quality data annealing, extending the context window, adding code or math.</P>
        <P>**Inference engineering:** KV caching, batching many users together, quantisation (storing weights in 8 or 4 bits) and speculative decoding make serving fast and cheap.</P>
        <P>**Agents:** how a trained model becomes an agent that uses tools inside a harness is its own section — Chapters 14–18.</P>
      </GoDeeper>
    </Page>
  )
}
