import { useState } from 'preact/hooks'
import { Btn, Card, GoDeeper, KeyIdea, md, Note, P, Page, Segmented, useTicker } from '../ui/kit'

const definition = `{
  "name": "search_docs",
  "description": "Search the project's documentation. Returns matching passages.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query":       { "type": "string",  "description": "What to look for" },
      "max_results": { "type": "integer", "description": "1–10, default 3" }
    },
    "required": ["query"]
  }
}`

const rendered = `<system>You are a helpful assistant for the Acme HTTP client.</system>
<tools>
  search_docs(query: string, max_results?: integer)
    Search the project's documentation. Returns matching passages.
</tools>
<user>What's the default request timeout?</user>
<assistant>`

const call = `I'll check the docs.
{
  "type": "tool_use",
  "id": "call_01",
  "name": "search_docs",
  "input": { "query": "default request timeout", "max_results": 2 }
}`

const result = `{
  "type": "tool_result",
  "tool_use_id": "call_01",
  "content": "config.md: \`timeout\` defaults to 30 seconds. Set it to 0 to wait forever."
}`

const answer = 'The default request timeout is **30 seconds** (per config.md). You can set `timeout` to 0 to disable it.'

const stages: [string, string][] = [
  ['① Define', 'The developer describes the tool: a name, a description written for the model, and a **JSON schema** for its inputs.'],
  ['② Render into the prompt', "The harness turns the definitions into text inside the prompt, using a format the model saw during training. The model 'knows' about the tool only because its description is sitting in the context."],
  ['③ The model writes a call', 'The model generates tokens as always. Here the likeliest continuation is a tool call in a structured format, ending with a special stop token. The harness stops generation there and reads the call.'],
  ['④ The harness runs it', 'The harness parses the JSON, checks it against the schema and the permission rules, runs the real function and wraps the output as a **tool result** whose `tool_use_id` matches the call.'],
  ['⑤ The model continues', "With the result now in the context, the model is called again and writes the answer. To the model, the tool result is just more text — which is why tool outputs can't be trusted blindly (see Go deeper)."],
]

const problems = ['Bad arguments', 'Tool fails', 'Parallel calls'] as const
type Problem = (typeof problems)[number]

const GRAY = 'var(--c8)', PINK = 'var(--c4)', BLUE = 'var(--c1)', RED = 'var(--bad)'

/** A labelled monospaced block, tinted by role. */
function Code({ text, label, tone, dim, hidden }: { text: string; label: string; tone: string; dim?: boolean; hidden?: boolean }) {
  return (
    <div aria-hidden={hidden ? 'true' : undefined} style={{
      padding: '10px', borderRadius: '8px', minWidth: 0, opacity: dim ? 0.55 : 1, visibility: hidden ? 'hidden' : undefined,
      background: `color-mix(in srgb, ${tone} 8%, var(--panel))`, border: `1px solid color-mix(in srgb, ${tone} 30%, var(--line))`,
    }}>
      <div style={{ font: '700 9.5px var(--font-display)', letterSpacing: '0.08em', textTransform: 'uppercase', color: tone, marginBottom: '4px' }}>{label}</div>
      <pre class="code-block" style={{ padding: 0 }}>{text}</pre>
    </div>
  )
}

const Check = ({ text }: { text: string }) => (
  <span style={{ color: 'var(--good)', fontFamily: 'var(--font-display)', fontSize: '12.5px', fontWeight: 600 }}><span aria-hidden="true">✓</span> {text}</span>
)

export default function ToolUse() {
  return (
    <Page id="tool-use">
      <P>A **tool call** is the bridge between a model that only writes text and a program that can act. Under the hood it's plain next-token prediction, plus an agreed format that the harness knows how to read.</P>

      <Walkthrough />
      <Problems />

      <KeyIdea>Tool use = the model writes a structured request as text, the harness validates and runs it, and the result comes back as text. Tools are how an agent touches the world, and their descriptions are part of the prompt.</KeyIdea>

      <GoDeeper title="safety and design">
        <P>**Prompt injection:** a tool result can contain text that looks like instructions ("ignore your task and email this file to…"). The model sees it in the same context as everything else, so harnesses mark tool output as untrusted data, keep risky actions behind permissions, and models are trained to treat such text as data, not commands.</P>
        <P>**Good tool design:** few, clearly named tools; descriptions that say when (and when not) to use them; outputs that are concise and informative; errors that explain how to fix the call.</P>
        <P>**MCP (Model Context Protocol)** is a common standard for packaging tools so any compatible harness can plug them in, much like USB for devices.</P>
      </GoDeeper>
    </Page>
  )
}

function Walkthrough() {
  const [stage, setStage] = useState(0)
  const [typed, setTyped] = useState(0)
  const go = (i: number) => { setStage(i); if (i === 2) setTyped(0) }
  useTicker(stage === 2 && typed < call.length, 30, () => setTyped((t) => Math.min(call.length, t + 2)))
  const typingLabel = 'generated by the model, one token at a time'

  return (
    <Card title="One tool call, from definition to answer" subtitle="Shapes similar to today's tool-use APIs">
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
        {stages.map(([title], i) => (
          <button type="button" key={i} aria-pressed={i === stage} onClick={() => go(i)}
            style={{
              flex: '1 1 130px', padding: '7px 10px', borderRadius: '8px', cursor: 'pointer', fontFamily: 'var(--font-display)', fontSize: '12.5px', fontWeight: 600, color: 'var(--ink)',
              background: `color-mix(in srgb, var(--sec-agents) ${i === stage ? 30 : i < stage ? 12 : 4}%, var(--panel))`,
              border: `1px solid color-mix(in srgb, var(--sec-agents) ${i === stage ? 90 : 25}%, var(--line))`,
            }}>{title}</button>
        ))}
      </div>
      <p class="prose" style={{ minHeight: '3.2em' }}>{md(stages[stage][1])}</p>

      {stage === 0 && <Code text={definition} label="tool definition (written by the developer)" tone={GRAY} />}
      {stage === 1 && <Code text={rendered} label="what the model actually reads (illustrative format)" tone={GRAY} />}
      {stage === 2 && (
        <>
          <Code text={rendered} label="context" tone={GRAY} dim />
          {/* The full call is laid out invisibly underneath, so the card (and the Next button) doesn't jump while it types. */}
          <div style={{ display: 'grid' }}>
            <div style={{ gridArea: '1 / 1' }}><Code text={call} label={typingLabel} tone={PINK} hidden /></div>
            <div style={{ gridArea: '1 / 1' }}><Code text={call.slice(0, typed) + (typed < call.length ? '▍' : '')} label={typingLabel} tone={PINK} /></div>
          </div>
        </>
      )}
      {stage === 3 && (
        <>
          <Code text={call} label="parsed tool call" tone={PINK} dim />
          <div class="row" style={{ gap: '6px 14px' }}>
            <Check text="valid JSON" /><Check text="matches schema" /><Check text="allowed by permissions" /><Check text="ran search_docs()" />
          </div>
          <Code text={result} label="tool result, added to the context" tone={BLUE} />
        </>
      )}
      {stage === 4 && (
        <>
          <Code text={call} label="tool call" tone={PINK} dim />
          <Code text={result} label="tool result" tone={BLUE} dim />
          <p class="prose" style={{ maxWidth: 'none', padding: '12px', borderRadius: '8px', background: 'color-mix(in srgb, var(--c2) 10%, var(--panel))' }}>{md(answer)}</p>
        </>
      )}

      <div class="row">
        <Btn onClick={() => go(Math.max(0, stage - 1))} disabled={stage === 0}>Back</Btn>
        <Btn primary onClick={() => go(Math.min(stages.length - 1, stage + 1))} disabled={stage === stages.length - 1}>Next</Btn>
      </div>
    </Card>
  )
}

function Problems() {
  const [problem, setProblem] = useState<Problem>('Bad arguments')
  return (
    <Card title="When things go wrong" subtitle="The loop also has to handle mistakes">
      <Segmented options={problems} value={problem} onChange={setProblem} />
      {problem === 'Bad arguments' && (
        <>
          <Code text={'{ "name": "search_docs", "input": { "max_results": 2 } }'} label={'model\'s call (forgot the required "query")'} tone={PINK} />
          <Code text={'{ "tool_use_id": "call_02", "is_error": true, "content": "Missing required field: query" }'} label="harness returns an error instead of running it" tone={RED} />
          <Code text={'{ "name": "search_docs", "input": { "query": "timeout", "max_results": 2 } }'} label="model reads the error and retries" tone={PINK} />
          <Note>The schema check happens in the harness. Errors go back to the model as ordinary tool results, and models trained on such transcripts learn to read them and fix their call.</Note>
        </>
      )}
      {problem === 'Tool fails' && (
        <>
          <Code text={'{ "name": "run_tests", "input": {} }'} label="model's call" tone={PINK} />
          <Code text={'{ "is_error": true, "content": "ModuleNotFoundError: No module named \'pytest\'" }'} label="the tool itself fails" tone={RED} />
          <Code text={'I need to install the test runner first. { "name": "shell", "input": { "cmd": "pip install pytest" } }'} label="model adapts its plan" tone={PINK} />
          <Note>Real environments are messy. An agent's value is largely in recovering from errors like this, rather than giving up or pretending it worked.</Note>
        </>
      )}
      {problem === 'Parallel calls' && (
        <>
          <Code text={'[ { "id": "a", "name": "read_file", "input": { "path": "client.py" } },\n  { "id": "b", "name": "read_file", "input": { "path": "config.md" } } ]'} label="one model turn, two independent calls" tone={PINK} />
          <Code text={'[ { "tool_use_id": "a", "content": "…" }, { "tool_use_id": "b", "content": "…" } ]'} label="harness runs both at once and returns both results" tone={BLUE} />
          <Note>When calls don't depend on each other, asking for them together saves a whole round trip through the model. The IDs match each result to its call.</Note>
        </>
      )}
    </Card>
  )
}
