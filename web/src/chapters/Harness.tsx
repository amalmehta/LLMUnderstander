import { useState } from 'preact/hooks'
import { AgentScript, ContextBudget } from '../engine/agents'
import { fmt } from '../engine/toys'
import { LineChart } from '../ui/charts'
import { Callout, Card, GoDeeper, KeyIdea, Note, P, Page, Segmented, Slider, Stat, Toggle } from '../ui/kit'

type Part = 'system' | 'tools' | 'loop' | 'permissions' | 'memory' | 'context' | 'subagents' | 'hooks'

const parts: { id: Part; title: string; icon: string; text: string }[] = [
  { id: 'system', title: 'System prompt', icon: '❝', text: "The first text the model sees on every call: who it is, how to behave, the conventions to follow. It's the cheapest, most powerful lever a harness has. Change it and the same weights behave like a different product." },
  { id: 'tools', title: 'Tool definitions', icon: '⚒', text: 'A name, a description and a JSON schema for each tool. The model chooses tools *only* from these descriptions, so writing clear ones is a big part of building a good agent. Every definition costs context tokens on every call.' },
  { id: 'loop', title: 'The loop', icon: '↻', text: 'Call the model; if it asked for tools, run them, append the results and call again; stop when it answers in plain text. Also: retries on errors, a cap on the number of steps, streaming output to the user, and handling interruptions.' },
  { id: 'permissions', title: 'Permissions & sandbox', icon: '⛨', text: "The model proposes; the harness disposes. Rules decide which tool calls run automatically, which need a human's approval and which are forbidden. Running tools inside a **sandbox** (a container or restricted file system) limits the damage of a bad call." },
  { id: 'memory', title: 'Memory files', icon: '✎', text: 'Models remember nothing between sessions; the weights are frozen. Harnesses fake long-term memory by saving notes to files (project instructions, preferences, past decisions) and loading them into the context at the start of each session.' },
  { id: 'context', title: 'Context management', icon: '⇲', text: 'The context window is finite (here 200k tokens), and agent runs fill it fast with tool output. Harnesses truncate huge tool results, drop stale ones, and **compact**: replace old conversation with a model-written summary so work can continue.' },
  { id: 'subagents', title: 'Sub-agents', icon: '⚇', text: 'A main agent can hand a self-contained job ("search the codebase for X") to a fresh model call with its own clean context. Only the short conclusion comes back, which keeps the main context small and lets work run in parallel.' },
  { id: 'hooks', title: 'Hooks & integrations', icon: '⛓', text: 'Deterministic code that runs at fixed points: before a tool call (block a dangerous command), after an edit (run the formatter), at the end of a turn (run the tests). Integrations such as MCP servers plug in extra tools — GitHub, databases, browsers — without retraining the model.' },
]

const policies = ['Ask before every tool', 'Allow reads, ask for writes', 'Allow all, inside a sandbox'] as const
type Policy = (typeof policies)[number]

const segColors: Record<string, string> = {
  'System prompt': 'var(--c8)', 'Tool definitions': 'var(--c4)', 'Memory files': 'var(--c5)',
  'Summary of earlier work': 'var(--c6)', 'Conversation + tool results': 'var(--c1)',
}

export default function Harness() {
  return (
    <Page id="harness">
      <P>The **harness** is everything around the model: the program that assembles the prompt, offers tools, runs the loop, enforces permissions and manages memory. The weights decide *how well* the agent thinks; the harness decides what it can see and do.</P>
      <P>Put the same model in two different harnesses and you get two noticeably different agents. Much of the progress in practical agents has come from better harnesses, not just better models.</P>

      <Anatomy />
      <ContextCard />
      <PermissionCard />

      <KeyIdea>Model = what it can think. Harness = what it can see and do. The harness turns a text predictor into a careful, useful worker by controlling its context, its tools and its permissions.</KeyIdea>

      <GoDeeper title="engineering tricks">
        <P>**Prompt caching:** the start of the context (system prompt, tool definitions, early conversation) stays the same from call to call, so providers can cache its processed form. Re-sending it then costs a fraction of the price and time. Harnesses keep the context's beginning stable to benefit.</P>
        <P>**Just-in-time context:** rather than stuffing everything in up front, give the agent tools to *look things up* (search, list files, read a range of lines) so only what's needed enters the context.</P>
        <P>**Structured notes:** for long tasks, agents keep a running to-do list or progress file. It survives compaction and keeps the plan on track.</P>
      </GoDeeper>
    </Page>
  )
}

function Anatomy() {
  const [part, setPart] = useState<Part>('context')
  const cur = parts.find((p) => p.id === part)!
  return (
    <Card title="Anatomy of a harness" subtitle="Click a part">
      <div class="row top" style={{ gap: '20px' }}>
        <div style={{ flex: '1 1 300px', maxWidth: '360px', display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '8px' }}>
          {parts.map((p) => {
            const on = p.id === part
            return (
              <button type="button" key={p.id} aria-pressed={on} onClick={() => setPart(p.id)}
                style={{
                  textAlign: 'left', padding: '10px', borderRadius: '8px', cursor: 'pointer', fontFamily: 'var(--font-display)', fontSize: '13px', fontWeight: 600,
                  color: 'var(--ink)', display: 'flex', gap: '7px', alignItems: 'center',
                  background: `color-mix(in srgb, var(--sec-agents) ${on ? 25 : 8}%, var(--panel))`,
                  border: `1px solid color-mix(in srgb, var(--sec-agents) ${on ? 90 : 25}%, var(--line))`,
                }}>
                <span aria-hidden="true" style={{ color: 'var(--sec-agents)', width: '16px', textAlign: 'center' }}>{p.icon}</span>{p.title}
              </button>
            )
          })}
        </div>
        <div class="grow" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <h3 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: '20px' }}><span aria-hidden="true" style={{ color: 'var(--sec-agents)' }}>{cur.icon}</span> {cur.title}</h3>
          <P>{cur.text}</P>
        </div>
      </div>
    </Card>
  )
}

function ContextCard() {
  const [turns, setTurns] = useState(40)
  const [compaction, setCompaction] = useState(true)
  const r = ContextBudget.segments(turns, compaction)
  const total = (s: { tokens: number }[]) => s.reduce((a, b) => a + b.tokens, 0)
  const used = total(r.segments)
  const W = ContextBudget.window
  const scale = Math.max(W, used)
  const history: { x: number; y: number }[] = []
  for (let t = 0; t <= 100; t += 2) history.push({ x: t, y: total(ContextBudget.segments(t, compaction).segments) })

  return (
    <Card title="The context window fills up" subtitle="A long simulated session. Each turn adds messages and tool output (some tool outputs are huge).">
      <div class="row">
        <div class="grow"><Slider id="harness-turns" label={`After ${turns} turns`} value={turns} min={0} max={100} step={1} onInput={setTurns} /></div>
        <Toggle id="harness-compaction" label="Compaction" checked={compaction} onChange={setCompaction} />
      </div>
      <div style={{ position: 'relative', height: '30px', borderRadius: '6px', overflow: 'hidden', background: 'var(--bg)', border: '1px solid var(--line)', display: 'flex' }}
        role="img" aria-label={`Context used: ${Math.round((100 * used) / W)}% of the window`}>
        {r.segments.filter((s) => s.tokens > 0).map((s) => (
          <div key={s.id} title={`${s.id}: ${fmt.big(s.tokens)}`}
            style={{ width: `${(100 * Math.min(s.tokens, W)) / scale}%`, background: `color-mix(in srgb, ${segColors[s.id]} 70%, var(--panel))`, flex: 'none' }} />
        ))}
        <div style={{ position: 'absolute', top: 0, bottom: 0, width: '2px', background: 'var(--bad)', left: `calc(${(100 * W) / scale}% - 2px)` }} />
      </div>
      <div class="legend">
        {r.segments.map((s) => <span key={s.id}><i style={{ background: `color-mix(in srgb, ${segColors[s.id]} 70%, var(--panel))` }} />{s.id} {fmt.big(s.tokens)}</span>)}
      </div>
      <LineChart height={160} xDomain={[0, 100]} yDomain={[0, Math.max(W * 1.05, ...history.map((h) => h.y))]} xLabel="turn" yFormat={fmt.big}
        rules={[{ y: W, color: 'var(--bad)', label: 'context limit 200K' }, { x: turns, color: 'var(--muted)' }]}
        series={[{ name: 'tokens in context', color: 'var(--c1)', points: history }]} />
      <div class="row">
        <div class="stats">
          <Stat label="Context used" value={`${Math.round((100 * used) / W)}%`} tone={r.overflow ? 'var(--bad)' : 'var(--c1)'} />
          <Stat label="Compactions so far" value={String(r.compactions)} tone="var(--c6)" />
        </div>
        {r.overflow && <div class="grow"><Callout tone="var(--bad)">⚠️ Over the limit: without compaction the harness would have to drop messages or stop the task.</Callout></div>}
      </div>
      <Note>The system prompt, tool definitions and memory are paid for on *every* call. Compaction (at 80% full here) has the model summarise older work, then continues with the summary plus recent messages. Some detail is lost each time, which is why good notes matter.</Note>
    </Card>
  )
}

function PermissionCard() {
  const [policy, setPolicy] = useState<Policy>('Allow reads, ask for writes')
  const calls = AgentScript.events.filter((e) => e.kind === 'tool call')
  const outcome = (isWrite: boolean): [string, string, string] => {
    if (policy === 'Ask before every tool') return ['Asks the user', '✋', 'var(--warn)']
    if (policy === 'Allow reads, ask for writes') return isWrite ? ['Asks the user', '✋', 'var(--warn)'] : ['Runs automatically', '✓', 'var(--good)']
    return ['Runs automatically (sandboxed)', '▣', 'var(--c6)']
  }
  const asks = calls.filter((c) => outcome(c.content.includes('edit_file'))[0].startsWith('Asks')).length
  return (
    <Card title="Permissions: who gets asked?" subtitle="The tool calls from Chapter 14's run, under three policies">
      <Segmented options={policies} value={policy} onChange={setPolicy} />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {calls.map((c) => {
          const [label, icon, tone] = outcome(c.content.includes('edit_file'))
          return (
            <div key={c.id} class="row" style={{ gap: '4px 12px', padding: '8px', borderRadius: '6px', background: `color-mix(in srgb, ${tone} 8%, var(--panel))`, border: `1px solid color-mix(in srgb, ${tone} 20%, var(--line))` }}>
              <span class="mono" style={{ flex: '1 1 220px', minWidth: 0, fontSize: '11.5px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={c.content}>{c.content}</span>
              <span style={{ flex: '0 1 230px', color: tone, fontFamily: 'var(--font-display)', fontSize: '12.5px', fontWeight: 600 }}><span aria-hidden="true">{icon}</span> {label}</span>
            </div>
          )
        })}
      </div>
      <Note>{policy === 'Allow all, inside a sandbox'
        ? 'No interruptions. The agent works on its own, and the sandbox limits the damage if it does something wrong.'
        : `${asks} interruption${asks === 1 ? '' : 's'} in a 4-call task. Safer, but a long task can mean dozens of approvals. That's why reads are usually auto-allowed and only risky actions ask.`}</Note>
    </Card>
  )
}
