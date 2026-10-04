import { useEffect, useRef, useState } from 'preact/hooks'
import { Actor, AgentEvent, AgentScript, Kind } from '../engine/agents'
import { Btn, Callout, Card, Formula, GoDeeper, KeyIdea, P, Page, Stat, useTicker, useWidth } from '../ui/kit'

const events = AgentScript.events
const last = events.length - 1

/** Colour of each kind of transcript message. */
export function kindColor(k: Kind): string {
  switch (k) {
    case 'system prompt': case 'tool definitions': return 'var(--c8)'
    case 'user message': return 'var(--c5)'
    case 'assistant text': case 'final answer': return 'var(--c2)'
    case 'tool call': return 'var(--c4)'
    case 'permission check': return 'var(--warn)'
    case 'tool result': return 'var(--c1)'
  }
}

/** One message in an agent transcript. */
export function EventBubble({ event, highlighted = false, dimmed = false }: { event: AgentEvent; highlighted?: boolean; dimmed?: boolean }) {
  const mono = event.kind === 'tool call' || event.kind === 'tool result' || event.kind === 'tool definitions'
  return (
    <div class={`bubble${highlighted ? ' hi' : ''}${dimmed ? ' dim' : ''}`} style={{ '--tone': kindColor(event.kind) }}>
      <header><span>{event.kind}</span><small>~{AgentScript.tokens(event)} tok</small></header>
      <div class={`body${mono ? ' mono' : ''}`}>{event.content}</div>
    </div>
  )
}

const lanes: { name: string; color: string; actors: Actor[] }[] = [
  { name: 'User & Harness', color: 'var(--c8)', actors: ['User', 'Harness'] },
  { name: 'Model', color: 'var(--c2)', actors: ['Model'] },
  { name: 'Tools', color: 'var(--c1)', actors: ['Tool'] },
]
const laneOf = (e: AgentEvent) => lanes.findIndex((l) => l.actors.includes(e.actor))

export default function AgentLoop() {
  const [step, setStep] = useState(2)
  const [playing, setPlaying] = useState(false)
  const [wrapRef, width] = useWidth<HTMLDivElement>()
  const scrollRef = useRef<HTMLDivElement>(null)
  const wide = width >= 560

  useTicker(playing, 1600, () => {
    if (step < last) setStep(step + 1)
    else setPlaying(false)
  })

  // Keep the newest message in view.
  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [step, wide])

  return (
    <Page id="agent-loop">
      <P>Everything so far produces a model that does one thing: given text, predict the next token. It can't open a file, run code or search the web. It can only *write*.</P>
      <P>An **agent** is that same model placed inside a loop by a program called a **harness**. The model writes a request for a tool, the harness runs the tool, and the result goes back into the model's context. Then the model is asked again. The loop repeats until the model answers without asking for a tool.</P>

      <Card title="The agent loop">
        <AgentLoopDiagram />
        <Formula>{`context = [system_prompt, tool_definitions, user_message]
while True:
    reply = model.generate(context)          # next-token prediction, as in Chapter 6
    context.append(reply)
    if not reply.tool_calls: break           # plain answer → done
    for call in reply.tool_calls:
        if harness.allowed(call):            # permissions live here, not in the model
            context.append(harness.run(call))  # tool result becomes new context
show(reply)`}</Formula>
      </Card>

      <Card title="Watch a real-shaped agent run" subtitle="A coding agent fixes a failing test. Step through every message that passes between harness, model and tools.">
        <div class="row">
          <Btn onClick={() => { setStep(Math.max(0, step - 1)); setPlaying(false) }} disabled={step === 0}>Back</Btn>
          <Btn primary onClick={() => setStep(Math.min(last, step + 1))} disabled={step === last}>Next</Btn>
          <Btn onClick={() => { if (step === last) setStep(0); setPlaying(!playing) }}>{playing ? 'Pause' : 'Play'}</Btn>
          <Btn onClick={() => { setStep(0); setPlaying(false) }}>Restart</Btn>
        </div>
        <div class="stats">
          <Stat label="Step" value={`${step + 1} / ${events.length}`} tone="var(--good)" />
          <Stat label="Context so far" value={`${AgentScript.contextTokens(step)} tokens`} tone="var(--c1)" />
          <Stat label="Model calls" value={String(AgentScript.modelCalls(step))} tone="var(--c2)" />
        </div>

        <div aria-live="polite"><Callout tone="var(--good)">{events[step].note}</Callout></div>

        <div ref={wrapRef}>
          {wide && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '8px', marginBottom: '6px' }}>
              {lanes.map((l) => <div key={l.name} style={{ textAlign: 'center', font: '600 12.5px var(--font-display)', color: l.color }}>{l.name}</div>)}
            </div>
          )}
          <div ref={scrollRef} style={{ height: '380px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', padding: '4px 2px' }}>
            {events.slice(0, step + 1).map((e) => {
              const lane = laneOf(e)
              return wide ? (
                <div key={e.id} style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '8px' }}>
                  <div style={{ gridColumn: lane + 1 }}><EventBubble event={e} highlighted={e.id === step} /></div>
                </div>
              ) : (
                <div key={e.id} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  <span style={{ font: '600 11.5px var(--font-display)', color: lanes[lane].color }}>{lanes[lane].name}</span>
                  <EventBubble event={e} highlighted={e.id === step} />
                </div>
              )
            })}
          </div>
        </div>
      </Card>

      <KeyIdea>The model never runs anything. It writes text that *describes* an action; the harness decides whether to do it and reports back. An agent is a language model, a loop, and a set of tools — nothing more mysterious than that.</KeyIdea>

      <GoDeeper title="what changes compared with a chatbot">
        <P>{`**Many model calls per request.** This small task took ${AgentScript.modelCalls(last)} separate calls to the model. Each call re-reads the entire context (prompt caching makes that cheaper), so agents use far more tokens than chat.`}</P>
        <P>**The context is the agent's only working memory.** Everything the model knows about the task — the error message, the file, its own earlier plan — must be in the context window. Chapter 15 is about managing that.</P>
        <P>**Errors compound.** If each step is right 95% of the time, a 20-step task succeeds only about 36% of the time (0.95²⁰). That's why checking work, like re-running the tests here, matters so much, and why agents are trained specifically for long tasks (Chapter 17).</P>
      </GoDeeper>
    </Page>
  )
}

// ---------------------------------------------------------------- diagram

function Box({ title, detail, tone }: { title: string; detail: string; tone: string }) {
  return (
    <div style={{ padding: '10px', borderRadius: '12px', minHeight: '80px', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '4px', minWidth: 0,
      background: `color-mix(in srgb, ${tone} 10%, var(--panel))`, border: `1px solid color-mix(in srgb, ${tone} 40%, var(--line))`, fontFamily: 'var(--font-display)' }}>
      <b style={{ color: tone, fontSize: '16px' }}>{title}</b>
      <small style={{ color: 'var(--muted)', fontSize: '12px', whiteSpace: 'pre-line', lineHeight: 1.35 }}>{detail}</small>
    </div>
  )
}

const tiny = { fontSize: '10px', color: 'var(--muted)', fontFamily: 'var(--font-display)' }

/** Two labelled arrows between neighbouring boxes; horizontal (→ ←) or vertical (↓ ↑). */
function Arrows({ top, bottom, vertical = false }: { top: string; bottom: string; vertical?: boolean }) {
  return vertical ? (
    <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', alignItems: 'center', gap: '2px 6px', padding: '3px 0', textAlign: 'center' }} aria-hidden="true">
      <span style={tiny}>{top}</span><span style={{ color: 'var(--muted)' }}>↓↑</span><span style={tiny}>{bottom}</span>
    </div>
  ) : (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1px', textAlign: 'center' }} aria-hidden="true">
      <span style={tiny}>{top}</span><span style={{ color: 'var(--muted)', lineHeight: 1 }}>→</span><span style={{ color: 'var(--muted)', lineHeight: 1 }}>←</span><span style={tiny}>{bottom}</span>
    </div>
  )
}

/** User ⇄ Harness ⇄ Model, Harness ⇄ Tools. Stacks vertically on narrow screens. */
function AgentLoopDiagram() {
  const [ref, width] = useWidth<HTMLDivElement>()
  const user = <Box title="User" detail={'asks for something,\napproves risky actions'} tone="var(--c5)" />
  const harness = <Box title="Harness" detail={'builds the context,\nruns the loop, checks\npermissions, runs tools'} tone="var(--good)" />
  const model = <Box title="Model" detail={'next-token prediction —\nwrites text or a\ntool call'} tone="var(--c2)" />
  const tools = <Box title="Tools" detail="files, shell, tests, search, browser…" tone="var(--c1)" />
  return (
    <div ref={ref} role="img" aria-label="The user talks to the harness; the harness sends context to the model and receives text and tool calls; the harness calls tools and receives results.">
      {width >= 560 ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 92px minmax(0, 1fr) 92px minmax(0, 1fr)', rowGap: '4px', alignItems: 'stretch' }}>
          {user}<Arrows top="request" bottom="answer" />{harness}<Arrows top="context" bottom="text + tool calls" />{model}
          <div style={{ gridColumn: '2 / 5', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
            <span style={{ ...tiny, display: 'flex', gap: '4px', alignItems: 'center' }}><span style={{ fontSize: '14px' }}>↑↓</span> calls / results</span>
            <div style={{ width: '230px', maxWidth: '100%' }}>{tools}</div>
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {user}
          <Arrows vertical top="request" bottom="answer" />
          {harness}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '8px' }}>
            <div><Arrows vertical top="context" bottom="text + tool calls" />{model}</div>
            <div><Arrows vertical top="calls" bottom="results" />{tools}</div>
          </div>
        </div>
      )}
    </div>
  )
}
