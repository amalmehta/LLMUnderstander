import { ComponentChildren, Fragment, JSX } from 'preact'
import { useEffect, useRef, useState } from 'preact/hooks'
import { chapterById, chapters, ChapterInfo, secVar } from '../chapters'
import { Trainer } from '../engine/trainer'
import { fmt } from '../engine/toys'

// ---------------------------------------------------------------- shared trainer

const automated = typeof location !== 'undefined' && /[?&]fresh\b/.test(location.search)
/** The one tiny model the whole site looks inside. */
export const trainer = new Trainer(!automated)

/** Re-render whenever the trainer changes. */
export function useTrainer(): Trainer {
  const [, setTick] = useState(0)
  useEffect(() => trainer.subscribe(() => setTick((n) => n + 1)), [])
  return trainer
}

/** Call `fn` every `ms` milliseconds, but only while `active` — idle pages use no CPU. */
export function useTicker(active: boolean, ms: number, fn: () => void) {
  const ref = useRef(fn)
  ref.current = fn
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => ref.current(), ms)
    return () => clearInterval(id)
  }, [active, ms])
}

/** Measure an element's width so charts can draw at real pixel size. */
export function useWidth<T extends HTMLElement>(fallback = 600): [preact.RefObject<T>, number] {
  const ref = useRef<T>(null)
  const [w, setW] = useState(fallback)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, Math.floor(e.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

// ---------------------------------------------------------------- tiny markdown

/** Renders **bold**, *italic* and `code` inside a plain string. */
export function md(text: string): JSX.Element {
  const parts: (string | JSX.Element)[] = []
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g
  let last = 0, m: RegExpExecArray | null, k = 0
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index))
    const s = m[0]
    if (s.startsWith('**')) parts.push(<b key={k++}>{s.slice(2, -2)}</b>)
    else if (s.startsWith('`')) parts.push(<code key={k++}>{s.slice(1, -1)}</code>)
    else parts.push(<em key={k++}>{s.slice(1, -1)}</em>)
    last = m.index + s.length
  }
  if (last < text.length) parts.push(text.slice(last))
  return <>{parts}</>
}

// ---------------------------------------------------------------- page scaffold

export function Page({ id, children }: { id: string; children: ComponentChildren }) {
  const c = chapterById(id)!
  const prev = chapters[c.number - 2], next = chapters[c.number]
  useEffect(() => { document.title = `${c.title} · LLM Understander` }, [id])
  return (
    <article class="page" style={{ '--sec': secVar(c.section) }}>
      <header class="page-head">
        <div class="eyebrow">Chapter {c.number} · {sectionTitle(c)}</div>
        <h1>{c.title}</h1>
        <p class="tagline">{c.tagline}</p>
      </header>
      {children}
      <nav class="pager" aria-label="Chapter navigation">
        {prev ? <a href={`#${prev.id}`}>← {prev.title}</a> : <span />}
        {next && <a class="next" href={`#${next.id}`} style={{ '--sec': secVar(next.section) }}>Next: {next.title} →</a>}
      </nav>
    </article>
  )
}

const sectionTitle = (c: ChapterInfo) => ({ start: 'Start Here', blocks: 'Building Blocks', learning: 'How It Learns', pretraining: 'Pretraining', posttraining: 'Post-training', wrap: 'Putting It Together', agents: 'Agents & Harnesses' })[c.section]

export const P = ({ children }: { children: ComponentChildren }) => <p class="prose">{typeof children === 'string' ? md(children) : children}</p>

export function Card({ title, subtitle, children, style }: { title?: string; subtitle?: string; children: ComponentChildren; style?: JSX.CSSProperties }) {
  return (
    <section class="card" style={style}>
      {title && <div class="card-head"><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>}
      {children}
    </section>
  )
}

export function KeyIdea({ children }: { children: string }) {
  return (
    <aside class="keyidea">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2a7 7 0 0 0-4 12.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26A7 7 0 0 0 12 2Zm-2 18h4v1a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1v-1Z" /></svg>
      <p>{md(children)}</p>
    </aside>
  )
}

export function GoDeeper({ title, children }: { title: string; children: ComponentChildren }) {
  return (
    <details class="deeper">
      <summary><span class="fx">ƒ</span> Go deeper: {title}</summary>
      <div class="deeper-body">{children}</div>
    </details>
  )
}

export const Formula = ({ children }: { children: string }) => <pre class="formula">{children}</pre>
export const Note = ({ children }: { children: string }) => <p class="note">{md(children)}</p>
export const Callout = ({ tone = 'var(--accent)', children }: { tone?: string; children: string }) => <p class="callout" style={{ '--tone': tone }}>{md(children)}</p>

export function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return <div class="stat" style={tone ? { '--tone': tone } : undefined}><small>{label}</small><span>{value}</span></div>
}

export function Chip({ text, color = 'var(--c1)', sub }: { text: string; color?: string; sub?: string }) {
  return <span class="chip" style={{ '--chip': color }}>{text}{sub !== undefined && <small>{sub}</small>}</span>
}

const palette = ['var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)', 'var(--c5)', 'var(--c6)', 'var(--c7)', 'var(--c8)']
export const tokenColor = (i: number) => palette[i % palette.length]

// ---------------------------------------------------------------- controls

export function Btn({ children, onClick, primary, disabled, small, title }: { children: ComponentChildren; onClick: () => void; primary?: boolean; disabled?: boolean; small?: boolean; title?: string }) {
  return <button type="button" class={`btn${primary ? ' primary' : ''}${small ? ' small' : ''}`} onClick={onClick} disabled={disabled} title={title}>{children}</button>
}

export function Segmented<T extends string | number>({ options, value, onChange, labels }: { options: readonly T[]; value: T; onChange: (v: T) => void; labels?: (v: T) => string }) {
  return (
    <div class="seg" role="group">
      {options.map((o) => <button type="button" key={String(o)} aria-pressed={o === value} onClick={() => onChange(o)}>{labels ? labels(o) : String(o)}</button>)}
    </div>
  )
}

export function Slider({ id, label, value, min, max, step = 'any', onInput }: { id: string; label: string; value: number; min: number; max: number; step?: number | 'any'; onInput: (v: number) => void }) {
  return (
    <label class="slider" for={id}>
      <span>{label}</span>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onInput={(e) => onInput(+(e.currentTarget as HTMLInputElement).value)} />
    </label>
  )
}

export function Toggle({ id, label, checked, onChange }: { id: string; label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label class="toggle" for={id}>
      <input id={id} type="checkbox" checked={checked} onChange={(e) => onChange((e.currentTarget as HTMLInputElement).checked)} />
      {label}
    </label>
  )
}

export function TextInput({ id, value, onInput, mono = true, placeholder }: { id: string; value: string; onInput: (v: string) => void; mono?: boolean; placeholder?: string }) {
  return <input id={id} class={`text${mono ? ' mono' : ''}`} value={value} placeholder={placeholder} spellcheck={false} onInput={(e) => onInput((e.currentTarget as HTMLInputElement).value)} />
}

// ---------------------------------------------------------------- flows

export interface FlowStep { title: string; detail?: string; tone: string; href?: string }
export function Flow({ steps }: { steps: FlowStep[] }) {
  return (
    <div class="flow">
      {steps.map((s, i) => {
        const body = <><b>{s.title}</b>{s.detail && <small>{s.detail}</small>}</>
        return (
          <Fragment key={i}>
            {s.href ? <a class="step" href={s.href} style={{ '--tone': s.tone }}>{body}</a> : <div class="step" style={{ '--tone': s.tone }}>{body}</div>}
            {i < steps.length - 1 && <span class="arrow" aria-hidden="true">→</span>}
          </Fragment>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------- model banner

export function ModelBanner() {
  const t = useTrainer()
  const tone = t.isTrained ? 'var(--good)' : 'var(--warn)'
  return (
    <div class="banner" style={{ '--tone': tone }}>
      <svg width="20" height="20" viewBox="0 0 24 24" fill={tone} aria-hidden="true">
        {t.isTrained ? <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm-1.2 14.2-4-4 1.4-1.4 2.6 2.6 5.6-5.6 1.4 1.4-7 7Z" /> : <path d="M1 21h22L12 2 1 21Zm12-3h-2v-2h2v2Zm0-4h-2v-4h2v4Z" />}
      </svg>
      <div class="grow">
        <b>{t.isTrained ? `Using the tiny model after ${t.step.toLocaleString()} training steps.` : t.step === 0 ? 'The tiny model is untrained — its insides are random noise.' : `The tiny model has only trained ${t.step} steps.`}</b>
        <small>Everything below is computed live in your browser by a real {fmt.big(t.parameterCount)}-parameter transformer.</small>
      </div>
      {t.isRunning ? <span>training… step {t.step}</span> : !t.finished && (
        <>
          <Btn onClick={() => t.start()}>{t.isTrained ? 'Keep training' : 'Train it now'}</Btn>
          <a href="#training-loop">Watch it learn</a>
        </>
      )}
    </div>
  )
}
