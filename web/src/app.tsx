import { useEffect, useState } from 'preact/hooks'
import { chapterById, chapters, menuTitle, numbered, sections, secVar } from './chapters'
import { progress, useProgress } from './progress'
import { trainer, useTrainer } from './ui/kit'
import { chapterViews } from './chapters/index'
import { Btn, Segmented } from './ui/kit'

const REPO = 'https://github.com/amalmehta/LLMUnderstander'

function currentId() {
  const h = location.hash.replace(/^#/, '')
  if (chapterById(h)) return h
  // First visit opens the Tour; after that, wherever the reader left off.
  return progress.last && chapterById(progress.last) ? progress.last : 'tour'
}

export function App() {
  const [id, setId] = useState(currentId)
  const [drawer, setDrawer] = useState(false)
  const [feedback, setFeedback] = useState(false)
  const [settings, setSettings] = useState(false)
  const p = useProgress()

  useEffect(() => {
    const on = () => { setId(currentId()); setDrawer(false); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  useEffect(() => { progress.setLast(id) }, [id])

  // [ and ] go to the previous and next chapter (not while typing).
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      if (e.metaKey || e.ctrlKey || e.altKey || el.closest('input, textarea, [contenteditable]')) return
      const i = chapters.findIndex((ch) => ch.id === currentId())
      if (e.key === ']' && i < chapters.length - 1) location.hash = chapters[i + 1].id
      if (e.key === '[' && i > 0) location.hash = chapters[i - 1].id
    }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [])

  const c = chapterById(id)!
  const View = chapterViews[id]

  return (
    <div class="shell">
      <header class="topbar">
        <button type="button" class="btn small" onClick={() => setDrawer(true)} aria-label="Open chapter list">☰ Chapters</button>
        <b>{menuTitle(c)}</b>
      </header>
      {drawer && <div class="scrim" onClick={() => setDrawer(false)} />}
      <nav class={`sidebar${drawer ? ' open' : ''}`} aria-label="Chapters">
        <a class="brand" href="#big-picture">
          <Logo />
          <b>LLM Understander</b>
        </a>
        <div class="toc">
          {sections.map((s) => (
            <div key={s.id}>
              {s.id !== 'tour' && <h4>{s.title}</h4>}
              {chapters.filter((ch) => ch.section === s.id).map((ch) => (
                <a key={ch.id} href={`#${ch.id}`} aria-current={ch.id === id ? 'page' : undefined} style={{ '--sec': secVar(ch.section) }}
                  class={ch.id === 'tour' ? 'tour-link' : undefined}>
                  <span class="n">{ch.id === 'tour' ? '▶' : ch.number}</span><span class="grow">{ch.title}</span>
                  {p.read.has(ch.id) && ch.id !== 'tour' && <span class="tick" aria-label="read">✓</span>}
                </a>
              ))}
            </div>
          ))}
        </div>
        <div class="sidebar-foot">
          <button type="button" class="linklike" onClick={() => setFeedback(true)}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 21l1.9-5.4A8 8 0 1 1 21 12Z" /></svg>
            Feedback
          </button>
          <button type="button" class="linklike" onClick={() => setSettings(true)} aria-label="Settings">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z" /></svg>
            Settings
          </button>
        </div>
      </nav>
      <main class="main" id="main">
        <View key={id} />
      </main>
      {feedback && <FeedbackSheet chapter={c.id === 'tour' ? 'Tour' : `Chapter ${c.number}: ${c.title}`} onClose={() => setFeedback(false)} />}
      {settings && <SettingsSheet onClose={() => setSettings(false)} />}
      <Opening />
    </div>
  )
}

/** The icon springs in and fades away once per visit; never blocks clicks; a plain fade with reduced motion. */
function Opening() {
  const [gone, setGone] = useState(false)
  useEffect(() => { const t = setTimeout(() => setGone(true), 1000); return () => clearTimeout(t) }, [])
  if (gone || /[?&]fresh\b/.test(location.search)) return null
  return (
    <div class="opening" aria-hidden="true">
      <div class="opening-mark"><Logo size={88} /><b>LLM Understander</b></div>
    </div>
  )
}

/** Everything the site keeps, in one place, with a way to delete it. */
function SettingsSheet({ onClose }: { onClose: () => void }) {
  const t = useTrainer()
  const p = useProgress()
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [])
  return (
    <div class="modal-wrap" onClick={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="set-title">
        <h3 id="set-title">Settings</h3>
        <h4>Your tiny model</h4>
        <p>{t.step === 0 ? 'Untrained.' : t.isTrained ? `Trained, ${t.step} steps.` : `${t.step} steps, not trained yet.`}
          {t.latestValLoss !== null && ` Validation loss ${t.latestValLoss.toFixed(2)} (random guessing: ${t.uniformLoss.toFixed(2)}).`}</p>
        <div class="row"><Btn disabled={t.step === 0} onClick={() => { if (confirm('Reset the tiny model? Training again takes about a minute.')) trainer.reset() }}>Reset model…</Btn></div>
        <h4>Reading</h4>
        <p>{numbered.filter((ch) => p.read.has(ch.id)).length} of {numbered.length} chapters read · {Object.keys(p.guesses).length} guesses made. A chapter counts as read after 20 seconds on it, or when you reach its end.</p>
        <div class="row"><Btn onClick={() => { if (confirm("Forget which chapters you've read and your guesses? Your model is kept.")) progress.reset() }}>Reset reading progress…</Btn></div>
        <h4>Privacy</h4>
        <p class="note">Everything stays in this browser's local storage: the model and your reading progress. Nothing is sent anywhere. Feedback opens a GitHub issue in a new tab, and nothing is sent until you submit it there. Animations follow your system's reduce-motion setting.</p>
        <div class="row" style={{ justifyContent: 'flex-end' }}><Btn primary onClick={onClose}>Done</Btn></div>
      </div>
    </div>
  )
}

/** The app icon: a causal-attention triangle. */
function Logo({ size = 30 }: { size?: number }) {
  const cells = []
  for (let r = 0; r < 4; r++) for (let c = 0; c <= r; c++) {
    cells.push(<rect key={`${r}${c}`} x={5 + c * 5.5} y={5 + r * 5.5} width="4.4" height="4.4" rx="1" fill="#fff" opacity={c === r ? 0.95 : 0.4} />)
  }
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4533b3" /><stop offset="1" stop-color="#149e9e" /></linearGradient></defs>
      <rect width="32" height="32" rx="7.5" fill="url(#lg)" />
      {cells}
    </svg>
  )
}

function FeedbackSheet({ chapter, onClose }: { chapter: string; onClose: () => void }) {
  const kinds = ['Idea', "Something's confusing", 'Bug'] as const
  const [kind, setKind] = useState<(typeof kinds)[number]>('Idea')
  const [text, setText] = useState('')
  const url = `${REPO}/issues/new?title=${encodeURIComponent(`${kind}: ${chapter} (website)`)}&body=${encodeURIComponent(`${text}\n\n---\nChapter: ${chapter}\nFrom: website`)}`
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [])
  return (
    <div class="modal-wrap" onClick={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="fb-title">
        <h3 id="fb-title">Send feedback</h3>
        <p class="note">This opens a pre-filled GitHub issue in a new tab. Nothing is sent until you submit it there.</p>
        <Segmented options={kinds} value={kind} onChange={setKind} />
        <textarea id="fb-text" class="text" rows={6} value={text} placeholder="What could be clearer, or what went wrong?" onInput={(e) => setText((e.currentTarget as HTMLTextAreaElement).value)} />
        <div class="row" style={{ justifyContent: 'space-between' }}>
          <span class="note">About: {chapter}</span>
          <div class="row">
            <Btn onClick={onClose}>Cancel</Btn>
            {text.trim()
              ? <a class="btn primary" href={url} target="_blank" rel="noopener" onClick={onClose} style={{ textDecoration: 'none' }}>Open on GitHub</a>
              : <Btn primary disabled onClick={() => {}}>Open on GitHub</Btn>}
          </div>
        </div>
      </div>
    </div>
  )
}
