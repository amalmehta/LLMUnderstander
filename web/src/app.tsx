import { useEffect, useState } from 'preact/hooks'
import { chapterById, chapters, sections, secVar } from './chapters'
import { chapterViews } from './chapters/index'
import { Btn, Segmented } from './ui/kit'

const REPO = 'https://github.com/amalmehta/LLMUnderstander'

function currentId() {
  const h = location.hash.replace(/^#/, '')
  return chapterById(h) ? h : chapters[0].id
}

export function App() {
  const [id, setId] = useState(currentId)
  const [drawer, setDrawer] = useState(false)
  const [feedback, setFeedback] = useState(false)

  useEffect(() => {
    const on = () => { setId(currentId()); setDrawer(false); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  const c = chapterById(id)!
  const View = chapterViews[id]

  return (
    <div class="shell">
      <header class="topbar">
        <button type="button" class="btn small" onClick={() => setDrawer(true)} aria-label="Open chapter list">☰ Chapters</button>
        <b>{c.number}. {c.title}</b>
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
              <h4>{s.title}</h4>
              {chapters.filter((ch) => ch.section === s.id).map((ch) => (
                <a key={ch.id} href={`#${ch.id}`} aria-current={ch.id === id ? 'page' : undefined} style={{ '--sec': secVar(ch.section) }}>
                  <span class="n">{ch.number}</span><span>{ch.title}</span>
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
        </div>
      </nav>
      <main class="main" id="main">
        <View key={id} />
      </main>
      {feedback && <FeedbackSheet chapter={`Chapter ${c.number}: ${c.title}`} onClose={() => setFeedback(false)} />}
    </div>
  )
}

/** The app icon: a causal-attention triangle. */
function Logo() {
  const cells = []
  for (let r = 0; r < 4; r++) for (let c = 0; c <= r; c++) {
    cells.push(<rect key={`${r}${c}`} x={5 + c * 5.5} y={5 + r * 5.5} width="4.4" height="4.4" rx="1" fill="#fff" opacity={c === r ? 0.95 : 0.4} />)
  }
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
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
