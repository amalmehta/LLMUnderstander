// Reading progress and guesses, kept in this browser only (localStorage).
// A chapter counts as read after 20 seconds on it, or on reaching its end.
import { useEffect, useState } from 'preact/hooks'

const KEY = 'llm-understander.progress.v1'
interface Store { read: string[]; guesses: Record<string, number>; last?: string }

const automated = typeof location !== 'undefined' && /[?&]fresh\b/.test(location.search)
let store: Store = load()
const listeners = new Set<() => void>()

function load(): Store {
  if (automated) return { read: [], guesses: {} }
  try { return { read: [], guesses: {}, ...JSON.parse(localStorage.getItem(KEY) || '{}') } } catch { return { read: [], guesses: {} } }
}
function save() {
  if (!automated) try { localStorage.setItem(KEY, JSON.stringify(store)) } catch { /* storage blocked: progress lasts this visit */ }
  listeners.forEach((f) => f())
}

export const progress = {
  get read() { return new Set(store.read) },
  get guesses() { return store.guesses },
  get last() { return store.last },
  get isFirstVisit() { return !store.last && store.read.length === 0 },
  markRead(id: string) { if (!store.read.includes(id)) { store = { ...store, read: [...store.read, id] }; save() } },
  setLast(id: string) { if (store.last !== id) { store = { ...store, last: id }; if (!automated) try { localStorage.setItem(KEY, JSON.stringify(store)) } catch { /* ignore */ } } },
  guess(id: string, option: number) { store = { ...store, guesses: { ...store.guesses, [id]: option } }; save() },
  clearGuess(id: string) { const g = { ...store.guesses }; delete g[id]; store = { ...store, guesses: g }; save() },
  reset() { store = { read: [], guesses: {} }; save() },
  subscribe(fn: () => void) { listeners.add(fn); return () => { listeners.delete(fn) } },
}

export function useProgress() {
  const [, set] = useState(0)
  useEffect(() => progress.subscribe(() => set((n) => n + 1)), [])
  return progress
}
