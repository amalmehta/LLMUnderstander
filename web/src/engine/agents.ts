import { Num, Rng } from './tensor'
import type { Point } from './trainer'

// =====================================================================
// A scripted, realistic agent run: a coding agent fixing a failing test.
// =====================================================================

export type Actor = 'Harness' | 'User' | 'Model' | 'Tool'
export type Kind = 'system prompt' | 'tool definitions' | 'user message' | 'assistant text' | 'tool call' | 'permission check' | 'tool result' | 'final answer'

export interface AgentEvent { id: number; actor: Actor; kind: Kind; content: string; note: string }

const raw: [Actor, Kind, string, string][] = [
  ['Harness', 'system prompt', "You are a coding agent working in the user's repository. Use the tools to inspect and change files. Run the tests before you finish. Be concise.",
    'Before anything else, the harness writes a system prompt. The model never sees the app, the files or the terminal, only this text and what follows.'],
  ['Harness', 'tool definitions', 'read_file(path: string) → file contents\nedit_file(path: string, old: string, new: string) → ok | error\nrun_tests() → test output',
    'The harness describes each tool: a name, what it does and a JSON schema for its arguments. To the model these are just more tokens in the prompt.'],
  ['User', 'user message', 'The test for average() is failing. Can you fix it?',
    "The user's request is appended. Now the harness sends everything so far to the model and asks for the next message."],
  ['Model', 'assistant text', "I'll start by running the tests to see the failure.",
    'The model predicts tokens exactly as in Chapter 6. Some of them are ordinary text…'],
  ['Model', 'tool call', '{"tool": "run_tests", "input": {}}',
    '…and some form a structured tool call. The model has stopped generating and is waiting. It cannot run anything itself; it has only *asked*.'],
  ['Harness', 'permission check', 'run_tests: read-only → allowed automatically',
    'The harness, not the model, decides whether the call may run. Running tests is on the allow-list, so no human is asked.'],
  ['Tool', 'tool result', 'FAILED test_stats.py::test_average\n    assert average([1, 2, 3, 4]) == 2.5\n    AssertionError: got 2',
    'The harness runs the tool and pastes the output into the context as a tool result. Then it calls the model again with the longer context.'],
  ['Model', 'tool call', '{"tool": "read_file", "input": {"path": "stats.py"}}',
    'A new forward pass over the whole context. Seeing the failure, the most likely next tokens are a call to read the code.'],
  ['Harness', 'permission check', 'read_file: read-only → allowed automatically', "Reads are safe, so they're allowed."],
  ['Tool', 'tool result', 'def average(xs):\n    return sum(xs) // len(xs)',
    "The file contents go into the context. Tool results are often the biggest thing in an agent's context."],
  ['Model', 'assistant text', 'The bug is integer division (//). It should be true division (/).',
    'The model reasons in text. This is still just next-token prediction, conditioned on everything above.'],
  ['Model', 'tool call', '{"tool": "edit_file", "input": {"path": "stats.py", "old": "sum(xs) // len(xs)", "new": "sum(xs) / len(xs)"}}',
    "An edit changes the user's files, so this call is riskier."],
  ['Harness', 'permission check', 'edit_file: modifies files → ask the user… approved',
    "Edits aren't on the allow-list, so the harness pauses and asks the human. Permissions live in the harness, outside the model."],
  ['Tool', 'tool result', 'ok: 1 replacement in stats.py',
    'The edit happened in the real world (well, the real file system), and the model is told it succeeded.'],
  ['Model', 'tool call', '{"tool": "run_tests", "input": {}}', 'Good agents check their work. The model asks to run the tests again.'],
  ['Harness', 'permission check', 'run_tests: read-only → allowed automatically', 'Allowed.'],
  ['Tool', 'tool result', '1 passed in 0.02s', 'Verified.'],
  ['Model', 'final answer', 'Fixed it: average() used integer division (//), which truncated 2.5 to 2. I changed it to true division (/) and the test now passes.',
    "This time the model writes an answer with no tool call. That is the harness's signal to stop the loop and show the reply to the user."],
]

export const AgentScript = {
  events: raw.map(([actor, kind, content, note], id): AgentEvent => ({ id, actor, kind, content, note })),
  inContext: (e: AgentEvent) => e.kind !== 'permission check',
  trainedOn: (e: AgentEvent) => e.actor === 'Model',
  /** Rough token count: ≈ 4 characters per token. */
  tokens: (e: AgentEvent) => Math.max(1, Math.floor(e.content.length / 4)),
  contextTokens(upTo: number) {
    return AgentScript.events.slice(0, upTo + 1).filter(AgentScript.inContext).reduce((s, e) => s + AgentScript.tokens(e), 0)
  },
  modelCalls(upTo: number) {
    let n = 0, prev = false
    for (const e of AgentScript.events.slice(0, upTo + 1)) { const m = e.actor === 'Model'; if (m && !prev) n++; prev = m }
    return n
  },
}

// =====================================================================
// Context budget in a long simulated session
// =====================================================================

export const ContextBudget = {
  window: 200_000, system: 3_000, tools: 12_000, memory: 4_000,
  turnTokens(t: number) {
    const r = new Rng(t * 2654435761 + 7)
    const base = 1500 + r.uniform() * 2500
    return base + (r.uniform() < 0.25 ? 6000 + r.uniform() * 14000 : 0)
  },
  segments(turns: number, compaction: boolean, threshold = 0.8) {
    let conversation = 0, summary = 0, compactions = 0
    const fixed = this.system + this.tools + this.memory
    for (let t = 0; t < turns; t++) {
      conversation += this.turnTokens(t)
      if (compaction && fixed + summary + conversation > threshold * this.window) {
        compactions++; summary = Math.min(summary + 4000, 15000); conversation *= 0.1
      }
    }
    const segs = [
      { id: 'System prompt', tokens: this.system }, { id: 'Tool definitions', tokens: this.tools }, { id: 'Memory files', tokens: this.memory },
      { id: 'Summary of earlier work', tokens: summary }, { id: 'Conversation + tool results', tokens: conversation },
    ]
    return { segments: segs, compactions, overflow: fixed + summary + conversation > this.window }
  },
}

// =====================================================================
// RL on an agent task with a verifiable reward (REINFORCE with a baseline)
// =====================================================================

export interface Strategy { id: number; name: string; detail: string; passRate: number; tampers: boolean; genuine: boolean }

export const AgentRL = {
  strategies: [
    { id: 0, name: 'Investigate, then fix', detail: 'run tests → read code → fix → re-run', passRate: 0.7, tampers: false, genuine: true },
    { id: 1, name: 'Guess an edit', detail: 'change code without reading it', passRate: 0.3, tampers: false, genuine: true },
    { id: 2, name: 'Delete the failing test', detail: "tests now 'pass'", passRate: 1.0, tampers: true, genuine: false },
    { id: 3, name: 'Give up', detail: 'explain and stop', passRate: 0.0, tampers: false, genuine: true },
  ] as Strategy[],
  initialLogits: [0.5, 0.8, -0.5, 0.3],

  newState() { return { logits: [...AgentRL.initialLogits], baseline: 0, episodes: 0, lastRewards: [] as number[] } },

  step(s: { logits: number[]; baseline: number; episodes: number; lastRewards: number[] }, robust: boolean, rng: Rng, batch = 16, lr = 0.4) {
    const p = Num.softmax(s.logits), g = p.map(() => 0), rewards: number[] = []
    for (let k = 0; k < batch; k++) {
      const a = Num.sample(p, rng), st = AgentRL.strategies[a]
      const passed = rng.uniform() < st.passRate
      const r = robust && st.tampers ? 0 : passed ? 1 : 0
      rewards.push(r)
      for (let j = 0; j < g.length; j++) g[j] += ((r - s.baseline) * ((j === a ? 1 : 0) - p[j])) / batch
    }
    s.logits = s.logits.map((l, j) => l + lr * g[j])
    s.baseline = 0.9 * s.baseline + 0.1 * Num.sum(rewards) / batch
    s.episodes += batch
    s.lastRewards = rewards
  },
  genuineSuccess(p: number[]) { return p.reduce((s, pi, i) => s + pi * (AgentRL.strategies[i].genuine ? AgentRL.strategies[i].passRate : 0), 0) },
}

// =====================================================================
// Multi-armed bandits and "ask several times" maths
// =====================================================================

export type BanditStrategy = 'Greedy' | 'ε-greedy (10%)' | 'UCB1' | 'Thompson sampling'
export const banditStrategies: BanditStrategy[] = ['Greedy', 'ε-greedy (10%)', 'UCB1', 'Thompson sampling']
export const banditExplanation: Record<BanditStrategy, string> = {
  'Greedy': 'Try each option once, then always pick the best average so far. Never explores, so an unlucky start locks it onto a worse option forever.',
  'ε-greedy (10%)': 'Pick the best average 90% of the time and a random option 10% of the time. Simple and robust, but it keeps wasting 10% on options it already knows are bad.',
  'UCB1': 'Pick the option with the highest optimistic estimate: average + a bonus that shrinks as an option gets tried more. Uncertainty is treated as opportunity.',
  'Thompson sampling': "Keep a belief (a Beta distribution) about each option's success rate, draw one plausible rate from each belief, and pick the highest draw. Explores exactly as much as its uncertainty warrants.",
}

export interface BanditState { pulls: number[]; wins: number[]; regret: number; regretHistory: Point[]; t: number }

export const Bandit = {
  arms: [
    { id: 0, name: 'Prompt A', rate: 0.3 }, { id: 1, name: 'Prompt B', rate: 0.45 }, { id: 2, name: 'Prompt C', rate: 0.62 },
    { id: 3, name: 'Prompt D', rate: 0.55 }, { id: 4, name: 'Prompt E', rate: 0.5 },
  ],
  get bestRate() { return Math.max(...Bandit.arms.map((a) => a.rate)) },
  newState(): BanditState { const k = Bandit.arms.length; return { pulls: new Array(k).fill(0), wins: new Array(k).fill(0), regret: 0, regretHistory: [], t: 0 } },
  mean(st: BanditState, i: number) { return st.pulls[i] ? st.wins[i] / st.pulls[i] : 0 },

  choose(s: BanditStrategy, st: BanditState, rng: Rng): number {
    const k = st.pulls.length, idx = [...Array(k).keys()]
    const untried = st.pulls.indexOf(0)
    if (untried >= 0 && s !== 'Thompson sampling') return untried
    const best = (f: (i: number) => number) => idx.reduce((b, i) => (f(i) > f(b) ? i : b), 0)
    switch (s) {
      case 'Greedy': return best((i) => Bandit.mean(st, i))
      case 'ε-greedy (10%)': return rng.uniform() < 0.1 ? rng.int(k) : best((i) => Bandit.mean(st, i))
      case 'UCB1': return best((i) => Bandit.mean(st, i) + Math.sqrt((2 * Math.log(st.t)) / st.pulls[i]))
      case 'Thompson sampling': {
        const draws = idx.map((i) => Bandit.betaSample(st.wins[i] + 1, st.pulls[i] - st.wins[i] + 1, rng))
        return best((i) => draws[i])
      }
    }
  },

  step(s: BanditStrategy, st: BanditState, rng: Rng) {
    const a = Bandit.choose(s, st, rng)
    const win = rng.uniform() < Bandit.arms[a].rate
    st.pulls[a]++; if (win) st.wins[a]++
    st.t++
    st.regret += Bandit.bestRate - Bandit.arms[a].rate
    if (st.t % 5 === 0) st.regretHistory.push({ x: st.t, y: st.regret })
  },

  run(s: BanditStrategy, steps: number, seed: number) {
    const st = Bandit.newState(), rng = new Rng(seed)
    for (let i = 0; i < steps; i++) Bandit.step(s, st, rng)
    return st
  },

  gammaSample(shape: number, rng: Rng): number {
    if (shape < 1) return Bandit.gammaSample(shape + 1, rng) * Math.pow(Math.max(rng.uniform(), 1e-12), 1 / shape)
    const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d)
    for (;;) {
      let x: number, v: number
      do { x = rng.normal(); v = 1 + c * x } while (v <= 0)
      v = v * v * v
      const u = rng.uniform()
      if (u < 1 - 0.0331 * x ** 4 || Math.log(Math.max(u, 1e-300)) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
    }
  },
  betaSample(a: number, b: number, rng: Rng) { const x = Bandit.gammaSample(a, rng), y = Bandit.gammaSample(b, rng); return x / (x + y) },
}

export const Sampling = {
  choose(n: number, k: number) { let r = 1; for (let i = 0; i < k; i++) r *= (n - i) / (i + 1); return r },
  /** Majority of n independent attempts (n odd), each right with probability p. */
  majority(p: number, n: number) {
    let s = 0
    for (let k = Math.floor(n / 2) + 1; k <= n; k++) s += Sampling.choose(n, k) * p ** k * (1 - p) ** (n - k)
    return s
  },
  /** Best-of-n with a perfect checker. */
  anyCorrect(p: number, n: number) { return 1 - (1 - p) ** n },
}
