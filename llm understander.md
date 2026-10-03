PROJECT NAME: llm understander

META-INSTRUCTIONS:

<Read it all before acting. Ask about anything unclear, contradictory or
 underspecified — before starting and mid-build. Ask in the question widget
 (AskUserQuestion): related questions batched, concrete options, your
 recommendation first. Plain text only if the widget isn't available.>

<Don't expand scope. Anything not listed here is a proposal, including changes
 to this file — propose it, don't do it.>

<Prefer doing over describing: run the code, write the files, test it.>

<Always in scope, no proposal needed: when it goes on GitHub, a README that is
 easy to read at a glance — a line on what it is, then clear visuals
 (screenshots, a diagram or a chart), then links. Everything else goes in
 linked files: docs/INSTRUCTIONS.md (setup, run, use) and
 docs/FILE-STRUCTURE.md (what's where). Also a small unobtrusive feedback tab
 if what you're building is an application rather than a script.>

<If what you're building is an application, build it as a Mac app first; the
 website comes after, as its own step.>

<Name things the way a person would say them — "Goal Tracker", not
 goal_tracker — for the app, its windows, titles, files people open, repo
 descriptions and README headings. When you create the GitHub repo, name it
 with no "_" or "-": one word or joined words, e.g. GoalTracker.>

<Finish by listing every deliverable: path, what it is, how to check it works.>

<Git rules (no Claude attribution, never commit .claude/) are in
 ~/.claude/CLAUDE.md and apply on their own — nothing to repeat here.>

<Keep the changelog at the bottom current.>

CONTEXT:

figure out how an llm training loop works from its basic building blocks to the entire training pipeline (pretraining, training, posttraining). create visuzalizations … go deep, but make it simple

DELIVERABLES:

mac app that covers the full stack

OPEN QUESTIONS / ASSUMPTIONS:

Asked and answered (2026-10-02):
- Native SwiftUI Mac app (not web views). The website is a later, separate step.
- Building blocks run on a real tiny model that trains live in the app. Pretraining at scale and post-training are interactive simulations on toy data, using the real update maths.
- "pretraining, training, posttraining" reads as: the core training loop, pretraining at scale, then SFT + preference tuning. 13-chapter outline approved.
- Private GitHub repo "LLMUnderstander". The feedback tab opens a pre-filled GitHub issue.

Decided without asking:
- Model: 2 blocks, 2 heads, width 32, context 32, character-level, about 28k weights, trained with AdamW, warm-up + cosine decay and gradient clipping. About 1,500 steps takes roughly a minute on this Mac.
- Training text is generated simple-English sentences, not real web text, so learning is visible in about a minute and nothing copyrighted is embedded.
- Post-training uses a toy policy over 8 whole answers to one prompt. Labeller length bias and an unlabelled "rambling essay" are built in on purpose, so reward hacking at a low KL penalty shows up.
- Scaling-law constants come from the Epoch AI (2024) replication of Chinchilla. Real-model figures are published or widely reported numbers; GPT-2's token count is marked as an estimate.
- The autograd engine is written in Swift, using Accelerate BLAS for matrix multiply. Exclusivity checking is disabled in the build for speed.
- README screenshots are made by the app itself (`--snapshots`).
- No app icon yet (proposal).

Agents & Harnesses section (asked and answered 2026-10-03): a new sidebar section of 4 chapters (14–17), built around a scripted step-through agent run (works offline, no API key).
Decided without asking: the scenario is a coding agent fixing a failing test; tool-call JSON follows the general shape of today's tool-use APIs and is labelled illustrative; the context-window model (200K limit, compaction at 80%) uses illustrative numbers; the agent-RL demo is real REINFORCE on a toy 4-strategy policy, tuned so a naive "tests pass" checker gets reward-hacked; the README screenshots were retaken in dark mode (the system appearance at the time).

CHANGELOG:

- 2026-10-02 — created
- 2026-09-15 — added meta-instruction: built-out applications include a small feedback tab
- 2026-09-15 — added meta-instruction: no "Claude" attribution in commits, PRs, or branches
- 2026-09-16 — added meta-instruction: always include a README when adding to GitHub
- 2026-09-16 — changed meta-instruction: ask clarifying questions in the question widget
- 2026-09-17 — added meta-instructions: Claude never a contributor; never commit .claude/
- 2026-09-26 — compressed the meta-instructions and every field prompt; git rules moved to the global instruction file
- 2026-09-27 — added meta-instruction: applications are built as a Mac app first, then a website
- 2026-09-28 — folded inputs, instructions, constraints, deliverables and done criteria into one free-form CONTEXT
- 2026-09-28 — changed meta-instruction: a README on GitHub always includes a visual
- 2026-09-28 — added meta-instruction: name things like a person would, never snake_case
- 2026-09-28 — changed meta-instruction: README leads with visuals; instructions live in a linked guide
- 2026-09-28 — changed meta-instruction: README is visuals and links; details in docs/INSTRUCTIONS.md and docs/FILE-STRUCTURE.md
- 2026-09-29 — changed meta-instruction: GitHub repo names have no "_" or "-"
- 2026-10-02 — added a DELIVERABLES field after CONTEXT
- 2026-10-02 — built v1: native Mac app with 13 chapters, live tiny GPT, tests, README + docs; private GitHub repo LLMUnderstander
- 2026-10-03 — added Agents & Harnesses section (chapters 14–17: agent loop, harness, tool use, training agents); scripted step-through run, live agent-RL demo
