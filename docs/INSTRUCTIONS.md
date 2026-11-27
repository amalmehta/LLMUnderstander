# Instructions

## Requirements

- macOS 14 (Sonoma) or later
- Xcode 15 or later (built and tested with Xcode 26.5)
- Optional: [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`), only if you change `project.yml`

## Run it

Open `LLM Understander.xcodeproj` in Xcode and press **Run** (⌘R).

Or build from the terminal:

```bash
xcodebuild -project "LLM Understander.xcodeproj" -scheme "LLM Understander" -configuration Release -derivedDataPath build/dd build
```

```bash
open "build/dd/Build/Products/Release/LLM Understander.app"
```

To install it, drag `LLM Understander.app` from `build/dd/Build/Products/Release/` into `/Applications`.

## Download it

Get `LLM-Understander.zip` from the [latest release](https://github.com/amalmehta/LLMUnderstander/releases/latest), unzip it and move **LLM Understander** to Applications. It's signed with a Developer ID and notarized by Apple, so it opens like any other app.

## Make a release

Bump `MARKETING_VERSION` in `project.yml` (and run `xcodegen`) and `version` in `web/package.json` to match; the script stops if they differ. Then:

```bash
scripts/release-mac-app.sh --publish
```

It builds a universal app, signs it with the Developer ID Application certificate and the hardened runtime, notarizes and staples it, and attaches `LLM-Understander.zip` to a new GitHub Release named after the version. The top of the script lists the one-time setup: the certificate and a `notarytool` keychain profile.

## The website

The same Tour and 21 chapters also run as a website, in `web/` (Preact + TypeScript, built with Vite into one self-contained HTML file). Everything runs in the browser, including training the tiny model, which runs in a Web Worker so the page stays smooth.

Run it locally (needs Node 20+):

```bash
npm --prefix web install
```

```bash
npm --prefix web run dev
```

Then open http://localhost:5173. To build the single-file site (`web/dist/index.html`) and the Artifact version (`web/artifact/LLM Understander.html`):

```bash
npm --prefix web run build
```

Website tests (the engine tests, ported):

```bash
npm --prefix web test
```

Notes:
- The trained model and reading progress are kept in the browser's local storage. **Settings** (bottom of the chapter list) resets either. Add `?fresh` to the URL to start untrained, with no progress, without touching what's saved.
- Training speed depends on the browser and how busy the computer is; on an idle Mac it's about the same as the Mac app. Both use the same random number generator, so with the same settings they show the same run.
- Each chapter has its own link, e.g. `#attention`. Press `[` and `]` for the previous and next chapter.

## Use it

- **The Tour** (top of the sidebar, and where the app opens the first time) follows one sentence through the whole chain as you scroll: tokens, embeddings, attention, the blocks, prediction, loss, backprop, the update, then pretraining, SFT, preferences, reasoning and agents. Each step links to its chapter.
- Pick a chapter from the sidebar, press ⌘0–⌘9, or ⌘[ and ⌘] for the previous and next chapter. Chapters build on each other, but each one stands alone. A tick marks the chapters you've read (20 seconds on the page, or reaching its end), and the app reopens where you left off.
- Each chapter opens with **Guess first**: pick what you expect before the demo, and the answer and why appear.
- **Chapter 9, The Training Loop**, trains a real 28k-weight transformer in about a minute. Press **Start training**.
  - Turn on **Slow motion** to follow each phase of a single step, and type your own prompt for the samples.
  - Press **Reset** to try other settings: plain SGD, or a huge learning rate.
  - The trained model, its optimizer state and charts are saved when training stops and reloaded next launch (`~/Library/Application Support/LLM Understander/Tiny Model.json`), so resuming continues the same run. **Reset** deletes it.
- Chapters 3–7, 10, 11, 14 and 16 look inside that same model. Each shows a banner with its validation loss; it counts as trained once that's at or below 60% of random guessing. **Train it now** starts training without leaving the page.
- **Chapter 11** fine-tunes your own model to answer questions; **13, Reasoning Models**, trains a toy reasoner with GRPO; **14, Evals**, scores your model on a 40-question benchmark as it trains; **16, Inference**, samples from it with temperature and top-p and counts what a KV cache saves.
- Chapters 17–21 (**Agents & Harnesses**) step through a coding agent fixing a failing test. They show the harness, tool calls, the context window filling up, and RL training an agent (including how a naive checker gets reward-hacked). Chapter 21 covers workflow patterns, voting and a live multi-armed-bandit race.
- Every chapter has a **Go deeper** section for the maths.
- **Settings** (⌘,) shows what's stored and where, and resets the model or your reading progress.
- The **Feedback** tab at the bottom of the sidebar (also **Help → Send Feedback…**) opens a pre-filled GitHub issue in your browser. Nothing is sent until you submit it there.
- VoiceOver reads the charts, heatmaps and landscapes by name and value.

## What is real and what is simplified

| Real computation | Simplified |
|---|---|
| Tokenizers (character-level, and BPE trained on the corpus) | The corpus is generated sentences, not the web |
| A 2-block, 2-head GPT with autograd, AdamW, warm-up + cosine, gradient clipping | Model size: ~28k weights instead of billions |
| Embeddings, attention maps, predictions and losses, all read from that model | Pretraining-at-scale numbers are published or illustrative figures |
| SFT, Bradley–Terry reward model, KL-regularised RL and DPO updates (exact maths) | Post-training uses a toy policy over 8 whole answers to one prompt |
| Optimizer race on 2-D loss surfaces; Chinchilla-form scaling law | |
| Agent RL: REINFORCE with sampled episodes on 4 strategies; context-window and compaction model; bandit strategies (greedy, ε-greedy, UCB1, Thompson sampling); exact voting maths | The agent run (Chapters 17–19) is scripted, not a live model; token counts are ≈ characters ÷ 4 |
| SFT on your own model (Chapter 11), the 40-item benchmark (Chapter 14) and sampling (Chapter 16) all run the real model | Reasoning (Chapter 13) uses a toy policy that picks a thinking length; the update is real GRPO |
| KV-cache work is counted exactly from the model's shape | Speculative decoding numbers come from the published formula, not a draft model |

## Tests

```bash
xcodebuild -project "LLM Understander.xcodeproj" -scheme "LLM Understander" -derivedDataPath build/dd test -test-timeouts-enabled YES -default-test-execution-time-allowance 240
```

Every push and pull request runs the engine tests and the website tests on GitHub Actions (`.github/workflows/ci.yml`), and the website deploys only if its tests pass. The UI tests run only locally, since they need a logged-in screen.

There are two test suites: 23 engine tests (about 2½ minutes) and 12 UI tests that launch the app and click through it (about 5 minutes; UI tests need the Mac unlocked, and the app will take the screen while they run). To run only the engine tests, add `-only-testing:"Engine Tests"`.

The engine tests cover:

- Gradient checks of every autograd op and of the full model
- Causal attention
- Training reducing loss
- BPE round-tripping
- SFT, the reward model, reward hacking at a low KL penalty, and DPO
- The scaling-law optimum
- The optimizer race
- Agents: RL hacking a naive reward but not a robust one, the scripted run and loss mask, compaction keeping the context under the limit
- Bandits (Thompson sampling beats greedy and ε-greedy on average), voting maths, Beta sampling
- Saving and reloading the trained model, and a resumed run matching an unbroken one
- "Trained" being reached by a full run; SFT on the real model teaching the answer format; GRPO raising accuracy and thinking length; the benchmark curves; KV-cache and speculative-decoding maths
- The shared fixture: the same corpus and first ten losses as the website's engine

The UI tests open the Tour and all 21 chapters, make a guess, open Settings, and use the main controls in each: training, back-propagation steps, the gradient check, SFT, RLHF, DPO, the agent step-throughs, the agent-RL and bandit runs, and the feedback sheet.

## Regenerate the screenshots

```bash
scripts/take-screenshots.sh
```

This builds the app, trains the model, visits every chapter and saves one PNG per chapter to `docs/screenshots`. The terminal needs Screen Recording permission. `scripts/release-mac-app.sh --publish` refuses to run while the screenshots are older than the latest change to the app's code.
