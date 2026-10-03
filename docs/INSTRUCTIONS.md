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

## Use it

- Pick a chapter from the sidebar, or press ⌘1–⌘9. Chapters build on each other, but each one stands alone.
- **Chapter 9, The Training Loop**, trains a real 28k-weight transformer in about a minute. Press **Start training**.
  - Turn on **Slow motion** to follow each phase of a single step.
  - Press **Reset** to try other settings: plain SGD, or a huge learning rate.
- Chapters 3, 4, 5 and 6 look inside that same model. Each shows a banner saying whether it's trained yet. **Train it now** starts training without leaving the page.
- Every chapter ends with a **Go deeper** section for the maths.
- The **Feedback** tab at the bottom of the sidebar (also **Help → Send Feedback…**) opens a pre-filled GitHub issue in your browser. Nothing is sent until you submit it there.

## What is real and what is simplified

| Real computation | Simplified |
|---|---|
| Tokenizers (character-level, and BPE trained on the corpus) | The corpus is generated sentences, not the web |
| A 2-block, 2-head GPT with autograd, AdamW, warm-up + cosine, gradient clipping | Model size: ~28k weights instead of billions |
| Embeddings, attention maps, predictions and losses, all read from that model | Pretraining-at-scale numbers are published or illustrative figures |
| SFT, Bradley–Terry reward model, KL-regularised RL and DPO updates (exact maths) | Post-training uses a toy policy over 8 whole answers to one prompt |
| Optimizer race on 2-D loss surfaces; Chinchilla-form scaling law | |

## Tests

```bash
xcodebuild -project "LLM Understander.xcodeproj" -scheme "LLM Understander" -derivedDataPath build/dd test
```

The 10 engine tests cover:

- Gradient checks of every autograd op and of the full model
- Causal attention
- Training reducing loss
- BPE round-tripping
- SFT, the reward model, reward hacking at a low KL penalty, and DPO
- The scaling-law optimum
- The optimizer race

## Regenerate the screenshots

```bash
"build/dd/Build/Products/Release/LLM Understander.app/Contents/MacOS/LLM Understander" --snapshots docs/screenshots
```

This trains the model, visits every chapter and saves one PNG per chapter.
