# File Structure

```
llm understander/
├── README.md                    Front page: screenshots and links
├── llm understander.md          Project brief, assumptions and changelog
├── project.yml                  XcodeGen spec (source of truth for the Xcode project)
├── LLM Understander.xcodeproj   Generated Xcode project
├── LLMUnderstander/             App source
│   ├── App/
│   │   ├── LLMUnderstanderApp.swift   App entry, window, sidebar, menus
│   │   ├── Chapter.swift              The 18 chapters, their sections, and shared app state
│   │   ├── Feedback.swift             Feedback tab and sheet (opens a GitHub issue)
│   │   └── Snapshotter.swift          --snapshots mode that saves README screenshots
│   ├── Engine/                  The learning machinery (no UI)
│   │   ├── Tensor.swift               2-D tensors, reverse-mode autograd, all ops, BLAS matmul
│   │   ├── TinyGPT.swift              The transformer: embeddings, attention, MLP, LayerNorm, tracing
│   │   ├── Optimizers.swift           SGD, Momentum, AdamW, gradient clipping, LR schedule
│   │   ├── Trainer.swift              Background training loop and the observable state the UI shows
│   │   ├── Tokenizer.swift            Character tokenizer and a real BPE trainer
│   │   ├── Corpus.swift               Generated training and validation text
│   │   ├── PreferenceToys.swift       SFT, reward model, RLHF and DPO on a toy 8-answer policy
│   │   ├── AgentToys.swift            Scripted agent run, context-budget model, agent RL (REINFORCE)
│   │   ├── Bandits.swift              Multi-armed bandit strategies, Beta sampling, voting / best-of-n maths
│   │   └── Landscapes.swift           2-D loss surfaces, scaling law, parameter counts, number formatting
│   ├── UI/
│   │   └── Components.swift           Page layout, cards, heatmaps, probability bars, banners
│   └── Chapters/                One file per chapter, in reading order
│       ├── BigPictureChapter.swift
│       ├── TokensChapter.swift
│       ├── EmbeddingsChapter.swift
│       ├── AttentionChapter.swift
│       ├── BlockChapter.swift
│       ├── NextTokenChapter.swift
│       ├── BackpropChapter.swift
│       ├── OptimizerChapter.swift
│       ├── TrainingLoopChapter.swift
│       ├── PretrainingChapter.swift
│       ├── SFTChapter.swift
│       ├── PreferencesChapter.swift
│       ├── PipelineChapter.swift
│       ├── AgentLoopChapter.swift     (also the shared transcript bubble and loop diagram)
│       ├── HarnessChapter.swift
│       ├── ToolUseChapter.swift
│       ├── AgentTrainingChapter.swift
│       └── WorkflowsChapter.swift
│   └── Assets.xcassets/         App icon (generated: causal-attention triangle)
├── Tests/
│   └── EngineTests.swift        Gradient checks and behaviour tests for the engine
├── UITests/
│   └── AppUITests.swift         Launches the app and clicks through every chapter
└── docs/
    ├── INSTRUCTIONS.md          Setup, run, use
    ├── FILE-STRUCTURE.md        This file
    └── screenshots/             One PNG per chapter
```

`build/` (Xcode derived data) is created by builds and is not committed.
