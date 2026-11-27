# LLM Understander

A Mac app that teaches how a large language model is trained, from tokens to RLHF and reasoning, and how it becomes an agent inside a harness. It does this with live visualizations driven by a real tiny transformer that trains on your Mac. Start with the Tour: scroll one sentence through the whole chain.

![The Tour](docs/screenshots/00%20Tour.png)

**Also on the web:** the same Tour and 21 chapters as a website that trains the model in your browser — [open LLM Understander on the web](https://amalmehta.github.io/LLMUnderstander/).

![The website](docs/screenshots/Website.jpg)

| | |
|---|---|
| ![The Training Loop](docs/screenshots/09%20The%20Training%20Loop.png) | ![Attention](docs/screenshots/04%20Attention.png) |
| ![The Transformer Block](docs/screenshots/05%20The%20Transformer%20Block.png) | ![The Optimizer](docs/screenshots/08%20The%20Optimizer.png) |
| ![Learning from Preferences](docs/screenshots/12%20Learning%20from%20Preferences.png) | ![Reasoning Models](docs/screenshots/13%20Reasoning%20Models.png) |
| ![Evals](docs/screenshots/14%20Evals.png) | ![Inference](docs/screenshots/16%20Inference.png) |
| ![The Harness](docs/screenshots/18%20The%20Harness.png) | ![Workflows & Bandits](docs/screenshots/21%20Workflows%20%26%20Bandits.png) |

## The 21 chapters

```mermaid
flowchart LR
    T[Tour] --> A
    A[1 Big Picture] --> B
    subgraph B[Building Blocks]
        B2[2 Tokens] --> B3[3 Embeddings] --> B4[4 Attention] --> B5[5 Transformer Block]
    end
    B --> C
    subgraph C[How It Learns]
        C6[6 Next Token & Loss] --> C7[7 Backprop] --> C8[8 Optimizer] --> C9[9 Training Loop]
    end
    C --> D[10 Pretraining at Scale]
    D --> E
    subgraph E[Post-training]
        E11[11 SFT] --> E12[12 RLHF & DPO] --> E13[13 Reasoning Models]
    end
    E --> F
    subgraph F[Putting It Together]
        F14[14 Evals] --> F15[15 Full Pipeline] --> F16[16 Inference]
    end
    F --> G
    subgraph G[Agents & Harnesses]
        G17[17 Model to Agent] --> G18[18 The Harness] --> G19[19 Tool Use] --> G20[20 Training Agents] --> G21[21 Workflows & Bandits]
    end
```

## Links

- [The website](https://amalmehta.github.io/LLMUnderstander/) · [its source](web/)
- [How to set up, run and use it](docs/INSTRUCTIONS.md)
- [What's where in the code](docs/FILE-STRUCTURE.md)
- [How it's designed](docs/SYSTEM-DESIGN.md)
- [Prompt log](docs/PROMPTS.md)
- [All chapter screenshots](docs/screenshots)
- [Project brief](llm%20understander.md)
