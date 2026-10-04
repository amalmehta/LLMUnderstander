# LLM Understander

A Mac app that teaches how a large language model is trained, from tokens to RLHF, and how it becomes an agent inside a harness. It does this with live visualizations driven by a real tiny transformer that trains on your Mac.

![The Training Loop](docs/screenshots/09%20The%20Training%20Loop.png)

**Also on the web:** the same 18 chapters as a website that trains the model in your browser — [open LLM Understander on the web](https://claude.ai/artifact/Ca1hMZNGCnTQHBocixMSeB) (private for now; share it from the page's Share menu).

![The website](docs/screenshots/Website.jpg)

| | |
|---|---|
| ![The Big Picture](docs/screenshots/01%20The%20Big%20Picture.png) | ![Attention](docs/screenshots/04%20Attention.png) |
| ![The Transformer Block](docs/screenshots/05%20The%20Transformer%20Block.png) | ![The Optimizer](docs/screenshots/08%20The%20Optimizer.png) |
| ![Learning from Preferences](docs/screenshots/12%20Learning%20from%20Preferences.png) | ![The Full Pipeline](docs/screenshots/13%20The%20Full%20Pipeline.png) |
| ![From Model to Agent](docs/screenshots/14%20From%20Model%20to%20Agent.png) | ![The Harness](docs/screenshots/15%20The%20Harness.png) |
| ![Training Models to Be Agents](docs/screenshots/17%20Training%20Models%20to%20Be%20Agents.png) | ![Workflows & Bandits](docs/screenshots/18%20Workflows%20%26%20Bandits.png) |

## The 18 chapters

```mermaid
flowchart LR
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
        E11[11 SFT] --> E12[12 RLHF & DPO]
    end
    E --> F[13 Full Pipeline]
    F --> G
    subgraph G[Agents & Harnesses]
        G14[14 Model to Agent] --> G15[15 The Harness] --> G16[16 Tool Use] --> G17[17 Training Agents] --> G18[18 Workflows & Bandits]
    end
```

## Links

- [The website](https://claude.ai/artifact/Ca1hMZNGCnTQHBocixMSeB) · [its source](web/)
- [How to set up, run and use it](docs/INSTRUCTIONS.md)
- [What's where in the code](docs/FILE-STRUCTURE.md)
- [All chapter screenshots](docs/screenshots)
- [Project brief](llm%20understander.md)
