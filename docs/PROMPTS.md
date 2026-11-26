# Prompt Log

Each session's prompts, word for word and in order, with a line on what each led to. Sessions before 2026-11-26 weren't logged here; their decisions are in the [project brief](../llm%20understander.md).

## 2026-11-26 — Release script, then a review round of 20 changes

1. "Review the uncommitted changes, commit them in sensible commits with clear messages, and push. Don't commit secrets or build output."
   Committed the release script and docs, then the version bump to 1.1, on a branch and pushed it.
2. (Create PR)
   Opened pull request #1.
3. "merge it"
   Rebase-merged pull request #1 into main.
4. `git restore project.yml docs/INSTRUCTIONS.md "LLM Understander.xcodeproj/project.pbxproj" && rm scripts/release-mac-app.sh && git pull` (run as a shell command)
   It ran in the worktree instead of the main checkout; the deleted script was restored and the main checkout was cleaned and pulled.
5. "0, 5, 15"
   The N,N,N shortcut: listed 5 things to add and 15 to improve, ranked by impact.
6. "Just improvements 1–5", then "do all"
   Building all 20. Answers to the follow-up questions: three new chapters (Reasoning Models, Evals, Inference), Mac first then the website, one guess per chapter, a pull request per batch then release 1.1.
