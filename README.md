# agent-engineering-experiments

Reproducible experiments behind [agent-engineering.ch/experiments](https://www.agent-engineering.ch/experiments/).
Each experiment tests one practitioner claim, pre-registers its hypotheses, and publishes per-question results.

| Experiment | Claim under test | Status |
|---|---|---|
| `tree_rag` | Tree-navigation retrieval (PageIndex) vs tuned vector RAG | Phase 0 |

## Layout
- `src/aex/common/` — LLM client, accounting, checkpointing, manifests, scoring (shared)
- `src/aex/experiments/<name>/` — one experiment per directory
- `corpus/` — source-document manifests (URL + sha256). Documents are downloaded, never committed.
- `serving/gb10/` — vLLM launch scripts for the two GB10 boxes

## Running tests
    uv sync && uv run pytest            # fast suite, no GPU, no network
    uv run pytest -m slow               # parse tests that download docling models
