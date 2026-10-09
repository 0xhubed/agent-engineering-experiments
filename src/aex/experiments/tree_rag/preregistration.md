# Tree RAG, phase 1 — pre-registration

**Status: DRAFT — not frozen.** Freezing = the commit that adds Daniel's predictions (§6) and removes this line.
That commit must predate the first test-split run (`git log` is the evidence; the article footer shows its hash).

## 1. Hypotheses (spec §2.1, verbatim)

| Regime | Corpus | The hard part | Prediction |
|---|---|---|---|
| **A — many near-identical docs** | Structured-product termsheets / final terms / KIDs | Picking the *right document* — same template, same vocabulary, different numbers; embeddings of "barrier 60%" and "barrier 65%" are near-identical | Pure vector RAG confuses products; tree navigation or a doc-selection step wins |
| **B — few very long docs** | Base prospectuses (300–600 pp) + long public product documentation | Finding the *right place* in one document | Tree navigation wins *if* the structure parses cleanly; hybrid+reranker is close |
| **Claim check** | FinanceBench open-150 | Reproducing the vendor setting | PageIndex beats naive vector RAG by a wide margin but a tuned hybrid closes most of the gap |

Corpora as collected (manifests in `corpus/`): regime A = 78 final terms (Julius Bär 31, RBI 28, Deutsche Bank 19,
the last German and issued 2023); regime B = 9 base prospectuses / securities notes / registration documents
(184–1,709 pages); claim check = FinanceBench open-150 over 84 filings, each question asked of its own document only
(`scope: {financebench: question_docs}`), mirroring the vendor setting.

## 2. Arms and models

Arms: `chunk_embed`, `hybrid_rerank` (the reference for every comparison), `raptor`, `pageindex` (SDK
`pageindex==0.2.21` with released defaults, local endpoint), `vec_tree`, `long_context` (N/A above 150k words,
never truncated). `pageindex_native` (PageIndex's own answer, judged) is reported alongside, not as an arm.

| Role | Model | Pin |
|---|---|---|
| Navigator + answerer | Qwen3.8-27B (`unsloth/Qwen3.8-27B-NVFP4`), MTP speculative decoding | `7d6f8d4` |
| Judge | the same model, thinking off | `7d6f8d4` |
| Embeddings | Qwen3-Embedding-4B | `5cf2132` |
| Reranker | Qwen3-Reranker-4B (prompt template applied client-side) | `22e6836` |

Serving image `vllm/vllm-openai@sha256:541e0e475418de6178b45c0d9ef420fb6be79bf43130a4d552cb668e425f4d27` on
`hubed-dgx`. Thinking on with `reasoning_effort: medium`, temperature 0, seed 17, answer budget 8,192 tokens (a
truncated answer is an error row, rerun, never scored). **Index-time calls** — RAPTOR's cluster summaries and
PageIndex's tree building — use the same weights with **thinking off** (`indexer` in the run config), identically
for both tree arms; query-time navigation keeps thinking on. Measured 2026-10-07: with thinking on, a single
RAPTOR summary ran past 4,096 tokens and PageIndex indexing calls past 10 minutes; with thinking off a summary
takes 10–18 s. Second family and frontier reference: added only if
decisions D6/D7 are settled before the run; otherwise reported as not run.

## 3. Primary metric and analysis

- **Primary:** accuracy per arm; each arm's difference vs `hybrid_rerank`, paired by question, with a 95% CI from a
  10,000-resample paired bootstrap. A difference is a "win" only if its CI excludes 0; otherwise "no detectable
  difference at n = …".
- **Scoring:** exact match for identifiers and dates; numbers within ±0.5% relative; LLM judge only for free-text
  golds. **Open until frozen:** whether a gold stated to fewer decimals (FinanceBench `0.01`, `0.8`) also accepts
  any answer within half a unit of its last digit (spec §6.3 "or the document's stated precision") — decided on the
  dev split, before freezing.
- **Judge validity:** Cohen's κ between the judge and Daniel on ≥100 dev answers; κ < 0.7 → judge prompt revised on
  dev and re-measured. κ is reported.
- **Secondary (spec §6.3):** evidence recall/precision against gold pages; wrong-document rate (regime A); latency
  p50/p95; sequential LLM calls per query; tokens (index amortised + query); GPU-seconds; failure class.
- **Breakdowns:** regime; question type; pre/post 2026-08-05 issue date (the answerer's release); regime-A
  question form (by ISIN vs by description).

## 4. Data and splits

- Gold: `gold/termsheets.jsonl`, `gold/prospectuses.jsonl` and FinanceBench regenerated from
  `patronus-ai/financebench@cc39aeb`. **Deviation from spec §5.5:** the termsheet and prospectus gold is
  model-verified, not human-verified. Claude (Opus 5.5) reviewed every LLM draft against the source pages
  (`verified_by: claude-opus-5.5`, one-line reason per decision in `gold/review/decisions.jsonl`), and no human
  audit was run (Daniel's decision, 2026-10-07). The article states this next to every result built on that gold. Regime-A questions exist as ISIN/description pairs sharing
  a `pair_id`.
- Split: dev 20% / test 80%, by `sha256("17:" + (pair_id or qid))`. Both forms of a pair share a split.
- Quotas before freezing (`python -m aex.gold.report`): ≥150 test questions per regime (a pair counts once),
  ≥10% unanswerable per regime, every question type ≥40.

## 5. Procedure and exclusions

1. Dev tuning: each arm ≤5 configurations, chosen by dev accuracy, log committed (`runs/dev-tuning.md`). PageIndex
   runs its released defaults only. A PageIndex tree that fails to build is retried up to 3 times under the same
   settings, never with other settings; if all fail, questions that need only that document are scored incorrect for
   `pageindex`, `pageindex_native` and `vec_tree` (failure class `index_failed`), never excluded, and the count is
   reported. Dev: all 113 trees built; `fb-johnson-johnson-2022-10k` failed in two runs (an SDK error while
   finalising its tree) and built in the third, so the failure is not deterministic.
2. Determinism check (plan 1c Task 3) before the test run; its outcome fixes the reproducibility criterion here,
   as an addendum committed before the test run.
3. One full test-split run. `error` rows are rerun until none remain; `not_applicable` rows are reported, never
   scored. No test question is inspected individually before the run completes.
4. Reproduction of 20 random test questions × all arms from a clean checkout, by the criterion from step 2.

## 6. Predicted ordering (Daniel — written before any test result)

> **TO BE WRITTEN BY DANIEL (decision D5).** For each of regime A, regime B and the claim check: the expected order
> of the six arms by accuracy, and optionally the expected size of the gap between `pageindex` and
> `hybrid_rerank`. The article shows these next to the observed results.

- Regime A:
- Regime B:
- Claim check (FinanceBench):
