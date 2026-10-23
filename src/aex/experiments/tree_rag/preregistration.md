# Tree RAG, phase 1 — pre-registration

**Status: frozen 2026-10-10**, after dev tuning (§5.1) and the determinism check (§5.2), before the first
test-split run (`git log` is the evidence; the article footer shows this commit's hash).

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
  golds. Answers are read in English and German forms (the Deutsche Bank final terms are German): dates such as
  "11. Februar 2028" or "01 Apr 2027"; decimal commas ("EUR 40,00"); a lone separator before exactly three digits
  ("500.000") accepts both readings, except in a percentage; an exact gold may be followed by a comma clause or a
  bracketed qualifier ("Deutsche Bank AG, Taunusanlage 12, …"). Found on dev (2026-10-09), where the old scorer
  marked 77 correct answers wrong across all arms; stored answers were rescored (`tree_rag.rescore`), not rerun. Numbers also match when within half a unit of the gold's last stated digit (spec §6.3 "or the document's
  stated precision"): a gold rounded to fewer decimals, such as FinanceBench's "0.8" or "0.01", accepts the
  unrounded value ("0.83", "0.0137"); a gold with two decimals ("$4.60") gains nothing from it. Decided before
  freezing on principle, not on results: on dev it changes no row (2026-10-09, `rescore --dry-run`).
- **Judge validity:** Cohen's κ between the judge and an independent grader on ≥100 dev answers; κ < 0.7 → judge
  prompt revised on dev and re-measured. κ is reported. **Deviation from spec §6.3:** the grader is Claude (Opus 5.5),
  not Daniel (Daniel's decision, 2026-10-09), grading blind to the judge's verdict with a one-line reason per item
  (`gold/review/judge_grades.claude-opus-5.5.jsonl`). This measures agreement between two models, not with a
  human; the article says so next to κ. **Result (dev, 2026-10-09):** κ = 0.895 on 120 answers (15 per arm incl.
  `pageindex_native`), agreement 117/120; in all 3 disagreements the judge was stricter than the grader. The judge
  prompt stays as it is (`gold/judge_kappa.json`).
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
   runs its released defaults only. Chosen (2026-10-10), frozen in `configs/tree_rag/phase1.yaml`: `chunk_embed`
   and `hybrid_rerank` k 45 (the 9,000-word evidence budget), `raptor` 9,000 words, `vec_tree` 8 documents. A PageIndex tree that fails to build is retried up to 3 times under the same
   settings, never with other settings; if all fail, questions that need only that document are scored incorrect for
   `pageindex`, `pageindex_native` and `vec_tree` (failure class `index_failed`), never excluded, and the count is
   reported. Dev: all 113 trees built; `fb-johnson-johnson-2022-10k` failed in two runs (an SDK error while
   finalising its tree) and built in the third, so the failure is not deterministic. Likewise, when the SDK's
   agent runs out of its default turn budget (`max_turns`) it is retried up to 3 times under the same settings,
   never with a larger budget; if all 3 run out, the question is scored incorrect (failure class `max_turns`) for
   `pageindex`, `pageindex_native` and `vec_tree`. Retries' tokens and latency count toward the question's cost.
2. Determinism check (plan 1c Task 3) before the test run; its outcome fixes the reproducibility criterion here,
   as an addendum committed before the test run.

   **Addendum (2026-10-10, before the test run).** Two runs of the frozen settings, started at the same time,
   20 dev questions (`--shard 0/11`) × all 8 arms (`configs/tree_rag/determinism-{a,b}.yaml`,
   `runs/determinism-compare.json`). Of 160 row pairs, 6.9% have identical answer text, 91.9% the same verdict
   (`correct`), 71.9% the same evidence pages, 66.2% both. The server is not deterministic at temperature 0
   under concurrent load: `long_context` and `oracle`, whose evidence is fixed, still change verdict on 1–2 of
   20 rows. Query embeddings vary in the 4th decimal, which reorders near-tied passages (evidence agreement:
   `raptor` 45%, `pageindex`/`vec_tree` 50%, `chunk_embed` 100%). Verdict flips are unbiased (7 wrong→right,
   6 right→wrong) and cluster on judge-graded questions: 9 of 49 LLM-judged rows flipped vs 4 of 111 rule-scored
   rows; 6 of the 13 flips are one FinanceBench free-text question on which the judge's verdict varied.
   Decisions:
   - **Concurrency stays as tuned.** Concurrency 1 would take about 8 days for the ~6,800 test rows (dev rows
     average ~100 s each) and was not shown to remove the noise (embedding drift and the judge are separate
     calls).
   - **Reproducibility criterion (step 4):** the rerun matches the full run on `correct` for **≥ 80%** of shared
     rows. 80% is the 1st percentile of agreement for 20 questions resampled by question (10,000 resamples,
     seed 17) from this check, so an intact pipeline fails by chance about 1 time in 100. Identical text and
     evidence pages are reported, not required.
   - **Run noise in the results.** The paired bootstrap treats each verdict as fixed, so it leaves out run-to-run
     noise. From the flip rates above (3.6% rule-scored, 18% judge-graded, 8.1% overall), the 95% half-width of
     that noise on a paired difference is about 2 points overall, 2 in regime A, 3 in regime B and 8 on
     FinanceBench. A "win" whose CI excludes 0 by less than that half-width is reported as **fragile**; the
     primary rule (§3) is unchanged. The article reports the flip rates.
3. One full test-split run. `error` rows are rerun until none remain; `not_applicable` rows are reported, never
   scored. No test question is inspected individually before the run completes.
4. Reproduction of 20 random test questions × all arms from a clean checkout, by the criterion from step 2.

## 6. Predicted ordering

**Deviation from spec (decision D5):** the predictions were to be Daniel's, written before any result. Daniel
delegated them to Claude (2026-10-09), so they are **Claude's (Opus 5.5), written after the dev run** and informed
by it. They are a forecast from dev, not a blind prior; the article labels them so and does not present them as
the author's expectations. Dev accuracy (n per arm in brackets) is given beside each, as the basis. Dev is small
outside regime A; a test result that departs from it is plausible, not a surprise.

- **Regime A:** `long_context` > `pageindex` > `raptor` > `hybrid_rerank` ≈ `vec_tree` > `chunk_embed`.
  `pageindex` − `hybrid_rerank` ≈ +25 points, CI excluding 0. Tuned dev (144): long_context 1.00 (142
  applicable), pageindex 0.83, raptor 0.69, hybrid_rerank 0.59, vec_tree 0.56, chunk_embed 0.47.
- **Regime B:** all six within a few points: `hybrid_rerank` ≥ `long_context` ≈ `pageindex` ≈ `vec_tree` ≥
  `raptor` ≈ `chunk_embed`. `pageindex` − `hybrid_rerank` ≈ 0 to −5 points: no detectable difference. Tuned dev
  (38): hybrid_rerank 0.95, long_context 0.90 (30 applicable), pageindex 0.89, vec_tree 0.89, raptor 0.87,
  chunk_embed 0.87.
- **Claim check (FinanceBench):** `raptor` ≈ `long_context` ≥ `pageindex` ≈ `hybrid_rerank` ≈ `vec_tree` >
  `chunk_embed`. `pageindex` − `hybrid_rerank` ≈ 0 to +5 points: no detectable difference, i.e. the tuned
  hybrid closes most of the gap, as §1 predicts. Tuned dev (32): raptor 0.78, long_context 0.77 (30 applicable),
  pageindex 0.72, hybrid_rerank 0.69, vec_tree 0.69, chunk_embed 0.63.

Revised after dev tuning (2026-10-10, `runs/dev-tuning.md`), before freezing: the first version (commit
225cdae) was written from the untuned arms and predicted a +40-point regime-A gap and a +15 to +25-point
FinanceBench gap; tuning moved `raptor` above `hybrid_rerank` and narrowed both gaps. Both versions are in
`git log`; the article shows the final one.
