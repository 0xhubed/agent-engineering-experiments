# Dev tuning log (plan 1c Task 2, preregistration §5.1)

Each tunable arm gets at most 5 configurations on the dev split (214 questions), its default included, and the
one with the highest dev accuracy is chosen. `pageindex` runs its released defaults only; `long_context` and
`oracle` have nothing to tune. Settings are chosen before their results are seen: the settings below were
committed with this log before any of them had finished; any added later is marked as such.

Fairness bound (spec §6.5): every arm answers from at most 9,000 evidence words, so retrieval depth is tuned up to
that budget, not past it. The default vector settings use 1,600 of the 9,000 words (k 8 × 200-word chunks).

Configs: `configs/tree_rag/dev-tuning/<name>.yaml`; the default point is `configs/tree_rag/dev-baseline.yaml`.

| Arm | Config | Change from default | Dev accuracy | A / B / FinanceBench | vs default, 95% CI | Chosen |
|---|---|---|---|---|---|---|
| chunk_embed | default | k 8, 200-word chunks | 83/214 (0.388) | 46/144 · 22/38 · 15/32 | | |
| chunk_embed | k20 | k 20 | 106/214 (0.495) | 55 · 30 · 21 | +0.107 [+0.061, +0.159] | |
| chunk_embed | k40 | k 40 | 119/214 (0.556) | 66 · 32 · 21 | +0.168 [+0.112, +0.224] | |
| chunk_embed | w400-k20 | 400-word chunks, k 20 | 106/214 (0.495) | 58 · 27 · 21 | +0.107 [+0.056, +0.159] | |
| chunk_embed | k45 | k 45 (the full budget) — *added after the first four* | 120/214 (0.561) | 67 · 33 · 20 | +0.173 [+0.117, +0.229] | **yes** |
| hybrid_rerank | default | k 8 of 30 reranked, 50+50 candidates | 108/214 (0.505) | 58/144 · 34/38 · 16/32 | | |
| hybrid_rerank | k20 | k 20 of 60 reranked, 100+100 candidates | 122/214 (0.570) | 67 · 36 · 19 | +0.065 [+0.019, +0.117] | |
| hybrid_rerank | k40 | k 40 of 100 reranked, 150+150 candidates | 141/214 (0.659) | 82 · 36 · 23 | +0.154 [+0.103, +0.210] | |
| hybrid_rerank | w400-k20 | 400-word chunks, k 20 of 60 reranked | 138/214 (0.645) | 79 · 36 · 23 | +0.140 [+0.089, +0.196] | |
| hybrid_rerank | k45 | k 45 of 100 reranked (the full budget) — *added after the first four* | 143/214 (0.668) | 85 · 36 · 22 | +0.164 [+0.107, +0.224] | **yes** |
| raptor | default | k_words 2,000 | 118/214 (0.551) | 73/144 · 24/38 · 21/32 | | |
| raptor | k4000 | k_words 4,000 (same tree) | 135/214 (0.631) | 84 · 30 · 21 | +0.079 [+0.042, +0.117] | |
| raptor | k8000 | k_words 8,000 (same tree) | 154/214 (0.720) | 97 · 33 · 24 | +0.168 [+0.121, +0.224] | |
| raptor | k9000 | k_words 9,000 (the full budget) — *added after the first three* | 157/214 (0.734) | 99 · 33 · 25 | +0.182 [+0.131, +0.238] | **yes** |
| vec_tree | default | doc_k 1 | 114/214 (0.533) | 62/144 · 31/38 · 21/32 | | |
| vec_tree | doc2 | doc_k 2 | 121/214 (0.565) | 64 · 35 · 22 | +0.033 [−0.009, +0.075] | |
| vec_tree | doc3 | doc_k 3 | 123/214 (0.575) | 68 · 35 · 20 | +0.042 [+0.000, +0.084] | |
| vec_tree | doc5 | doc_k 5 — *added after doc2/doc3* | 132/214 (0.617) | 74 · 35 · 23 | +0.084 [+0.037, +0.136] | |
| vec_tree | doc8 | doc_k 8 — *added after doc5* | 137/214 (0.640) | 81 · 34 · 22 | +0.107 [+0.056, +0.164] | **yes** |

CIs: paired bootstrap by question against the arm's default row, 2,000 resamples (seed 17); tuning-time only, not
the reported analysis. A truncated answer (one `chunk_embed` k40 row, `max_tokens`) was rerun, never scored.

## Outcome (2026-10-10)

Each arm takes its highest-dev-accuracy setting (a tie would go to the cheaper one; none occurred). In every arm
that is the setting handing the answerer the most evidence: chunk_embed and hybrid_rerank k 45, raptor 9,000
words (each filling the 9,000-word budget), vec_tree 8 documents. For the two vector arms the last step added
one question (k 40 → 45), so they sit at a plateau, not on a slope. vec_tree used all five settings and was still
rising (doc_k 5 → 8: +5 questions); a larger doc_k was not tried, the budget being spent.

Frozen in `configs/tree_rag/phase1.yaml`. `pageindex` (released defaults), `long_context` and `oracle` are
unchanged. Tuned dev accuracy next to the untuned arms (dev-baseline):

| Arm | All (214) | A (144) | B (38) | FinanceBench (32) |
|---|---|---|---|---|
| long_context | 192/202 (0.950) | 142/142 | 27/30 | 23/30 |
| pageindex | 177/214 (0.827) | 120 | 34 | 23 |
| raptor (tuned) | 157/214 (0.734) | 99 | 33 | 25 |
| hybrid_rerank (tuned) | 143/214 (0.668) | 85 | 36 | 22 |
| vec_tree (tuned) | 137/214 (0.640) | 81 | 34 | 22 |
| chunk_embed (tuned) | 120/214 (0.561) | 67 | 33 | 20 |

Tuning narrowed `pageindex` − `hybrid_rerank` on dev from +32 points to +16 (regime A +43 → +24, B 0 → −5,
FinanceBench +22 → +3). Dev results are for tuning only and are not reported as findings.
