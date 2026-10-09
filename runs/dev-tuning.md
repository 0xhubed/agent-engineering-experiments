# Dev tuning log (plan 1c Task 2, preregistration §5.1)

Each tunable arm gets at most 5 configurations on the dev split (214 questions), its default included, and the
one with the highest dev accuracy is chosen. `pageindex` runs its released defaults only; `long_context` and
`oracle` have nothing to tune. Settings are chosen before their results are seen: the settings below were
committed with this log before any of them had finished; any added later is marked as such.

Fairness bound (spec §6.5): every arm answers from at most 9,000 evidence words, so retrieval depth is tuned up to
that budget, not past it. The default vector settings use 1,600 of the 9,000 words (k 8 × 200-word chunks).

Configs: `configs/tree_rag/dev-tuning/<name>.yaml`; the default point is `configs/tree_rag/dev-baseline.yaml`.

| Arm | Config | Change from default | Dev accuracy | Chosen |
|---|---|---|---|---|
| chunk_embed | default | k 8, 200-word chunks | 83/214 (0.388) | |
| chunk_embed | k20 | k 20 | running | |
| chunk_embed | k40 | k 40 | running | |
| chunk_embed | w400-k20 | 400-word chunks, k 20 | running | |
| hybrid_rerank | default | k 8 of 30 reranked, 50+50 candidates | 108/214 (0.505) | |
| hybrid_rerank | k20 | k 20 of 60 reranked, 100+100 candidates | running | |
| hybrid_rerank | k40 | k 40 of 100 reranked, 150+150 candidates | running | |
| hybrid_rerank | w400-k20 | 400-word chunks, k 20 of 60 reranked | queued | |
| raptor | default | k_words 2,000 | 118/214 (0.551) | |
| raptor | k4000 | k_words 4,000 (same tree) | running | |
| raptor | k8000 | k_words 8,000 (same tree) | running | |
| vec_tree | default | doc_k 1 | 114/214 (0.533) | |
| vec_tree | doc2 | doc_k 2 | running | |
| vec_tree | doc3 | doc_k 3 | running | |
