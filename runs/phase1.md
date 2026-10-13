# Phase 1 test run — log

Pre-registration frozen at e8e1778 (2026-10-10 15:57). Config `configs/tree_rag/phase1.yaml`, checkpoint
`runs/phase1.sqlite` (not committed). Everything below is operational; no arm setting changed.

## 2026-10-10 15:59 — start

All arms sharded at once (oracle 2, long_context 3, others 3–4 shards; each indexed arm's shard 0 builds its
index for the test documents first). 58 FinanceBench filings reached only by test questions had no PageIndex
tree or RAPTOR tree yet.

## 2026-10-10 17:00 — restart in stages

The server was saturated (8 running, 13 waiting): 18 rows in the first hour, and index-time calls queued behind
answering shards until they timed out. RAPTOR's index build stopped after one summary call timed out 3 times
(`LLMError … transport error: timed out`); PageIndex built no new tree in that hour. All answering shards were
stopped (finished rows kept, in-flight rows redone) and the index builds were restarted alone.

From 18:42 one answering shard each ran for oracle, long_context, chunk_embed and hybrid_rerank. They slowed the
sequential index builds 3–5× (RAPTOR from ~9 to ~50 min per filing), so from 22:09 only the light oracle shard
ran beside them. The other shards start once pageindex, raptor and vec_tree have recorded their indexes.
Order of work does not change any row's settings.

## 2026-10-11 01:30 — deviation: PDF decryption dependency

`fb-adobe-2022-10k` failed its PageIndex tree build 3 times: "could not read PDF: PyCryptodome is required for
AES algorithm". The PDF is AES-encrypted with an empty user password; the SDK's PDF reader (PyPDF2) needs the
optional `pycryptodome` package for AES, which the harness had not installed. 38 other FinanceBench PDFs are
encrypted with RC4, which needs no extra package, and built normally. This is a missing dependency in our
environment, not a failure of the method, so the pre-registered `index_failed` rule (§5.1) is not applied:
`pycryptodome` was added and the tree rebuilt under the same settings. The PageIndex process was stopped
before it had written any row or its index cost, so no row was scored under the failure and the index cost is
recorded once, with this tree included. The article lists this as a deviation.

## 2026-10-11 09:50 — fewer long_context shards

With all 20 shards running the server was saturated (~2.2k prompt tok/s, 15 running, 5–12 waiting) and calls
from the PageIndex SDK (`pageindex`, `vec_tree`) waited past the SDK's own 600 s request timeout: 20 `error` rows
by 09:44, 11 of them `vec_tree` (of its 17 rows). The timeout is part of the released defaults and is not
changed. Instead `long_context` (≈200k prompt tokens per test row, the longest prompts in the queue) runs on one
shard instead of three until the other arms finish; its other two shards resume then. `error` rows are rerun
at the end of the run, as pre-registered (§5.3).

## 2026-10-12 02:57 — long_context shard 1/3 resumed

`long_context` shard 0/3 finished its 301 questions. Shard 1/3 was resumed in its place, so the number of
`long_context` processes stays at one; shard 2/3 resumes when the cheaper arms finish. The process manifest
now records 0e74a70, which differs from 174c74e only by this log.

Each process prints a per-arm accuracy summary of the whole checkpoint when it exits. Checking that the shard
had exited normally showed that summary for the partial test run (aggregates only, no question-level rows).
Nothing was changed because of it: all settings were frozen at e8e1778 and the shard schedule follows the plan
above. Further checks of finished shards read row counts from the checkpoint, not the log tail.

## 2026-10-12 05:35 — oracle done; vec_tree on four shards

`oracle` finished (847 rows). Its slot went to `vec_tree`, the slowest arm (274 of 847 rows): its three shards
were stopped and restarted as four (`--shard i/4`). Finished rows are skipped; the questions in flight were
redone. Its recorded index is reused, not rebuilt.

## 2026-10-12 11:12 — long_context on two shards, vec_tree on five

`hybrid_rerank` shard 0/3 and `raptor` shard 2/3 finished, freeing two slots. `long_context` (shard 1/3
running, 2/3 still paused; 368 rows left) was stopped and restarted as `--shard i/2`, and `vec_tree` (486 left)
as `--shard i/5`. Finished rows are skipped; questions in flight were redone. Earlier shard logs moved to
`runs/phase1/old/`.

## 2026-10-12 13:35 — vec_tree on seven shards, pageindex on five

`chunk_embed` finished (847 rows); `hybrid_rerank` and `raptor` have one or two shards left with a few rows
each. `pageindex` shard 0/4 finished. The server had room (13 running, 0 waiting, KV cache 38%), so
`vec_tree` (452 rows left) was restarted as `--shard i/7` and `pageindex` (208 left) as `--shard i/5`, to
bring the remaining arms to similar finishing times. `long_context` stays on two shards, as its ~200k-token
prompts caused the earlier timeouts. The restarted processes also retry their arm's `error` rows.

## 2026-10-12 20:20 — vec_tree back to five shards

With only `long_context`, `pageindex` and `vec_tree` left, the server stayed saturated (5–12 requests waiting,
~2.6k prompt tok/s, ~22 generated tok/s in total). Calls from the PageIndex SDK waited past its 600 s timeout:
7 of the 34 `vec_tree` rows in the last 100 minutes were `error` rows. `vec_tree` was restarted as
`--shard i/5` to shorten the queue. All 21 `error` rows so far are request timeouts; they are rerun as
pre-registered (§5.3).

## 2026-10-13 06:53 — long_context done; vec_tree on nine shards

`long_context` finished (847 rows, one `error` row), and `pageindex` shards 1/5 and 2/5 finished. With three
`pageindex` shards left, `vec_tree` (237 rows left, 18 `error` rows) was restarted as `--shard i/9`, keeping
twelve processes in total as before.

## 2026-10-13 08:30 — pageindex done; error-row reruns start; one long_context row cannot be rerun

`pageindex` finished (847 rows, 13 `error` rows, all SDK timeouts). With the queue short (0–4 waiting), the
`error` rows of finished arms are rerun alongside `vec_tree` (nine shards): `pageindex` on three shards,
`hybrid_rerank` (1 row) and `long_context` (1 row) on one each.

The `long_context` row is not a timeout. The vLLM server rejects the request: "maximum context length is 262144
tokens … you requested 8192 output tokens and your prompt contains at least 253953 input tokens". The document is
under the pre-registered 150k-word N/A threshold (§2) but tokenizes above the window, so the word count
underestimated its tokens. A rerun fails the same way. §2's intent is "N/A when it does not fit — never truncated"
and §5.3 says `error` rows are rerun until none remain; this row can satisfy neither as written. The row is left as
an `error` row in the checkpoint until a ruling (recorded as a deviation in the article either way). Not truncated.

## 2026-10-13 11:31 — run complete

`vec_tree` finished at 11:10; the last `error` rows (3 `pageindex`, 1 `vec_tree`, all timeouts) were rerun on an
otherwise idle server and succeeded. The checkpoint holds 6,776 rows (7 arms × 847 test questions, plus
`pageindex_native`). One `error` row remains: the `long_context` context overflow above, pending a ruling. Every
timeout `error` row was resolved by a rerun.

## 2026-10-13 13:29 — reproduction (Task 5): verdicts agree on 96.4% of 168 rows

A clean clone at 22cb395 (same code as the run; `git_dirty` false in its manifest) ran the 21 test questions of
`--shard 0/48` × all arms with `configs/tree_rag/reproduce.yaml` into a fresh checkpoint: 168 rows, no `error`
rows. Index artefacts under `data/` were shared with the full run, so this reproduces answering, navigation and
judging, not the index builds. `tree_rag.compare` against `runs/phase1.sqlite` (`runs/reproduce-compare.json`):

| arm | n | raw | correct | evidence (ordered) |
|---|---|---|---|---|
| all | 168 | 4.8% | **96.4%** | 56.5% |
| chunk_embed | 21 | 0.0% | 95.2% | 0.0% |
| hybrid_rerank | 21 | 0.0% | 100.0% | 95.2% |
| long_context | 21 | 9.5% | 100.0% | 100.0% |
| oracle | 21 | 19.1% | 100.0% | 100.0% |
| pageindex | 21 | 0.0% | 85.7% | 52.4% |
| pageindex_native | 21 | 4.8% | 95.2% | 52.4% |
| raptor | 21 | 0.0% | 95.2% | 0.0% |
| vec_tree | 21 | 4.8% | 100.0% | 52.4% |

Verdict agreement is 96.4% (6 of 168 rows flip; 3 in `pageindex`), above the ≥80% criterion fixed in the
determinism addendum (which measured 91.9% on dev). **The run reproduces.**

`compare`'s evidence column compares the ordered page list. For `chunk_embed` and `raptor` it reads 0% because
the prospectuses share near-identical boilerplate: chunks from different documents tie on score and come back in a
different order. Computed ad hoc (not a pre-registered criterion): the evidence page *set* agrees on 71.4% of rows
overall (chunk_embed 76.2%, raptor 38.1%), and the scored pair (evidence recall, evidence precision) agrees on
85.1% (chunk_embed 90.5%, raptor 81.0%, pageindex 66.7%, vec_tree 76.2%). Evidence scores are therefore noisier
than verdicts, most of all for the navigating arms; the article reports that alongside the evidence metrics.
