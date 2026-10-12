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
