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
