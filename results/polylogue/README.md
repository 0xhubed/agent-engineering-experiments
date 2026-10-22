# Multi-agent communication (polylogue), phase 1: results

The main run of the experiment pre-registered in [`preregistrations/07-multi-agent-communication.md`](../../preregistrations/07-multi-agent-communication.md): 2 suites × 30 scored tasks × 5 conditions × 2 seeds = 600 runs on Qwen3.8-27B (`unsloth/Qwen3.8-27B-NVFP4`, revision `7d6f8d4`), from 2026-10-17 to 2026-10-22, on one DGX Spark. Three runs ended in an infrastructure error and were rerun once, as pre-registered; none failed twice.

- `main-runs.csv`: one row per scored cell (600 rows). Infra-error cells show their rerun.
- `main-analysis.json`: the output of the frozen analysis (`polyprep.analysis` in the harness repository, analysis plan of 2026-10-18), including verdicts, intervals and event-log metrics.

Neither file contains question text, answers or transcripts. BrowseComp-Plus keeps its questions obfuscated to keep them out of training data, so `source_id` gives only the BrowseComp-Plus query id. For `math` it is the generator kind and seed.

## Columns of `main-runs.csv`

| column | meaning |
|---|---|
| `suite`, `task_id`, `source_id` | `bcp` (BrowseComp-Plus) or `math` (generated); the task; its upstream id |
| `condition`, `seed` | `single`, `isolated`, `stigmergic`, `polylogue`, `polylogue-bare`; seed 1 or 2 |
| `run_id`, `end_reason` | the run that counts (a rerun ends in `-r1`); `solved`, `wrong_claim`, `budget`, `killed`, `all_idle` |
| `first_claim_solved` | the first claim of the run passed the verifier: the primary score for every condition |
| `any_claim_solved` | any claim passed. This differs from the first claim only for `isolated`, whose agents cannot see each other, so its run goes on after a wrong claim (best-of-4 with the verifier, a reference score) |
| `claims`, `wrong_claims` | claims submitted and failed |
| `total_tokens`, `wall_s` | tokens charged to the team pool (prompt and output of every call), and wall-clock seconds |
| `agents`, `messages`, `substantive_messages`, `blackboard_entries` | team size; messages sent; messages whose content a receiver used (definition in the pre-registration, amended 2026-10-18); blackboard entries |

## Verdict

H2, primary (`bcp`, `polylogue` against `isolated`, first claim): **ONLY PARALLELISM**. The difference is −3.3 points (95% interval −15.0 to +6.7, 60 pairs, McNemar p = 0.79). `isolated` against `single` is +23.3 points (+11.7 to +36.7). The secondary hypotheses (H1, speed; H3, domain; H4, collapse without norms) were not supported. Details are in `main-analysis.json`.
