# Multi-agent communication (polylogue): results

- [Phase 1](#phase-1-talking-vs-working-alone): talking vs working alone (pre-registration 07).
- [Phase 2](#phase-2-forking): agents that fork from a shared start (pre-registration 08).

## Phase 1: talking vs working alone

The main run of the experiment pre-registered in [`preregistrations/07-multi-agent-communication.md`](../../preregistrations/07-multi-agent-communication.md): 2 suites × 30 scored tasks × 5 conditions × 2 seeds = 600 runs on Qwen3.8-27B (`unsloth/Qwen3.8-27B-NVFP4`, revision `7d6f8d4`), from 2026-10-17 to 2026-10-22, on one DGX Spark. Three runs ended in an infrastructure error and were rerun once, as pre-registered; none failed twice.

- `main-runs.csv`: one row per scored cell (600 rows). Infra-error cells show their rerun.
- `main-analysis.json`: the output of the frozen analysis (`polyprep.analysis` in the harness, [0xhubed/polylogue](https://github.com/0xhubed/polylogue), analysis plan of 2026-10-18), including verdicts, intervals and event-log metrics.

Neither file contains question text, answers or transcripts. BrowseComp-Plus keeps its questions obfuscated to keep them out of training data, so `source_id` gives only the BrowseComp-Plus query id. For `math` it is the generator kind and seed.

### Columns of `main-runs.csv`

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

### Verdict

H2, primary (`bcp`, `polylogue` against `isolated`, first claim): **ONLY PARALLELISM**. The difference is −3.3 points (95% interval −15.0 to +6.7, 60 pairs, McNemar p = 0.79). `isolated` against `single` is +23.3 points (+11.7 to +36.7). The secondary hypotheses (H1, speed; H3, domain; H4, collapse without norms) were not supported. Details are in `main-analysis.json`.

## Phase 2: forking

The main run of the experiment pre-registered in [`preregistrations/08-polylogue-forking.md`](../../preregistrations/08-polylogue-forking.md): the same 60 scored tasks, 3 conditions (`fork`, `fork-silent`, `isolated`) × 2 seeds = 360 runs on the same model and revision, from 2026-10-25 to 2026-10-28, on one DGX Spark. No run ended in an infrastructure error. In `fork` and `fork-silent`, one root agent works alone, is told to fork into three copies of itself once 10% of the team pool is charged, and writes a direction for each copy; `fork` keeps messaging after the fork, `fork-silent` has none.

- `phase2-runs.csv`: one row per scored cell (360 rows).
- `phase2-analysis.json`: the output of the frozen analysis (`polyprep.analysis`, phase-2 path, harness commit `4423cd9`), including verdicts, intervals, the mechanism check and fork mechanics.

Neither file contains question text, answers, transcripts or the directions the agents wrote (directions can repeat clues from a question).

### Columns of `phase2-runs.csv`

The columns of `main-runs.csv`, without `blackboard_entries` (no phase-2 condition has a blackboard), and:

| column | meaning |
|---|---|
| `condition` | `fork`, `fork-silent` or `isolated` |
| `agents` | team size at the end of the run: 4, or 1 for a run that ended before the fork |
| `forked`, `fork_fallback` | the run forked; the runtime forked for the root with the generic direction because the root had not forked within 2 calls of the instruction. Empty for `isolated` |
| `model_calls`, `read_doc_calls` | model calls that reached the model (calls cut off at the end of a run are logged with 0 tokens and not counted); `read_doc` calls, including failed ones |
| `pre_fork_tokens` | tokens charged before the fork, all of them by the root. Empty without a fork |
| `inherited_prompt_tokens` | exploratory, not pre-registered: over the copies' calls after the fork, the input tokens up to the size of the root's history at the fork (the root's last pre-fork call, input plus output), an estimate of what the copies paid to re-read the context they inherited. Empty without a fork |

### Verdicts

- **F1, primary** (`bcp`, `fork` against `isolated`, first claim): **FORKING HURTS**. 20 of 60 against 32 of 60, −20.0 points (95% interval −33.3 to −6.7, 60 pairs, McNemar p = 0.012, Holm 0.035).
- **F2** (both suites, `fork` against `fork-silent`): **NO GAIN**, +1.7 points (−2.5 to +6.7, 120 pairs). Manipulation check: no `fork` run had a substantive message (0.00 messages per agent on `bcp`, 0.01 on `math`), so the two conditions differ in what the agents could do, not in what they did.
- **F3** (the gain is larger on `bcp` than on `math`): does not hold (difference of differences −20.0, one-sided bound −33.3). On `math`, `fork` and `isolated` both solved 23 of 60.
- **Mechanism check:** on `bcp`, the share of documents read by two or more agents of a team fell from 39% (`isolated`) to 18% (`fork`), −20.6 points (−31.1 to −10.3). The frozen duplicate-code measure for `math` is 0 in every cell, as in phase 1: no agent ran code identical to another agent's after stripping whitespace and comments.
- **Fork mechanics:** every `bcp` run forked; on `math`, 10 of 60 runs in each fork condition hit the 30-minute limit before the fork (the threat in §9 of the pre-registration) and count as unsolved. The fallback fork was used in 5 of 60 `bcp` runs per condition and in 8 (`fork`) and 6 (`fork-silent`) of 60 `math` runs; it was 0 in the pilot.
- **Cost:** tokens per solved task, `fork` / `isolated`: 1.85 on `bcp` (1.28 to 2.98), 0.99 on `math` (0.68 to 1.47).

### Exploratory follow-ups (not pre-registered; no verdict depends on them)

- **Where the `bcp` budget went in `fork`** (from `phase2-runs.csv`): the root spent a median of 64,203 tokens before the fork, 26% of all tokens charged in `fork` runs. The instruction arrives at the root's next turn boundary after 10% of the pool is charged, and each of the root's calls re-sends its whole history, so the fork landed at about 20% of the pool. The copies' re-reading of the inherited history was 50% of the tokens. Median model calls per run were 17.5 in `fork` and 27 in `isolated`; median `read_doc` calls, 6 and 9. The pool charges every input token in full, as in phase 1; with input caching billed at a discount, the cost of a shared start would be lower, which this experiment does not test.
- **Near-duplicate code on `math`:** counting a `run_python` call as a near-duplicate when it is at least 90% similar (Python `difflib` ratio, after the frozen normalisation) to an earlier call by another agent of the run, the share is 10% in `fork`, 9% in `fork-silent` and 1% in `isolated`; in `fork`, 15 of those 20 calls match the root's pre-fork code.
