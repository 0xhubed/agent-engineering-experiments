# Do agents that fork from a shared start beat independent agents? (polylogue phase 2) — pre-registration

**Status: FROZEN, 2026-10-25.** This file was committed before the first scored run of phase 2; `git log` of this file is the evidence. The design, conditions, prompts, analysis code and run configuration are frozen in the harness repository `0xhubed/multi-agent-communication` at commit `4423cd9`: `configs/phase2.json`, `packages/runtime/src/fork/texts.ts`, `packages/runtime/src/conditions.ts`, `prep/polyprep/analysis.py`, and the design `docs/superpowers/specs/2026-10-22-polylogue-phase2-fork-design.md`. The tasks, answer-key hashes, aliases and metric definitions are the phase-1 ones, unchanged (`suites/*/frozen.json`, `docs/pilot-definitions.md` with its amendment of 2026-10-18). That repository is private. The phase-1 harness is public as a snapshot ([`0xhubed/polylogue`](https://github.com/0xhubed/polylogue)), and the phase-2 code will be published the same way with the results.

Phase 1 is [`07-multi-agent-communication.md`](07-multi-agent-communication.md); its results are in [`results/polylogue/`](../results/polylogue/).

## 1. Question and why it matters

Phase 1 found that four Qwen3.8-27B agents that can message each other do not beat four agents working alone at the same total budget (verdict ONLY PARALLELISM: on `bcp`, `isolated` beat `single` by 23 points, and `polylogue` was 3 points below `isolated`). The agents barely used the channel. All four started from the same prompt and did the same work: about 40% of the documents an `isolated` team read were read by two or more of its agents, and 29 of the 30 unsolved `isolated` runs on `bcp` ended because the budget ran out.

Noam Brown's account of OpenAI's multi-agent system describes two things phase 1 did not have. Agents **fork**: a helper starts as a copy of its parent's full context, not from the task description. And there is **no coordinator**, only flat messaging. Brown also says that good coordination comes from training, and that untrained agents collapse into independent solving, which is what phase 1 measured.

Phase 2 tests the forking half of that design on the same untrained model: **do helpers that start from a shared understanding, with directions the agent chose itself, beat four independent starts at the same budget?** Training is not part of this phase.

## 2. Conditions (same model, tools, tasks and team budget)

| Condition | What it is |
|---|---|
| `fork` | One root agent works alone. At a fixed point it must fork into 3 copies, with a direction for each that it writes itself. The 4 agents then work on with flat messaging (send, publish, subscribe, roster); no blackboard and no norms. |
| `fork-silent` | The same fork, with no communication tools after it. The claim rule is the same as in `fork`, so the only difference from `fork` is messaging. |
| `isolated` | As in phase 1: 4 agents, no contact, scored by first claim; best-of-4 with the verifier reported as a reference. Rerun so that all three conditions share one runtime version and the same days. |

N = 4 throughout. Forks do not fork again.

## 3. The fork mechanism

1. **Root alone.** The run starts with one agent, `a1`, with the work tools only. It **cannot claim before it has forked**: `claim` becomes available to all four agents after the fork. Its system prompt says that it will split the work with three copies of itself and that claiming opens after the split.
2. **Fork point.** When the tokens charged to the team pool reach **10% of the pool** (`bcp`: 32,000 of 320,000; `math`: 15,200 of 152,000), the runtime tells `a1` at its next turn boundary to fork now and gives it the `fork` tool. **Early instruction:** if `a1` ends a turn without a tool call before that point, the runtime gives the instruction then, because the agent cannot go on without a new message. The 10% is fixed a priori and was not tuned in the pilot.
3. **The fork call.** `fork(directions)` takes exactly 3 directions, each at most 300 characters, pairwise different. A malformed call (wrong count, duplicates, too long) gets an error result and counts as one of 2 allowed calls. If `a1` has not forked within 2 calls of the instruction, the runtime forks for it with the direction "Take a different approach from your siblings." for all three copies and logs the fork as a fallback.
4. **Copies.** Agents `a2`–`a4` each start with `a1`'s full message history up to and including the fork call, plus their own direction as a new user message, and a system prompt that names them, their siblings and (in `fork`) the messaging tools. `a1` gets a tool result listing the three directions and continues as the fourth agent. Every agent's context can be rebuilt from the event log alone.
5. **Prompts** were reviewed and approved before the pilot (Daniel, 2026-10-24; `docs/phase2-fork-prompts-for-review.md` in the harness repository).

## 4. Hypotheses and verdicts

All scores are first-claim scores, paired by (task, seed).

- **F1, primary (`bcp`):** `fork` − `isolated`. **FORKING HELPS** if the effect is ≥ +10 points and its 95% interval excludes 0; **FORKING HURTS** if the interval lies entirely below 0; **NO GAIN** otherwise.
- **F2 (both suites pooled):** `fork` − `fork-silent`: does talking after the fork add anything? **TALK ADDS** if ≥ +10 points with the interval excluding 0; **NO GAIN** otherwise. F2 is pooled over `bcp` and `math`, not stratified; per-suite values are reported next to it.
- **F3, domain:** the `fork` − `isolated` gain is larger on `bcp` than on `math` (difference of differences, one-sided). F3 holds if the point estimate is > 0 and the one-sided 95% bound is > 0.
- **Mechanism check (descriptive, with intervals, no verdict):** duplicate reads (`bcp`) and duplicate Python code (`math`) per team, `fork` and `fork-silent` against `isolated`. The frozen definitions apply unchanged over the whole run: `a1`'s reads before the fork count once, and a copy re-reading a document it inherited counts as a duplicate, since that content is already in its context. A post-fork-only share is reported next to it.
- **Fork mechanics (descriptive):** per suite and condition, the share of runs that forked, that ended before the fork, and that used the fallback; whether each run's directions are distinct (a floor, since the tool rejects exact duplicates) and the maximum pairwise word overlap (Jaccard) of each run's directions.
- **Manipulation check for F2:** messages per agent, and the share of `fork` runs with at least one substantive message (`docs/pilot-definitions.md`), are reported next to F2, with no threshold.
- **Cost:** tokens per solved task by condition, and the `fork` / `isolated` ratio with its interval. No verdict depends on it.

**Multiple testing:** Holm over three tests at a family-wise α of 0.05: F1 (exact McNemar), F2 (exact McNemar), F3 (bootstrap one-sided p). As in phase 1, the F1 and F2 verdicts follow the interval. If the Holm-adjusted McNemar p is ≥ 0.05 while the interval excludes 0, the report says so next to the verdict.

**Phase-1 numbers are not a comparison.** Phase 2 has its own `isolated` baseline; phase-1 runs are not pooled into any phase-2 test.

## 5. Tasks and sample

The same 60 scored tasks as phase 1 (30 `bcp`, 30 `math`; §4 of the phase-1 pre-registration). Nothing is trained or tuned on them, and reuse allows a task-by-task comparison with phase 1, reported descriptively. 3 conditions × 60 tasks × 2 seeds = **360 runs**, in a seeded random order (`orderSeed` 1), so the conditions are interleaved over the same days. About 3 days on one Spark.

## 6. Budgets, verification and stopping

As in phase 1 unless stated: per-agent budget B (`bcp` 80k, `math` 38k), one team pool of 4 × B for the whole run, with reservations so that the pool is never exceeded. `a1`'s tokens before the fork count against the pool, and so does each copy's re-reading of its inherited context: that is the cost of forking. The first claim ends the run in `fork` and `fork-silent`; `isolated` goes on after a wrong claim and is scored both ways. 30-minute wall clock, per-call cap 32,768 tokens, the same model revision (`unsloth/Qwen3.8-27B-NVFP4` at `7d6f8d4`) and sampling (temperature 0.6, top-p 0.95, top-k 20), and an outside verifier.

**Intention to treat.** Runs are scored in the condition they were assigned to. A run that ends before it forks (by the wall clock or the budget, since claiming is closed until the fork) stays in its condition as unsolved, and the rate of such runs is reported.

## 7. Analysis

`prep/polyprep/analysis.py`, phase-2 path, as committed at `4423cd9`. It uses the Python standard library only. Effects are differences in solve rate in percentage points. Intervals are 95% percentile bootstrap intervals clustered by task (10,000 resamples, generator seed 20261017); the F3 bootstrap resamples tasks within each suite independently. The analysis refuses to give verdicts while any cell is missing, and stops if any run's task fingerprint differs from `frozen.json`.

## 8. Exclusions

Infrastructure failures are rerun once, then counted as unsolved. The aliases and answer keys are frozen; a task is removed from all conditions only if its answer key is shown to be wrong by its source, decided from the claims without their condition labels, and every removal is reported. No other exclusions.

## 9. Threats

- **The root's solo start costs wall-clock time.** Before the fork, one agent works where `isolated` has four. In pilot P2, the root's first call on two `math` tasks reasoned for 16–27 minutes, and those runs hit the 30-minute limit before they could fork (4 of 20 `math` pilot runs; no `bcp` run). This was seen before the scored run and is kept as designed (Daniel, 2026-10-25). No change to the fork trigger would help: the instruction can only arrive at a turn boundary, and these runs never reached one. Such runs count as unsolved under intention to treat, and the rate is reported. For comparison, the phase-1 main run lost 4 of 60 `isolated` `math` runs to the wall clock.
- **Shared start between `fork` and `fork-silent`.** Both conditions give the root the same prompt, and the agents' sampling seeds depend only on the run seed. So a `fork` run and a `fork-silent` run with the same task and seed often have the same history up to the fork, down to the directions. That pairing is intended for F2, which then compares runs that differ mostly after the fork. It does not apply to `isolated`, whose prompt differs.
- **Forced timing.** When to fork is not the agent's decision in this phase, so a positive result says that a shared start helps at this fork point, not that the agent would choose to fork.
- **Untrained coordination**, one model family, and an outside verifier with the known bias of phase 1 (a correct answer written in another form can be missed), which affects every condition alike.

## 10. Pilot P2 and freeze (2026-10-24/25)

A one-run smoke test and pilot P2 (`fork` and `fork-silent` on the 10 pilot tasks, 2 seeds, 40 runs) checked the mechanics only. Solve rates were not used for any decision.

- 36 of 40 runs forked; the other 4 are the `math` runs in §9 that ended before the fork. **0 fallbacks** (the stop rule was more than 20%).
- In every forked run the three directions were distinct. Read one by one, they took separate clues of the question, separate methods, or separate checks of an answer the root already had; in a few `math` runs two of the three were variants of one method (the same sieve at a different size). The median of the maximum pairwise word overlap was 0.19 (1.00 = the same words).
- Every copy's first call carried the root's full history. A check in the pilot report that flagged 9 copies as short was a false alarm: those calls were cut off at the end of the run and logged with 0 tokens. The check was fixed.
- No crashes. Two runs were lost to dropped connections and reran normally; a short network outage stopped the batch once, and it was resumed on the same log.

## 11. Decisions (all made by Daniel)

- **Conditions:** `fork`, `fork-silent` and a fresh `isolated` (2026-10-22).
- **Fork timing:** forced at 10% of the pool; directions written by the agent (2026-10-22).
- **No claim before the fork** (2026-10-22).
- **Prompts** approved (2026-10-24).
- **Wall-clock runs that end before the fork:** keep the design and score as assigned (2026-10-25).
