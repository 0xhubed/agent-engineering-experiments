# Does letting agents talk to each other help? (polylogue) — pre-registration

**Status: FROZEN, 2026-10-17.** This file was committed before the first scored run; `git log` of this file is the evidence. Drafted from 2026-10-10 on and amended through three pilots (§9a). Tasks, budgets, limits and definitions are frozen in the harness repository `0xhubed/multi-agent-communication` at commit `de104a5` (`suites/*/frozen.json`, `configs/main.json`, `docs/pilot-definitions.md`). That repository is private until the results are published. The answer keys are kept out of git; `frozen.json` holds their hashes.

Built from `multi-agent-communication/IDEA.md` (research question, hypotheses H1–H5, conditions, system design). This pre-registration adds what that design needs: fixed thresholds, a compute-matched comparison, and a sample size that fits the hardware.

## 1. Question and why it matters

When N agents work on the same problem, does letting them **talk** beat the same N agents working **in isolation**? Noam Brown reports ~2× speed from 4 agents at ~2× cost, but with no control experiment and with coordination learned in training. Off-the-shelf models have no such training. Anthropic reports multi-agent setups use about 15× the tokens of a chat. GitSwarm (Meta Superintelligence Labs et al., Oct 2026) shows gains from many agents coordinating only through a shared Git repository, but against a single agent prompted to continue, with compute not matched; on its proof benchmark best-of-N with an oracle verifier reaches the same ceiling. The practical question: is communication worth its tokens, or is "N agents" just "more compute", which N independent attempts plus a verifier (best-of-N) would buy more cheaply?

## 2. Conditions (same model, tools and per-agent budget)

| Condition | What it is |
|---|---|
| `single` | 1 agent |
| `isolated` | N agents, no communication. Scored two ways: by its **first claim** (the team answer, the H2 baseline) and by **best-of-N** with the verifier (any agent's claim passes, the cheap-compute reference) |
| `stigmergic` | N agents sharing only a blackboard (`findings.md`) |
| `polylogue` | N agents with direct messaging and topic broadcast, plus the blackboard, with the protocol norms from IDEA §5 (the first norm is a kickoff: publish a one-line plan before working) |
| `polylogue-bare` | as `polylogue` but **without** the norms in the prompt (tests H4) |

Phase 1 fixes **N = 4**. Scaling to N = 8, 16 is phase 2, run only if phase 1 shows an effect (IDEA §10 step 6). A second phase-2 candidate (2026-10-11): an artifact-building suite (code or a long document) with a shared Git repository as the workspace, plus a `gitswarm` condition (shared repo, no messaging) as in GitSwarm (§8). Phase 1 keeps its question-answering tasks, where branches add little over the blackboard.

## 3. Hypotheses and verdicts (phase 1)

Primary comparison: `polylogue` vs `isolated`, paired by task. Throughout §3, `isolated` means its first-claim score (§2); its best-of-N score is reported alongside as a reference.

- **H2, quality at equal compute (primary):** at the same *total* token budget for the team, `polylogue` solves ≥ 10 points more tasks than `isolated` on `bcp` (95% interval excludes 0). Verdict **TALK HELPS** if met; **ONLY PARALLELISM** if `isolated` ≥ `single` + 10 points but `polylogue` does not beat `isolated`; **NO GAIN** otherwise.
- **H1, speed:** median wall-clock to the first *verified* solution for `polylogue` ≤ 0.8× `isolated`, on tasks both solve.
- **H3, domain:** the `polylogue` − `isolated` gain is larger on the high-parallelism suite than on the medium one (difference of differences, one-sided). The predicted order of the gain is `bcp` > `math`, and the pre-registered H3 test is `bcp` against `math` (D5).
- **H4, collapse without norms:** in `polylogue-bare`, fewer than 30% of agents send a substantive message (one that another agent's next step cites or acts on), and its solve rate is within 5 points of `isolated`.
- **H5, cost:** report tokens per solved task by condition; prediction: `polylogue` costs ≥ 1.5× `isolated` per solved task. Confirmed or not; no verdict hangs on it.
- **Shared knowledge vs conversation:** `stigmergic` vs `polylogue` is reported as a secondary comparison.

## 4. Tasks (phase 1: two suites, automatically verified)

Design: `multi-agent-communication/docs/superpowers/specs/2026-10-11-polylogue-task-suites-design.md`.

- **High parallelism, established (`bcp`):** BrowseComp-Plus, research questions over its **fixed offline corpus** (100,195 documents, BM25 search), so search is reproducible and has no web drift. Selection uses only task properties: an oracle screen (the model answers correctly when given the gold documents) and then the questions whose evidence is most spread out. 30 tasks.
- **Designed suite (`ledger`), dropped from phase 1 (Daniel, 2026-10-15):** generated record collections with aggregation questions, meant to need a split of the reading (the condition Dunas names, §8). In calibration (2026-10-14/15, five world sizes, 36 runs at B = 160k), no condition solved a single world. One agent reads 54–80 documents before its budget runs out, because the runtime charges the whole context on every call. A `polylogue` team divided the reading and covered all 92–100 documents of the smallest worlds, but ran out of its pool while polling for each other's results and aggregating. So a world too large for one agent leaves a team almost no budget for coordination: the team reached only about 25% further than one agent with a quarter of its tokens. That cost of coordination under full-context charging is reported as a pilot finding and goes into the phase-2 design.
- **Medium parallelism (`math`):** generated problems with answers computed in code, and a sandboxed Python tool. No contest problem qualifies, because Qwen3.8-27B was released after AIME and HMMT February 2026. The problems are sublinear number theory (sums of φ(n) and of primes to about 10^11, squarefree counts, coprime pairs, circle lattice points, coprime residue counts), sized so that a correct method only just fits the tool's limits. In calibration `single` solved 9 of 18 without a budget limit, and 3 of 10 pilot problems in P0. 30 tasks.
- **Low parallelism** (writing, design): deferred to phase 2. IDEA §9 lists it as the open problem (no unbiased judge yet), and a phase that cannot be scored cleanly would weaken the rest.

Each task is run with 2 seeds per condition: 2 suites × 30 tasks × 5 conditions × 2 seeds = 600 runs. Five further tasks per suite are used only in the pilot.

## 5. Budgets, verification and stopping

- **Equal compute:** every multi-agent condition gets the same total token budget (4× the single agent's), split however the agents use it. The per-agent budget B is set per suite in the pilot. For `math` it is 1.5 × the median tokens of solved `single` pilot runs (38k). For `bcp` it is the B at which `isolated` solves about 50–60% of pilot tasks (80k): at that rule's 160k, `isolated` solved 9 of 10, which left H2 no room. Messages count against it. A second view at equal *wall-clock* is reported, not used for verdicts.
- **Verification** is outside the agents (answer check or test), as IDEA §5's control layer. **One team answer:** in `stigmergic`, `polylogue` and `polylogue-bare` the first claim ends the run, right or wrong, and the agents are told so. `isolated` agents cannot see each other, so their run goes on until a claim passes and is scored both ways from the log (§2). `single` ends at its claim. A run also ends when the budget or the global wall-clock limit is reached. The budget is hard: calls reserve against the team pool (task-suites spec, amendments).
- **Serving fairness:** N agents share the Sparks' throughput. A pilot measures tokens/s per agent at N = 1 and 4; wall-clock results are reported with that measurement, since part of any speedup is batching.
- **Rate limits** against message storms and the delivery rule (at turn boundaries, `urgent` may jump the queue) are fixed before the first scored run.

## 6. Analysis

Solve rates paired by task (McNemar, Holm over the primary and H1/H3 tests), differences with bootstrap intervals clustered by task. Event-log metrics (messages per agent, idle time, duplicate work, wrong consensus, time to first useful message) are reported descriptively; "duplicate work" and "substantive message" get operational definitions in the pilot, frozen before the main run.

## 7. Exclusions

Infrastructure failures are rerun once, then counted as unsolved. Tasks found to have a wrong answer key are removed from all conditions and reported.

## 8. Threats

Prompted norms are not trained coordination (Brown's point, and H4 tests it); one model family. **Manipulation check:** H2 is only interpretable if `polylogue` agents actually communicate. In pilot P1 (old norms), no agent sent a message on `bcp` or `math` in 40 runs, which is why the kickoff norm was added. Messages per agent for every condition, and the share of `polylogue` runs with at least one substantive message (definition: `multi-agent-communication/docs/pilot-definitions.md`), are reported next to the H2 verdict, with no threshold. In P1b, with the kickoff norm, every `polylogue` run sent messages, but only one opening plan per agent. The plans were all alike and none was followed up. So H2 tests talk as untrained agents practise it. Claims per run and the false-claim rate are reported; P1 had no wrong claims (56 claims).

**Related framing (2026-10-11):** K. Dunas, ["On the Nature of the Swarm"](https://www.primeintellect.ai/blog/on-the-nature-of-the-swarm) (Prime Intellect, essay without experiments) argues that agents in parallel mainly add context, roughly one window per agent, and pay off only when the task splits into parts that communicate in short messages. If phase-1 tasks fit in one window, that effect cannot show: the task-suite spec (D2) should include tasks whose evidence exceeds one window. His tree-vs-mesh point (the root as bottleneck) suggests a coordinator-led condition for phase 2.

**Related work (2026-10-11):** Shah et al., ["GitSwarm: Decentralized Compounding Inference"](https://arxiv.org/abs/2610.04862) (arXiv:2610.04862) is in effect a large-scale `stigmergic` condition: no messaging, a shared branchable Git repo, and a mandatory `informed_by=` field on every commit. Three points for this design: it has no `isolated` (best-of-N) control at matched compute, which is the gap this experiment fills; on proofs later workers mostly select among existing candidates while on code 95% of contributions are reused, which supports H3's ordering; and its reuse links and abstentions (about one in six code workers stopped after finding equivalent work) inform the pilot's definitions of "substantive message" and "duplicate work". Those are measured from the event log; no citation field is added to the tools, since that would nudge the agents.

## 9. Cost

P0 measured about 4 min per `single` run on `bcp` and about 25 min on `math`, where a first call can reason for 32k tokens before using the tool. At an average of about 15 min, 600 runs take about 150 hours on one Spark and about 75 on two, plus the pilot (P1, 100 runs) and the calibrations. Building the runtime and viewer (IDEA §10 steps 1–3) comes first and is the larger effort.

## 9a. Pilot and freeze (2026-10-17)

P0, P1 and P1b (100 runs each for P1 and P1b, on 5 pilot tasks per suite) fixed the budgets and limits. The suites, budgets and definitions were frozen in `multi-agent-communication` at commit de104a5: `bcp` B = 80k, `math` B = 38k, 30 minutes, 10 messages per minute, 2,000 characters. P1b solved (of 10): bcp single 1, isolated 5, stigmergic 6, polylogue 6, polylogue-bare 4; math 2, 3, 4, 4, 4. These are pilot runs, not results.

## 10. Decisions (all made by Daniel)

- **D1 Model:** ~~a fast MoE or the pinned model~~ **Decided 2026-10-11: the pinned Qwen3.8-27B**, for consistency with the other experiments.
- **D2** ~~High-parallelism suite~~ **Decided 2026-10-11: three suites** — BrowseComp-Plus (BM25, oracle screen, spread ranking), a designed `ledger` suite whose evidence exceeds one agent's budget, and competition math with a Python tool (§4). **Amended 2026-10-15: `ledger` dropped from phase 1** after its calibration (§4); math is all generated.
- **D5** ~~Primary suite~~ **Decided 2026-10-12: `bcp` carries the primary H2 test**; `math` is secondary (`ledger` was dropped on 2026-10-15). H3 contrasts `bcp` with `math`.
- **D3** ~~Whether `polylogue-bare` is in phase 1~~ **Decided 2026-10-11: included** (+20% runs), so a win can be attributed to the norms or to the talking.
- **D4** ~~A frontier model as an independent verifier~~ **Decided 2026-10-17: none.** P1 and P1b had no wrong claims, and the deterministic verifier's known bias (missing correct answers written in another form) affects every condition alike.
