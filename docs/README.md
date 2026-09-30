# Documentation guide

## Competition retrospectives and top solutions

- [`competitive_ml_agentic_playbook.md`](competitive_ml_agentic_playbook.md) — exhaustive post-mortem introspection: why we didn't win gold, 5 critical blind spots, agentic orchestration archetypes, and the 5-phase reusable playbook for future competitions.
- [`top_solutions_and_competitive_postmortem.md`](top_solutions_and_competitive_postmortem.md) — in-depth review of top solutions (3rd, 12th, 14th, 361st), key meta learnings, and side-by-side comparative post-mortem with our solution.
- [`final_competition_retrospective_and_results.md`](final_competition_retrospective_and_results.md) — official final competition results, submission ledger, official private rank, and retrospective.

## Active competition status and plan

- **Start here:** [`endgame_next_steps_and_blockers.md`](endgame_next_steps_and_blockers.md) — current verified state, explicit blockers, ordered next actions, CPU/GPU requirements, and what (if anything) is needed from the user.
- [`competition_idea_backlog.md`](competition_idea_backlog.md) — ranked, testable ideas B0–B6. It is a hypothesis queue, not evidence of score gains.
- [`kernel_audits/0948_cpu_preflight.md`](kernel_audits/0948_cpu_preflight.md) — candidate-specific source/provenance audit and timestamped checks.
- [`final_selection_strategy.md`](final_selection_strategy.md) — submission selection principles; candidate-two settings are hypotheses until validated.

## Evaluation and experiment records

- [`local_evaluation.md`](local_evaluation.md) — official-style local-evaluation protocol and required per-video diagnostics.
- [`local_cv/README.md`](local_cv/README.md) — explanation of the tracked local CV control result.
- [`experiment_tracking.md`](experiment_tracking.md) — append-only experiment/result conventions.
- Machine-readable local CV results: `../results/local_cv/experiments.csv`.
- Machine-readable Kaggle submissions: `../results/kaggle_lb/submissions.csv`.

## Research snapshots and historical plans

- [`kaggle_deep_dive_2026-09-25.md`](kaggle_deep_dive_2026-09-25.md) is a dated public-notebook and leaderboard research snapshot; its ranks are not live.
- [`endgame_status_and_strategy.md`](endgame_status_and_strategy.md), [`kaggle_battle_plan.md`](kaggle_battle_plan.md), [`gold_training_session_handoff.md`](gold_training_session_handoff.md), [`kaggle_cutting_edge_insights.md`](kaggle_cutting_edge_insights.md), [`kaggle_late_stage_research_insights.md`](kaggle_late_stage_research_insights.md), and [`gold_training_baseline_secured.md`](gold_training_baseline_secured.md) are historical or archived. Their old GPU-reset instructions and suggested scores are not current instructions.
- [`0_946_baseline_review.md`](0_946_baseline_review.md) records the proven anchor, while [`final_selection_strategy.md`](final_selection_strategy.md) includes unvalidated Candidate 2 hypotheses; neither replaces the current blocker/evidence checklist.
- Research reports and historical experiment writeups preserve what was learned. For any conflict about the current next action, follow `endgame_next_steps_and_blockers.md`.

## Evidence rules

- Keep notebook names and public claims separate from verified Kaggle scores.
- The 0.948 notebook's named receipt is an internal gate, not a Kaggle-issued artifact. Never invent, hand-edit, or substitute it; the local gate must consume evidence emitted by a real paired fold-disjoint evaluation of the exact candidate and baseline.
- Cached matched predictions can enable CPU-only metric aggregation/post-processing analysis. Generating missing neural predictions may require GPU.
- Kaggle push/run and competition submission are separate external actions; obtain explicit user approval for each.
