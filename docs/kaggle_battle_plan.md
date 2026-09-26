# Archived: Kaggle Gold-Zone Battle Plan

> **Do not execute this plan.** It was written before the 0.948 candidate's provenance and validation blockers were established and contains stale GPU-reset assumptions, unsupported score expectations, and superseded training steps. The active plan is [`endgame_next_steps_and_blockers.md`](endgame_next_steps_and_blockers.md); the ordered experiments are in [`competition_idea_backlog.md`](competition_idea_backlog.md).

This document is retained only to explain the prior direction. It proposed running the public Gold OOF workflow, restarting custom training, and later ensembling checkpoints. Those actions were not shown to validate the staged EMA candidate and are not the current competition endgame. In particular, do not resume `temporal-pu-a` from epoch 10: its recorded OOF score (`0.0079`) and node recall (`0.017`) must be diagnosed before any new training allocation.

No kernel push/run or competition submission is authorized here. The current next action is CPU-side: define a real fail-closed paired fold-disjoint evaluation/receipt path and locate the required OOF artifacts. Fresh model predictions may require GPU; a Kaggle inference run and a competition submission each require separate explicit approval after readiness checks.
