# Archived gold-training session handoff (2026-09-14)

> **Historical record only — do not follow the old GPU-quota instructions.** The active competition endgame, current blockers, and CPU/GPU split are maintained in [`endgame_next_steps_and_blockers.md`](endgame_next_steps_and_blockers.md). The original plan predated the 0.948 provenance audit and should not be used to launch Kaggle training or inference.

The experiment ledger records the clean public 50-epoch reference checkpoint at **0.90325** on its two disjoint prefix holdouts (Fold A `0.89746`, Fold B `0.90903`, run recorded 2026-09-14). This is a historical local CV result for that model; it is not evidence for the staged 0.948 EMA change or a Kaggle public score.

The prior handoff's claims that the pipeline was “bug-free,” that GPU work should start immediately after quota reset, and that custom training would reach `0.94+` were plans/expectations, not acceptance evidence. Do not treat them as current results or instructions. For training-workflow design, see [`gold_training_implementation.md`](gold_training_implementation.md) and [`cloud_kfold_training_design.md`](cloud_kfold_training_design.md); for the current competition priority, follow the canonical endgame status document.
