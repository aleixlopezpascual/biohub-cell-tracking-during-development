# Biohub Endgame: Current Status, Blockers, and Next Actions

**Status checked:** 2026-09-26 11:58 CEST.

**Competition deadline on record:** 2026-09-29 23:59 UTC (from the project research snapshot; refresh Kaggle's competition page if planning near the deadline).

**Purpose:** Single current source of truth for what remains, what requires GPU, and what is blocked.

**Priority:** B0 is the current readiness/evidence gate; B1–B6 are the ordered experiment backlog, not commitments or demonstrated gains. See [`competition_idea_backlog.md`](competition_idea_backlog.md) for experiment rationale.

## Bottom line

The proven anchor remains Kaggle submission `56132481` at public score **0.946**. The staged `biohub-0-948-momentum-deepcenter-tta` is **unscored**: `0.948` is part of its name, not a result. No new Kaggle run or submission has been verified. Do not push, run, submit, or select the candidate until its provenance/readiness and validation requirements are resolved; a Kaggle push/run and a competition submission each require separate explicit user approval.

**No receipt is available for the user to fetch.** The staged notebook names `bidirectional_blend_union13_receipt.json`, but the repository has no producer or validator for it. This is a notebook-declared internal promotion artifact, not a Kaggle-issued file or official competition requirement. Its audit cell records a requirement; it does not read the receipt or enforce promotion. The notebook's referenced upstream diagnostic source was not verified and must not be treated as evidence. The actual need is a genuine paired fold-disjoint evaluation artifact; either implement a real validator/producer for it or correct the notebook's local gate to consume that real evidence. Do not fabricate a receipt, rename another experiment's `promotion.json`, or treat the synthetic EMA tests as validation evidence.

## Verified state

- At **2026-09-26 11:58 CEST**, a read-only Kaggle submissions query returned five records. The newest scored result remains submission `56132481`, score `0.946`; there is no `0.948` submission record. Private scores were blank.
- Kaggle's kernel listing reports `aleixlopez/biohub-0-948-momentum-deepcenter-tta` with raw `lastRunTime` `2026-09-24T15:14:08.107000` (timezone not supplied). That timestamp does not establish a successful run or available output. Earlier read-only checks returned no kernel files, and the session status/output endpoints returned 404.
- Current local notebook source differs in cells 0, 1, 5, and 6. Cells 0, 1, and 6 are the approved score/provenance/report corrections; the only tracking-algorithm change is cell 5's EMA motion relinker. DeepCenter is already in both. The synthetic CPU tests exercise that helper only: focused tests **2 passed**; the full suite most recently passed **125 passed, 4 skipped**. They do not measure score, OOF quality, runtime artifact loading, or promotion.
- Searches of the project and the known prior CV workspace did not find the named receipt, matching fold-held-out predictions/graphs, or candidate model checkpoints. The project search also found no local validation Zarr stores. Raw training images/labels, a different candidate's `promotion.json`, or public weights trained on the validation movies are not substitutes for fold-disjoint candidate evidence.
- GPU availability is **not live-verified**. The user previously reported it was unavailable. The staged kernel sidecar requests GPU/T4, while its embedded notebook metadata says GPU disabled; resolve this discrepancy and confirm actual T4 visibility before relying on a Kaggle run.
- The public-rank figures in the September 25 research snapshot are historical only. Refresh the leaderboard before making current rank/medal claims.

## Blockers

| Blocker | What is missing | What clears it | GPU? |
|---|---|---|---|
| **B0.1 — No executable promotion-evidence path** | A real schema, producer, and validator for the internal artifact the notebook declares. Current report fields are descriptive only; this is not a Kaggle-issued requirement. | Implement a fail-closed evaluator/evidence path that verifies provenance and derives promotion from actual paired results; tests must reject missing, mismatched, or failing evidence. Correct the local notebook gate to consume that real evidence. | **No** to design, implementation, and synthetic tests. |
| **B0.2 — No valid paired fold-disjoint evidence** | Parent and EMA outputs on identical held-out movies, from weights that did not train on those held-out movies, plus the split and artifact provenance. | Generate or locate matched OOF predictions, then score both variants with the official metric and per-movie diagnostics. Keep all variables fixed except the EMA relinking change. | **Conditional:** CPU if valid cached OOF predictions and labels already exist; **GPU likely** if fresh neural inference or fold-excluded model predictions must be generated. Training fold-excluded models would also need suitable compute. |
| **B0.3 — Runtime/source provenance mismatch** | Conflicting GPU flags; the remote Kaggle copy predates local score/provenance corrections; referenced upstream source and actual runtime artifact hashes remain unverified. | Confirm local corrections, verify/resolve source attribution, align notebook/sidecar config, and check model/checkpoint hashes plus active DeepCenter/dual-seed behavior in an authorized runtime. | **No** for static reconciliation; **yes** to establish actual GPU visibility and verify mounted runtime artifacts. |
| **B0.4 — No verified candidate inference output** | No readable run output, runtime log, or candidate submission score. | After readiness checks, run the staged candidate and inspect schema/topology, hashes, fallbacks, runtime, and Kaggle read-back. | **Yes** for the staged T4 inference path. |

## Ordered next actions

1. **CPU — make the validation gate real.** Specify and implement the paired-OOF evaluator and receipt validator. The report must identify the exact parent and candidate, commit/source hashes, folds and fold-excluded weights, prediction provenance, official metric version, aggregate and per-video metrics, and the predeclared pass/fail criteria. It must fail closed when any evidence is missing or mismatched. Do not set `promote=true` by hand.
2. **CPU discovery; GPU may be needed to fill gaps — check the exact evidence inputs.** Locate the fold split, held-out labels, and any parent/candidate OOF detections/edge predictions and fold-excluded weights. Record hashes and training-fold provenance. If the only model weights were trained on the evaluation movies, they cannot validate those same movies; generate proper fold-excluded predictions instead. If artifacts exist outside the previously searched locations, their path can be supplied, but no receipt needs to be supplied.
3. **CPU scoring once evidence exists; GPU likely for fresh predictions.** Run the same fold-held-out predictions through baseline and EMA tracking, changing only the relinker. Compare official aggregate score and per-fold/per-video adjusted edge Jaccard, division Jaccard, node-count ratio, duplicates, fragmented/detection-lost/wrong-association edges, and runtime. Register a real result in `results/local_cv/experiments.csv`; write an explanatory report under `docs/`. Only the evaluator may create a passing receipt.
4. **Before any Kaggle kernel run — CPU metadata check, then GPU verification.** Align `kernel-metadata.json` with embedded notebook metadata, verify dataset versions and expected checkpoint hashes, then confirm the T4 is actually visible and the runtime manifest shows all required models/vetoes loaded. A metadata flag alone is not proof.
5. **After validation and separate user approval — GPU candidate inference.** Run the current corrected local notebook, inspect the actual output and logs, and read back Kaggle's result. Record only the real result. A test-set run is not a substitute for the fold-disjoint promotion evaluation.
6. **After a separately approved competition submission — CPU read-back/logging.** Verify the submission status and score, append the record to `results/kaggle_lb/submissions.csv`, and preserve `56132481` as the proven anchor. Do not fill a second final-submission slot without a materially distinct, validated candidate.
7. **Only after the B0 evidence exists:** B1 node-count stability audit, then B2 isolated mutual-best association test. Consider B3 neighborhood-flow only if held-out failures justify it. B4 pretrained division verifier and B5 DoG are lower priority; B6 synthetic-data training is post-competition work.

## GPU and CPU map for the experiment backlog

| Item | GPU requirement | Current state |
|---|---|---|
| Provenance, metadata review, receipt/evaluator implementation, synthetic tests | **No** | Partly complete; receipt/evaluator remains to be implemented. |
| Paired OOF model outputs for B0 | **Only if missing predictions must be generated; likely yes** | Missing in searched project/known CV paths. |
| Official metric aggregation / paired analysis for B0 and B1 | **No**, if matching predictions and labels are present | Blocked by missing inputs, not by the scorer requiring a GPU. |
| B1 node-count stability | **No** with cached fold-held-out parent/candidate outputs; **GPU likely** to regenerate neural outputs | Blocked on OOF artifacts. |
| B2 mutual-best association | **No** with cached OOF edge scores; **GPU likely** to regenerate them | Blocked on OOF edge predictions. |
| B3 neighborhood-flow override | **No** for CPU post-processing on available clips/predictions; **GPU only if model outputs need regeneration** | No OOF graph/image cache; conditional and lower priority. |
| B4 pretrained division verifier | **No new training planned.** Inference may need GPU depending on the model/artifact and runtime. | Optional; artifact/license/offline availability must be checked first. |
| B5 multi-scale DoG proposal source | **No** for the detector/post-processing experiment | Needs validation images and paired scoring; none are currently available locally. |
| B6 synthetic microscopy model training | **Yes** for practical training | Defer until after the competition. |
| Final Kaggle submission selection | **No additional GPU** to select/read back, though candidate generation may have needed it | Requires explicit user approval; do not select an unvalidated candidate. |

## Explicit no-go items

- Do not look for or manually create `bidirectional_blend_union13_receipt.json`; it must be produced by the real paired evaluation once that path exists.
- Do not substitute `gold_oof_runner`'s `promotion.json`; it is for a different candidate.
- Do not call the staged candidate `0.948` as a measured score.
- Do not bypass provenance, fold-disjoint validation, hash checks, or the candidate's fail-closed quality gate to save quota.
- Do not run blind threshold/short-track/division sweeps on the public board, use the patched metric exploit, or resume `temporal-pu-a` without diagnosing its recorded poor result.
- Do not push/run or submit without the applicable explicit approval.

## What is needed from the user

**Nothing needs acceptance or a receipt from the user right now.** The helpful optional updates are: (a) whether GPU access has changed, or (b) the location of any matching fold-disjoint OOF predictions/weights/splits not in the previously searched paths. We will ask separately before any Kaggle push/run or competition submission.

## Related documents

- Ordered hypotheses and experimental designs: [`competition_idea_backlog.md`](competition_idea_backlog.md)
- Candidate-specific evidence and audit trail: [`kernel_audits/0948_cpu_preflight.md`](kernel_audits/0948_cpu_preflight.md)
- Local CV commands and metrics: [`local_evaluation.md`](local_evaluation.md) and [`local_cv/README.md`](local_cv/README.md)
- Historical notebook/discussion research snapshot: [`kaggle_deep_dive_2026-09-25.md`](kaggle_deep_dive_2026-09-25.md)
- Append-only experiment and Kaggle submission ledgers: `results/local_cv/experiments.csv`, `results/kaggle_lb/submissions.csv`
