# Biohub Endgame: Current Status, Blockers, and Next Actions

**Status checked:** 2026-09-27 15:59 CEST for local code/tests; the latest read-only Kaggle submissions query recorded below is 2026-09-26 11:58 CEST.

**Competition deadline on record:** 2026-09-29 23:59 UTC (from the project research snapshot; refresh Kaggle's competition page if planning near the deadline).

**Purpose:** Single current source of truth for what remains, what requires GPU, and what is blocked.

**Priority:** B0 is the current readiness/evidence gate; B1–B6 are the ordered experiment backlog, not commitments or demonstrated gains. See [`competition_idea_backlog.md`](competition_idea_backlog.md) for experiment rationale.

## Bottom line

The proven anchor remains Kaggle submission `56132481` at public score **0.946**. The staged `biohub-0-948-momentum-deepcenter-tta` is **unscored**: `0.948` is part of its name, not a result. No new Kaggle run or submission has been verified. Do not push, run, submit, or select the candidate until its provenance/readiness and validation requirements are resolved; a Kaggle push/run and a competition submission each require separate explicit user approval.

**No promotion receipt or real score evidence is available.** The notebook names `bidirectional_blend_union13_receipt.json`; it is a notebook-declared internal promotion artifact, not a Kaggle-issued file or official competition requirement. Its audit cell records a requirement but does not read or enforce the receipt. The CPU-only validator at `scripts/evaluate_promotion_gate.py` now checks edge-score arithmetic but deliberately always returns `promote=false`: schema v1 lacks trusted runtime/scorer attestation. A limited `scripts/run_oof_checkpoint.py --motion-relink` Gold-model ablation path emits local per-prediction execution records, but it does not emit the validator's complete four-run `evidence.json` contract or reproduce the exact `0.946` Kaggle pipeline. The validator is not wired into the Kaggle notebook. No real EMA predictions, paired evidence, or score improvement were generated. Do not hand-author the evidence manifest, fabricate a receipt, rename another experiment's `promotion.json`, or treat synthetic tests as validation evidence.

## Verified state

- At **2026-09-26 11:58 CEST**, a read-only Kaggle submissions query returned five records. The newest scored result remains submission `56132481`, score `0.946`; there is no `0.948` submission record. Private scores were blank.
- Kaggle's kernel listing reports `aleixlopez/biohub-0-948-momentum-deepcenter-tta` with raw `lastRunTime` `2026-09-24T15:14:08.107000` (timezone not supplied). That timestamp does not establish a successful run or available output. Earlier read-only checks returned no kernel files, and the session status/output endpoints returned 404.
- Current local notebook source differs in cells 0, 1, 5, and 6. Cells 0, 1, and 6 are the approved score/provenance/report corrections; the only tracking-algorithm change is cell 5's EMA motion relinker. DeepCenter is already in both. The original synthetic CPU tests exercised that helper only: focused tests **2 passed**; at that earlier checkpoint the full suite passed **143 passed, 4 skipped**. These tests do not measure score, OOF quality, runtime artifact loading, or promotion.
- Searches of the project and the known prior CV workspace did not find the named receipt, matching fold-held-out predictions/graphs, or candidate model checkpoints. The project search also found no local validation Zarr stores. Raw training images/labels, a different candidate's `promotion.json`, or public weights trained on the validation movies are not substitutes for fold-disjoint candidate evidence.
- Local code update at **2026-09-26 17:25 CEST**: added the CPU paired-OOF validator and **18 synthetic-only tests passed**. The tests exercise its input contract and do not create usable model predictions, attest that EMA code ran, or establish any score. SHA256 checks establish artifact identity/integrity; the later Gold OOF mode adds local execution metadata but does not yet provide the complete, independently reviewed paired-run evidence required for promotion.
- Local code update at **2026-09-27 14:57 CEST**: added the physical-micron EMA relinker and opt-in Gold OOF `--motion-relink` path with strict indexed-edge validation, source/config records, and per-GEFF hash-bound sidecars. Focused tests: **18 passed**; full suite: **155 passed, 4 skipped**; CLI help, compilation, and `git diff --check` passed. No inference, OOF artifacts, or score were generated. The path does not pin the external predictor source or emit the complete four-run promotion-gate package, and it does not validate the exact `0.946` Kaggle pipeline. Ruff/mypy are unavailable in this Python environment.
- Independent review blockers were addressed locally and verified at **2026-09-27 15:59 CEST**: reject non-component candidate/dataset names before path use; bind EMA reuse sidecars to checkpoint and upstream Python-source hashes; recompute adjusted edge Jaccard from TP/FP/FN and node-count ratio; and make the schema-v1 structural report non-promotable without trusted attestation. Focused regression tests passed **42**; the full suite passed **161 tests, 4 skipped**. CLI help, compilation, and `git diff --check` passed. No real OOF data or scores were produced.
- GPU availability is **not live-verified**. The user previously reported it was unavailable. The staged kernel sidecar requests GPU/T4, while its embedded notebook metadata says GPU disabled; resolve this discrepancy and confirm actual T4 visibility before relying on a Kaggle run.
- The public-rank figures in the September 25 research snapshot are historical only. Refresh the leaderboard before making current rank/medal claims.

## Blockers

| Blocker | What is missing | What clears it | GPU? |
|---|---|---|---|
| **B0.1 — Trusted paired evidence and promotion attestation missing** | The schema-v1 evaluator reports metric results but always blocks promotion because its hashes are self-declared and do not attest source/scorer execution. The optional Gold OOF relinker does not emit the complete four-run `evidence.json`; the notebook does not consume a local assessment. The one-to-one relinker also omits the candidate notebook's later division-repair stage. | Design a trusted execution/scorer attestation and reviewed producer; generate baseline and EMA variants on both folds with real fold-excluded weights; bind scorer/split/config/checkpoint/upstream source/predictions; then independently verify the report. This Gold OOF ablation does not validate the exact `0.946` Kaggle pipeline. | **No** for packaging/CPU scoring; **GPU likely** if the missing neural predictions must be generated. |
| **B0.2 — No valid paired fold-disjoint evidence** | Parent and EMA outputs on identical held-out movies, from weights that did not train on those held-out movies, plus the split and artifact provenance. | Generate or locate matched OOF predictions, then score both variants with the official metric and per-movie diagnostics. Keep all variables fixed except the EMA relinking change. | **Conditional:** CPU if valid cached OOF predictions and labels already exist; **GPU likely** if fresh neural inference or fold-excluded model predictions must be generated. Training fold-excluded models would also need suitable compute. |
| **B0.3 — Runtime/source provenance mismatch** | The remote Kaggle copy predates local score/provenance corrections and conflicts on GPU metadata; actual runtime artifact loading remains unverified. Local EMA sidecars now hash the configured upstream Python source tree and checkpoint, but these are still self-declared local records. | Confirm local corrections, verify/resolve source attribution, align notebook/sidecar config, and check model/checkpoint hashes plus active DeepCenter/dual-seed behavior in an authorized runtime. | **No** for static reconciliation; **yes** to establish actual GPU visibility and verify mounted runtime artifacts. |
| **B0.4 — No verified candidate inference output** | No readable run output, runtime log, or candidate submission score. | After readiness checks, run the staged candidate and inspect schema/topology, hashes, fallbacks, runtime, and Kaggle read-back. | **Yes** for the staged T4 inference path. |

## Ordered next actions

1. **CPU — design a trusted evidence path.** The schema-v1 structural evaluator checks arithmetic and regression rules but always emits `promote=false` because it cannot authenticate execution or scorer provenance. The limited Gold OOF mode binds local sidecars to source/checkpoint/prediction hashes, but those are not trusted attestations; it also omits the candidate notebook's later division-repair stage. Do not set `promote=true` by hand or create an evidence manifest from unverified claims.
2. **CPU discovery; GPU may be needed to fill gaps — check the exact evidence inputs.** Locate the fold split, held-out labels, and any parent/candidate OOF detections/edge predictions and fold-excluded weights. Record hashes and training-fold provenance. If the only model weights were trained on the evaluation movies, they cannot validate those same movies; generate proper fold-excluded predictions instead. If artifacts exist outside the previously searched locations, their path can be supplied, but no receipt needs to be supplied.
3. **CPU scoring once evidence exists; GPU likely for fresh predictions.** Run the same fold-held-out predictions through baseline and EMA tracking, changing only the relinker. Enforce pooled gain of at least `+0.001` and no fold or held-out-volume score drop beyond `-0.003`; review per-fold/per-video adjusted edge Jaccard, division Jaccard, node-count ratio, duplicates, fragmented/detection-lost/wrong-association edges, and runtime. Register a real result in `results/local_cv/experiments.csv`; write an explanatory report under `docs/`. A schema-v1 report cannot authorize promotion; add and independently verify a trusted attestation path first.
4. **Before any Kaggle kernel run — CPU metadata check, then GPU verification.** Align `kernel-metadata.json` with embedded notebook metadata, verify dataset versions and expected checkpoint hashes, then confirm the T4 is actually visible and the runtime manifest shows all required models/vetoes loaded. A metadata flag alone is not proof.
5. **After validation and separate user approval — GPU candidate inference.** Run the current corrected local notebook, inspect the actual output and logs, and read back Kaggle's result. Record only the real result. A test-set run is not a substitute for the fold-disjoint promotion evaluation.
6. **After a separately approved competition submission — CPU read-back/logging.** Verify the submission status and score, append the record to `results/kaggle_lb/submissions.csv`, and preserve `56132481` as the proven anchor. Do not fill a second final-submission slot without a materially distinct, validated candidate.
7. **Only after the B0 evidence exists:** B1 node-count stability audit, then B2 isolated mutual-best association test. Consider B3 neighborhood-flow only if held-out failures justify it. B4 pretrained division verifier and B5 DoG are lower priority; B6 synthetic-data training is post-competition work.

## GPU and CPU map for the experiment backlog

| Item | GPU requirement | Current state |
|---|---|---|
| Provenance review, structural report validation, synthetic tests | **No** | Validator and limited Gold OOF relinker/sidecar path implemented; trusted attestation, real OOF artifacts, complete paired evidence package, and notebook integration remain missing. |
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

- Do not look for or manually create `bidirectional_blend_union13_receipt.json` or the validator's input `evidence.json`; the latter must be emitted by a reviewed producer from real paired runs.
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
