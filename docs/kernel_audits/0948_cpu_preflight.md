# 0.948 Kaggle Kernel CPU Preflight Audit

## Scope and timestamp

Initial static audit on **2026-09-25 14:17 CEST** from the local project workspace. A follow-up read-only Kaggle state check was performed on **2026-09-26 00:47 CEST**, with a fresh submission/kernel-list query on **2026-09-26 11:58 CEST**. No notebook cells were executed, no Kaggle kernel was pushed or run, and no competition submission was made during these checks.

Reviewed candidate:

- `scripts/kaggle_kernels/biohub_0_948_momentum_deepcenter_tta/biohub-0-948-momentum-deepcenter-tta.ipynb`
- `scripts/kaggle_kernels/biohub_0_948_momentum_deepcenter_tta/kernel-metadata.json`

Compared with:

- `scripts/kaggle_kernels/biohub_0_946_edge_feature_tta_tuned/biohub-0-946-edge-feature-tta-tuned.ipynb`
- `scripts/kaggle_kernels/biohub_0_946_edge_feature_tta_tuned/kernel-metadata.json`
- `results/kaggle_lb/submissions.csv`
- `docs/endgame_status_and_strategy.md`

Notebook cell numbers below are zero-based; line numbers within a cell are one-based. At audit start, the branch was `master...origin/master`; `?? .hermes/` was already present for the plan file created in the preceding turn and was left untouched.

## Candidate and baseline provenance

- The submission ledger's best scored entry is still submission `56132481`, candidate `biohub-0-946-edge-feature-tta-tuned`, public score `0.946`, submitted 2026-09-09. There is **no 0.948 score entry** in `results/kaggle_lb/submissions.csv`.
- Both staged notebooks contain 12 code cells. Before the approved provenance correction, only cell 5 differed. Comparing the current local files now finds source differences in cells **0, 1, 5, and 6**: cells 0, 1, and 6 contain the documented score/provenance/report corrections, while cell 5 contains the tracking behavior change.
- The only tracking-algorithm delta is cell 5's motion relinker: the 0.948 version maintains velocity per track, predicts positions using that velocity, and updates it as `0.6 * instant_vel + 0.4 * prior_vel`. The changes to cells 0, 1, and 6 update score labels, diagnostics, and the provenance/runtime report; they do not add a second detection/linking method.
- DeepCenter settings and code are present in both notebook versions: cell 0 requests DeepCenter and requires the veto; it sets gap and safe-division vetoes and threshold values. DeepCenter is therefore **not an additional algorithmic delta** between these two staged files.
- Before the approved correction, candidate cell 0 and cell 1 contained inconsistent historical score labels (`0.939` and `0.913`). The current local versions now explicitly say **unscored** and identify the 0.946 parent/EMA delta. Neither the old labels nor the staged name is score evidence; the ledger still contains no 0.948 score.

## Metadata and attached artifacts

- Both sidecar `kernel-metadata.json` files specify a private Python notebook for the competition, `enable_gpu: true`, no TPU/internet, a Tesla T4 machine, and the same three dataset sources: the DeepCenter checkpoint, the temporal U-Net seed 314159 checkpoint, and the 50-epoch support pack.
- Both embedded notebook metadata blocks instead contain `metadata.kaggle.isGpuEnabled: false`, while retaining `accelerator: nvidiaTeslaT4`. This conflicts with the sidecar's `enable_gpu: true`. Because neither notebook was pushed or run, this audit cannot establish which setting Kaggle will honor. Treat actual accelerator availability as unresolved until the Kaggle notebook configuration is verified in an authorized run.
- The code statically checks SHA-256 values for the primary model, secondary model, DeepCenter checkpoint, and materialized support-repository files (candidate cell 3). That verifies identity only if the cell executes successfully against the attached artifacts; those files were not available for local verification here.
- The candidate notebook writes a graph/submission audit report in cell 6 (source lines 123–178). It labels the artifact `candidate_unverified_quality`, names `bidirectional_blend_union13_receipt.json` as the required receipt, requires `promote=true`, sets `execute_push_submit` to `FORBIDDEN_UNTIL_REQUIRED_CONDITION`, and leaves `validated_receipt_sha256` as `None` (lines 125–130, 167–173). A search found no file named `bidirectional_blend_union13_receipt.json` in this workspace. This is an explicit quality-promotion blocker in the notebook's own recorded status; do not bypass it.

## Configuration and guard consistency

- Candidate cell 0 assigns 45 `BIOHUB_*` environment variables. Candidate cell 1 checks only six numeric and two text variables. The guard therefore does not lock most declared settings, including several motion, DeepCenter, gap-repair, division, and rescue settings against accidental drift.
- The sidecar kernel metadata says GPU enabled, but the embedded notebook metadata says GPU disabled; see the unresolved metadata conflict above.
- Candidate cell 0 sets `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT` to `0.15` (line 61). Cell 1's numeric guard expects `0.15` (line 12), the executable patch cell 4 independently rejects values other than `0.15` (source lines 171–180), and the resolved-state diagnostic in cell 11 reports the live environment value (source lines 20–22). These executable checks agree.
- The approved cell 1 correction removed the stale `0.200` reverse-time weight, `0.913` baseline, and incorrect harmonic-association attribution. It now labels the candidate unscored, names EMA relinking as the code delta, and prints the configured bidirectional weight. The independent executable `0.15` guard in cells 1/4 and resolved-state diagnostic in cell 11 remain the runtime checks.
- The approved cell 6 correction replaced the prior historical-looking experiment/parent/config values with the 0.948 candidate identity, the 0.946 parent, an explicit unscored state, and an allowlisted snapshot of declared `BIOHUB_*` environment settings. This records configured values, not proof of which values or weights actually loaded during a Kaggle run. The upstream diagnostic reference remains unverified.
- Current classification: prior score/attribution inconsistencies in cells 0/1 and stale cell 6 config have been corrected locally; the upstream source attribution and actual runtime/artifact state remain **unverified**; sidecar versus embedded GPU settings remain **conflicting**. No local execution was used to guess at their runtime effects.

## EMA motion relinking review

The only tracking-algorithm delta is in candidate cell 5's `motion_relink_edges` function (source lines 420–545). In particular:

- It keeps `track_velocity_um` keyed by node ID (line 455).
- It uses the previous tracked velocity, or derives one from the predecessor position, when projecting a source position (lines 471–476).
- It updates the target's velocity with `0.6 * instant_vel + 0.4 * prior_vel` when a prior estimate exists (lines 535–541).

This establishes that the staged code contains EMA-style motion projection. It does **not** establish that the change improves tracking quality: there is no local validation data, cached candidate graph, or OOF evaluation for this delta. Existing `tests/test_tracking.py` tests the separate generic `HungarianTracker`; they do not execute this notebook's `motion_relink_edges` implementation.

## DeepCenter veto review

Candidate cell 0 sets `BIOHUB_USE_DEEPCENTER_VETO=1`, `BIOHUB_REQUIRE_DEEPCENTER_VETO=1`, expected epoch 2, gap veto, safe-division veto, and thresholds 0.25. Candidate cell 11 (source lines 24–34) reports whether a DeepCenter detector was actually loaded and whether each veto is enabled; candidate cell 3 checks the checkpoint path and checksum. This is useful runtime observability, but the audit only confirms wiring and fail-closed intent from source. It cannot verify the mounted checkpoint, epoch, checksum, or behavior on actual microscopy volumes. These DeepCenter cells are identical in the staged 0.946 notebook.

## Submission validation review

Candidate cell 6 statically checks that the generated CSV:

- exists, is non-empty, and has the exact columns `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id` (source lines 10–23);
- has contiguous row IDs and exactly `node`/`edge` row types (lines 20–23);
- covers exactly the mounted competition test datasets (lines 25–32);
- has unique and complete frame-retention diagnostic coverage and internally consistent fallback decisions (lines 34–63);
- has nodes with non-negative time/coordinates, edge references to existing nodes in the next frame, maximum indegree 1, and maximum outdegree 2 (lines 65–103).

This is a meaningful runtime graph/schema guard, but its checks have not been exercised on a generated submission. It is not a substitute for running the official local CV gate before recommending a candidate.

## Checks run

- `python3 -m json.tool` on the 0.948 notebook and sidecar metadata: **PASS**.
- `python3 -m json.tool` on the 0.946 notebook and sidecar metadata: **PASS**.
- Read-only AST parse of all 12 code cells in the 0.948 notebook: **PASS**. No cell was executed.
- Initial audit test run on 2026-09-25, before the EMA characterization test was added: `PYTHONPATH=src python3 -m pytest -q` — **119 passed, 4 skipped in 7.13s**.
- Search for `*.zarr` under the project: **0 files found**.
- Search for `*.geff` under `outputs/`: **0 files found**.
- Search for the required `bidirectional_blend_union13_receipt.json` in the project: **0 files found**.
- Source search found no checked-in producer or validator for that receipt. Candidate cell 6 only writes the `required_receipt`, `required_condition`, `execute_push_submit`, and null `validated_receipt_sha256` fields into a report; it does not load a receipt or enforce the condition. The provenance test checks these source strings, not receipt contents or a real promotion result.
- `scripts/kaggle_kernels/gold_oof_runner/kernel.py` is not that producer: it runs a separate Gold-training OOF workflow (defaults to `temporal-pu-a`, epoch 10) and documents `promotion.json` among its outputs. Kaggle lists its lastRunTime as `2026-09-22T14:37:51.693000`, but `kaggle kernels files aleixlopez/biohub-gold-oof-runner` returned an empty list. Its output is neither available here nor evidence for the 0.948 EMA ablation.

## Read-only Kaggle follow-up — 2026-09-26

- `kaggle kernels list --user aleixlopez --search "biohub-0-948"` found the private kernel `aleixlopez/biohub-0-948-momentum-deepcenter-tta`, with `lastRunTime` reported as `2026-09-24T15:14:08.107000`. This list timestamp is not evidence that the run completed successfully or passed validation.
- `kaggle kernels pull ... --metadata` retrieved the remote sidecar: `enable_gpu: true`, `machine_shape: NvidiaTeslaT4`, and the three expected dataset sources. The remote notebook still embeds `metadata.kaggle.isGpuEnabled: false`. The T4's actual visibility and the run's attached artifacts remain unverified.
- Comparing the pulled remote notebook with the committed local notebook found source differences in cells 0, 1, and 6—the exact cells changed by commit `31ec907`. The remote Kaggle copy therefore does not contain the latest local provenance corrections.
- `kaggle kernels files` returned an empty list. `kaggle kernels output` and `kaggle kernels status` returned HTTP 404 from the session-output/status endpoints, so the prior run's log, submission, and receipt could not be inspected through this CLI session.
- A fresh `kaggle competitions submissions` query returned five records; none is the 0.948 candidate. The latest scored record remains submission `56132481` at `0.946`.
- The referenced source slug `raykkretzschmar/biohub-bidirectional-primary-union13-diagnostic-v1` could not be verified: Kaggle denied `kernels.get` for that exact reference. A separately named harmonic-association notebook was readable, but its pulled source did not emit or validate a `bidirectional_blend_union13_receipt.json`. Do not treat either as the required promotion receipt.
- A separate prior CV workspace contains raw training Zarr/GEFF data and a public-rule-control report, but targeted searches found no candidate OOF prediction arrays, model checkpoints, or promotion receipt there. Raw labels alone do not enable the paired 0.946-vs-0.948 comparison required by B1/B2.

## Read-only status refresh — 2026-09-26 11:58 CEST

- `kaggle competitions submissions biohub-cell-tracking-during-development --format json --page-size 20` returned five records. The latest scored record remains submission `56132481` at public score `0.946`; no `0.948` submission is present, and private scores were blank.
- `kaggle kernels list --user aleixlopez --search biohub-0-948 --page-size 10 --format json` still lists `aleixlopez/biohub-0-948-momentum-deepcenter-tta` with `lastRunTime` `2026-09-24T15:14:08.107000`. This listing is not evidence that inference completed or produced a usable output.
- Current endgame order, GPU classification, and user-approval boundaries are summarized in [`endgame_next_steps_and_blockers.md`](../endgame_next_steps_and_blockers.md). The immediate path is CPU-side provenance/evaluator work and locating genuine fold-excluded OOF artifacts; only then consider GPU inference.

## CPU-only EMA characterization follow-up — 2026-09-26

- Added `tests/test_kaggle_notebook_ema_relink.py`, which AST-extracts only `_position_um`, `motion_relink_edges`, and the relevant default settings. It does not execute notebook cells.
- The synthetic cases verify velocity-based target selection and the EMA update on small trajectories. Focused result: **2 passed**. Full suite: **125 passed, 4 skipped**. `git diff --check`: **PASS**.
- This is helper-level regression coverage only. It does not validate the Kaggle patch path, mounted model artifacts, OOF quality, leaderboard score, or promotion receipt.

## CPU paired-evidence validator follow-up — 2026-09-26 17:25 CEST

- Added `src/biohub_tracking/evaluation/promotion_gate.py` and `scripts/evaluate_promotion_gate.py`. The CPU-only checker verifies a hash-referenced A/B parent/candidate evidence manifest, fold train/evaluation separation, exact prediction/report coverage, paired weights/config/scorer, summary-vs-metric consistency, and the documented score/diagnostic gates.
- Focused result: `PYTHONPATH=src python3 -m pytest -q tests/test_evaluation_promotion_gate.py` — **18 passed**. Full suite: **143 passed, 4 skipped**. Every fixture is synthetic and written under pytest `tmp_path`; no real receipt, OOF predictions, or score were produced.
- The validator is still not the full evidence producer. `scripts/run_oof_checkpoint.py --motion-relink` applies a standalone port of the EMA rule to Gold OOF model output and records its method/config hash and per-prediction run counters, but it does not execute the exact staged Kaggle notebook pipeline or emit the validator's complete four-run contract. Hash binding checks artifact identity, not independent proof that the source executed. Real fold-excluded runs and notebook integration remain required.

## Local Gold OOF relinker follow-up — 2026-09-27 14:57 CEST

- Added `src/biohub_tracking/tracking/motion_relink.py` and the opt-in `--motion-relink` path in `scripts/run_oof_checkpoint.py`. It converts indexed `(t,z,y,x)` predictions to physical microns, applies the two-pass EMA Hungarian relinker before graph construction/ILP, and separates candidate outputs from the default path.
- The sidecar binds config/counters and a combined hash of the local adapter/helper to each GEFF file/directory hash. The edge tuple contract is checked at runtime; unsupported shapes fail instead of silently changing IDs.
- Focused tests: **18 passed**; full suite: **155 passed, 4 skipped**. CLI help, Python compilation, and whitespace checks passed. No real OOF inference or score was generated; no promotion receipt was written.
- Ruff and mypy could not run because the modules are unavailable. This Gold-model ablation is not the exact `0.946` Kaggle pipeline and the records do not yet form the validator's complete four-run evidence package.

## Findings and severity

1. **High — EMA-aware candidate promotion remains unproven.** Cell 6 writes `candidate_unverified_quality`, the receipt filename, `promote=true`, `FORBIDDEN_UNTIL_REQUIRED_CONDITION`, and a null receipt hash into a report; it does not load the file or block execution. The CPU validator exists, and the local OOF runner now has an optional Gold-model EMA relinker with per-prediction records, but that path does not emit the validator's complete four-run evidence contract and does not execute the exact staged candidate. The Kaggle notebook is not connected to the receipt. This remains a local internal gate, not an official Kaggle-issued file/rule. Treat the candidate as unpromoted until genuine paired fold-excluded evidence is produced and validated. Do not fabricate a receipt or relabel another experiment's promotion output.
2. **High — Accelerator metadata conflicts.** The remote sidecar requests GPU/T4, but the embedded notebook metadata says GPU disabled; the same conflict is present in the local candidate. Although Kaggle reports a prior `lastRunTime`, its runtime status/logs are not retrievable here, so actual accelerator visibility remains unresolved. Verify it before spending quota; do not infer a working T4 from metadata alone.
3. **High — Upstream source and remote runtime provenance remain unverified.** The local candidate's score/parent/config-report corrections are now in place, but the referenced upstream diagnostic source could not be verified, and the remote Kaggle copy predates those local corrections. The current allowlisted configuration block records declared environment values, not proof of actual runtime settings or loaded checkpoint hashes. Confirm the source attribution and runtime artifacts before syncing or running.
4. **Resolved locally — Stale score/run diagnostics.** The prior `0.200` reverse-time weight, `0.913` baseline label, and incorrect association attribution were replaced by explicit unscored/EMA diagnostics. This corrects local text only; it is not validation evidence, and the old remote copy remains stale.
5. **Medium — Config guard is narrow.** Only 8 of 45 cell-0 environment assignments are checked in cell 1. Several important declared options can drift without triggering that guard. The dynamic cell 11 manifest helps show resolved state but is diagnostic output, not a comprehensive expected-value assertion.
6. **Low / validation gap — Basic synthetic helper coverage added; end-to-end behavior remains unvalidated.** `tests/test_kaggle_notebook_ema_relink.py` exercises the notebook EMA helper on synthetic trajectories, but does not cover Kaggle-only imports, dynamic patch anchors, mounted artifacts, or score quality.
7. **No score conclusion.** The absence of matching validation inputs and candidate OOF predictions prevents a paired candidate CV run or CPU-only post-processing comparison. No claim of improvement over 0.946 is supported by this audit.

## Pending runtime and local-validation checks

### GPU/Kaggle checks — only after evidence gates and separate user approval

1. After the receipt gate is legitimately satisfied, verify in Kaggle UI/runtime that the T4 is actually enabled and the three expected dataset versions are attached.
2. Check that the exact upstream support-script patch anchors still match once, and that primary, secondary, and DeepCenter artifact manifests/checksums resolve as expected.
3. Confirm the runtime manifest reports dual-seed weights found, bidirectional fusion at the intended value, and DeepCenter loaded with the expected checkpoint/epoch.
4. Push/run only after the required receipt is obtained and reviewed; inspect output schema/topology checks and the runtime/error log. Do not use the remote pre-correction notebook as the candidate evidence.

### CPU-capable paired evaluation — once genuine OOF artifacts exist

Once valid fold-excluded predictions and labels are available, run the paired official-metric local evaluation and inspect per-dataset diagnostics. This is CPU-capable and does not itself require Kaggle authorization; generating missing neural predictions may require GPU.

These checks require Kaggle execution and/or local competition validation data; they were not performed here.

## Readiness verdict and next action

**BLOCKED ON FOLD-DISJOINT EVIDENCE AND RUNTIME READINESS.** The CPU structural validator and an optional Gold OOF `--motion-relink` path now exist; the latter writes method/config hashes and per-prediction execution records bound to GEFF hashes. No real fold-excluded predictions, paired scores, complete four-run evidence package, or notebook receipt integration exist, and the local Gold-model ablation does not validate the exact `0.946` Kaggle pipeline. The sidecar and embedded accelerator flags still disagree, and upstream/remote runtime provenance remains unverified. Fresh neural predictions are likely to require GPU. Verify an actual T4 and consider any Kaggle run only after the evidence path is valid and with separate user approval. See [`endgame_next_steps_and_blockers.md`](../endgame_next_steps_and_blockers.md). This audit does not authorize or perform a kernel push or competition submission.

## Independent code-review hardening follow-up — 2026-09-27 15:59 CEST

- The OOF runner now rejects candidate and split dataset identifiers that are not single safe path components before using them in data/output paths.
- EMA reuse sidecars now bind the exact checkpoint SHA256 and a combined hash of local relinker code plus all Python sources in the configured upstream `scripts/` and `src/` trees. These are integrity bindings, not a trusted attestation that the declared code executed.
- The paired-evidence validator now recomputes adjusted edge Jaccard from TP/FP/FN and `node_count_ratio` using the documented `alpha=0.1` adjustment; inconsistent CSV metrics are rejected.
- Schema-v1 assessment reports always retain `promote=false` and a trusted-provenance blocker, even when metric thresholds pass. A signed/otherwise independently trusted runtime and scorer attestation is not yet implemented.
- The Gold OOF relinker remains an isolated one-to-one ablation and does not reproduce the candidate notebook's subsequent division-repair stage. It is not validation of the exact Kaggle pipeline.
- Synthetic focused regression tests: **42 passed**; full suite: **161 passed, 4 skipped**. CLI help, compilation, and `git diff --check` passed. No OOF inference, candidate score, Kaggle run, or submission was produced by this hardening work.
