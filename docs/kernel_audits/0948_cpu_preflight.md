# 0.948 Kaggle Kernel CPU Preflight Audit

## Scope and timestamp

Audited on **2026-09-25 14:17 CEST** from the local project workspace. This was a static, CPU-only review; no notebook cells were executed, no Kaggle kernel was pushed or run, and no competition submission was made.

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
- Both staged notebooks contain 12 code cells. A source comparison found that **only code cell 5 differs** between the 0.948 and 0.946 notebook files; cells 0–4 and 6–11 are identical.
- The sole code delta is the motion relinking implementation in cell 5: the 0.948 version maintains velocity per track, predicts positions using that velocity, and updates it as `0.6 * instant_vel + 0.4 * prior_vel`. This supports describing EMA velocity projection as the actual staged code change from the 0.946 notebook.
- DeepCenter settings and code are present in both notebook versions: cell 0 requests DeepCenter and requires the veto; it sets gap and safe-division vetoes and threshold values. DeepCenter is therefore **not an additional code delta** between these two staged files, even though the endgame document describes it as an improvement “over 0.946.”
- The 0.948 notebook's cell 0 labels the score axis `public 0.939 base + holdout-selected post-process configuration` (lines 12–13). Its cell 1 says the baseline is public LB `0.913` (line 41). These are not consistent with each other or with the 0.946 score recorded in the ledger. The staged name/title is a candidate label, not score evidence.

## Metadata and attached artifacts

- Both sidecar `kernel-metadata.json` files specify a private Python notebook for the competition, `enable_gpu: true`, no TPU/internet, a Tesla T4 machine, and the same three dataset sources: the DeepCenter checkpoint, the temporal U-Net seed 314159 checkpoint, and the 50-epoch support pack.
- Both embedded notebook metadata blocks instead contain `metadata.kaggle.isGpuEnabled: false`, while retaining `accelerator: nvidiaTeslaT4`. This conflicts with the sidecar's `enable_gpu: true`. Because neither notebook was pushed or run, this audit cannot establish which setting Kaggle will honor. Treat actual accelerator availability as unresolved until the Kaggle notebook configuration is verified in an authorized run.
- The code statically checks SHA-256 values for the primary model, secondary model, DeepCenter checkpoint, and materialized support-repository files (candidate cell 3). That verifies identity only if the cell executes successfully against the attached artifacts; those files were not available for local verification here.
- The candidate notebook writes a graph/submission audit report in cell 6 (source lines 123–178). It labels the artifact `candidate_unverified_quality`, names `bidirectional_blend_union13_receipt.json` as the required receipt, requires `promote=true`, sets `execute_push_submit` to `FORBIDDEN_UNTIL_REQUIRED_CONDITION`, and leaves `validated_receipt_sha256` as `None` (lines 125–130, 167–173). A search found no file named `bidirectional_blend_union13_receipt.json` in this workspace. This is an explicit quality-promotion blocker in the notebook's own recorded status; do not bypass it.

## Configuration and guard consistency

- Candidate cell 0 assigns 45 `BIOHUB_*` environment variables. Candidate cell 1 checks only six numeric and two text variables. The guard therefore does not lock most declared settings, including several motion, DeepCenter, gap-repair, division, and rescue settings against accidental drift.
- The sidecar kernel metadata says GPU enabled, but the embedded notebook metadata says GPU disabled; see the unresolved metadata conflict above.
- Candidate cell 0 sets `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT` to `0.15` (line 61). Cell 1's numeric guard expects `0.15` (line 12), the executable patch cell 4 independently rejects values other than `0.15` (source lines 171–180), and the resolved-state diagnostic in cell 11 reports the live environment value (source lines 20–22). These executable checks agree.
- In contrast, candidate cell 1 prints `Reverse-time association weight: 0.200` (line 43), and describes a fixed-90 dual-seed baseline at `0.913` (line 41); those lines disagree with the active `0.15` weight and the candidate's other score-axis label. Cell 1 also describes “harmonic mutual-support association fusion” as the single model-level change (line 42), although the only source-code delta versus the 0.946 notebook is the EMA relinker in cell 5.
- Candidate cell 6's `configuration` provenance block (source lines 135–149) records several values unlike cell 0's current environment assignments—for example detector threshold `0.96875` versus `0.96`, ILP disappearance weight `1.5` versus `2`, and gap-close distance `5.8` versus `5.0`. The report also calls its experiment `harmonic_bidirectional_association_v1` and names a separate parent experiment (lines 125–130). This may be a historical experiment receipt rather than the active runtime config, but its placement inside the current candidate's audit report is ambiguous. Resolve the provenance before treating it as a candidate run record.
- Classification: the weight and score strings in cell 1 are **diagnostic/provenance-only mismatches** because the active patch guard and final runtime manifest use the declared `0.15`; the cell 6 configuration block is a **provenance mismatch requiring confirmation**; whether the embedded GPU flag changes runtime is **unresolved**. No local execution was used to guess at their runtime effects.

## EMA motion relinking review

The only code delta is in candidate cell 5's `motion_relink_edges` function (source lines 420–545). In particular:

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
- `PYTHONPATH=src python3 -m pytest -q`: **119 passed, 4 skipped in 7.13s**.
- Search for `*.zarr` under the project: **0 files found**.
- Search for `*.geff` under `outputs/`: **0 files found**.
- Search for the required `bidirectional_blend_union13_receipt.json` in the project: **0 files found**.

## Findings and severity

1. **High — Candidate quality gate is unsatisfied.** The notebook's own cell 6 report declares `candidate_unverified_quality`, requires a receipt with `promote=true`, says push/submit is forbidden until that condition, and has no validated receipt hash. The required receipt is absent locally. Do not push or submit this candidate until its required quality evidence is obtained and reviewed.
2. **High — Accelerator metadata conflicts.** The sidecar requests GPU, but embedded notebook metadata says GPU disabled. Both staged versions contain the conflict. Determine which setting Kaggle uses before spending quota; do not assume the T4 is active solely from the sidecar or accelerator label.
3. **High — Current candidate provenance is ambiguous.** The cell 6 report contains historical-looking configuration values that disagree with cell 0 and names a separate parent experiment. The notebook's score labels (`0.939`, `0.913`) do not match each other, and neither establishes a 0.948 result. Confirm whether this report belongs in this notebook and identify the authoritative candidate configuration.
4. **Medium — Stale run diagnostics.** The printed `0.200` reverse-time weight conflicts with the actual, guarded `0.15` value; the cell 1 description of the model-level change also does not describe the only source-code change versus 0.946. This can mislead future run interpretation even if it does not alter the active inference configuration.
5. **Medium — Config guard is narrow.** Only 8 of 45 cell-0 environment assignments are checked in cell 1. Several important declared options can drift without triggering that guard. The dynamic cell 11 manifest helps show resolved state but is diagnostic output, not a comprehensive expected-value assertion.
6. **Low / validation gap — EMA code lacks notebook-specific synthetic regression tests.** The generic tracker tests do not call this notebook helper. Static AST validity does not cover candidate behaviors, Kaggle-only imports, or string-patch anchor matching.
7. **No score conclusion.** The absence of local `.zarr` inputs, CV pack, and `.geff` predictions prevents a local paired CV run or CPU-only post-processing sweep. No claim of improvement over 0.946 is supported by this audit.

## GPU-only checks still required

Only after the provenance and readiness blockers above are resolved, and after the user authorizes the external action:

1. Verify in Kaggle UI/runtime that the T4 is actually enabled and the three expected dataset versions are attached.
2. Check that the exact upstream support-script patch anchors still match once, and that primary, secondary, and DeepCenter artifact manifests/checksums resolve as expected.
3. Confirm the runtime manifest reports dual-seed weights found, bidirectional fusion at the intended value, and DeepCenter loaded with the expected checkpoint/epoch.
4. Run the notebook only after any explicit quality gate is met; inspect its output schema/topology checks and runtime/error log.
5. Run the mandatory local evaluation/paired-OFF validation if predictions and validation data become available; inspect per-dataset diagnostics before recommending a score change.

These checks require Kaggle execution and/or local competition validation data; they were not performed here.

## Readiness verdict and next action

**BLOCKED ON PROVENANCE CONFIRMATION.** The sidecar and embedded accelerator flags disagree; the score/provenance labels are inconsistent; and the notebook's own quality-promotion record explicitly remains unverified with its required receipt absent. The next safe action is to confirm the intended experiment/configuration and satisfy the receipt gate, then separately decide whether to repair the notebook and request a GPU run. This audit does not authorize or perform a kernel push or competition submission.
