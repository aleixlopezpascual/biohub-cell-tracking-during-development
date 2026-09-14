# Gold training implementation notes

The repository now supplies the local and cloud-facing controls needed to run
the gated campaign without making PyTorch, tracksdata, or GEFF core imports.

## Implemented components

- `biohub_tracking.training.targets`: anisotropic Gaussian center targets and
  positive-unlabeled loss weights.
- `biohub_tracking.training.checkpoints`: atomic full-state checkpoints and
  validated immutable run manifests.
- `biohub_tracking.training.session`: checkpoint cadence, official-score early
  stopping state, and a fresh per-job wall-clock deadline.
- `biohub_tracking.training.gates`: official-score checkpoint selection,
  learning-curve promotion, correlation-aware model shortlisting, and
  ensemble/runtime gates.
- `biohub_tracking.evaluation.oracle`: current-detection oracle linking and
  detection-versus-linking headroom reports.
- `biohub_tracking.baselines.royerlab.linking`: physical motion features,
  deterministic hard-negative mining, OOF temperature calibration, and
  bidirectional probability fusion.
- `biohub_tracking.baselines.royerlab.peaks`: confidence-based node-budget
  selection, OOF image-statistic count calibration, and local intensity
  centroid refinement.

The official model implementation remains an optional Kaggle-side dependency.
`scripts/prepare_gold_training.py` validates its location, and the campaign
runner dynamically reuses the real `train_unet_transformer.py` APIs; the local
fallback `scripts/train.py` is not
misrepresented as a competitive tracker.

`scripts/run_oof_checkpoint.py` calls the upstream predictor and official GEFF
scorer directly for exactly the held-out datasets in the immutable split. It
does not use the upstream predictor's global output directory, so unrelated
predictions cannot be cleared or accidentally included in a score. The
one-command `scripts/run_gold_stage.py` joins preparation, resumable training,
OOF inference, official scoring, and append-only score/error registration.

## Cloud execution boundary

The unmodified official trainer selects its internal checkpoint using a proxy
metric and does not preserve complete optimizer state. The prepared bootstrap
therefore calls `scripts/continue_royerlab_training.py` from epoch zero. That
runner reuses the upstream model, lazy dataset, training epoch, and validation
functions, but saves the exact model, optimizer, scheduler, RNG, session, and
manifest state at five-epoch intervals and at the requested gate. It also
emits an adjacent plain model state for official OOF inference.

The continuation runner deliberately handles exactly one configured target
epoch per invocation. It logs the official trainer's loss/accuracy/recall only
as diagnostics; it never promotes on them. Run OOF inference and the official
graph scorer, pass the resulting two-prefix scores through
`scripts/evaluate_training_gates.py`, then resume to the next stage only if the
gate passes. A legacy upstream plain checkpoint may be migrated explicitly,
but its missing optimizer history cannot be reconstructed.

Detection-driven batches can contain a sample with no peaks in one frame. The
pinned support-pack transformer otherwise passes an all-true padding mask to
PyTorch multi-head attention, which produces non-finite values. The
continuation runner guards that case with one zero-valued dummy key while
leaving the real-node mask unchanged, so the dummy region remains excluded
from edge loss. The finite-gradient check remains a final safety stop; it is
not used to tolerate systematic invalid batches.

Resume also normalizes saved PyTorch generator states back to CPU byte tensors.
This is required because loading the rest of a checkpoint directly onto CUDA
would otherwise move the CPU RNG state to CUDA, which `torch.set_rng_state`
rejects before the first resumed batch.

Pinned metric functions also perform sibling imports lazily during evaluation.
Official scoring therefore runs inside the upstream import context instead of
only using that context while loading `evaluate.py`. Calibrated OOF runs can
override the sigmoid detection threshold without changing the immutable
training configuration; they write to threshold-specific directories and log
the effective threshold with the checkpoint identity. A bounded detector-only
probe evaluates several thresholds from one set of TTA logits, selecting by
estimated node-count ratio with annotated-node recall as a tie-breaker.

AMP remains disabled until a measured Kaggle benchmark can add GradScaler to
the upstream per-batch loop without changing numerical behavior. Cosine LR
scheduling and supported TemporalUNet3D gradient checkpointing are applied by
the native exact-epoch campaign path.
