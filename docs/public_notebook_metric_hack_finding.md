# Public notebook "metric hack" finding — do not submit

## What happened

Following up on "why did we get 0.251 instead of the ~0.966 some public
notebooks show", we forked one of the downloaded top public notebooks
(`amanatar/biohub-v6-ultra-best`, family: TemporalUNet3D + node transformer +
tracksdata ILP) as a private kernel (`aleixlopez/biohub-v6-ultra-best-fork`),
ran it against the real competition test data on a GPU, and inspected the
`submission.csv` it produced **before submitting**.

## Finding: this is not a genuine solution, it is a scorer exploit

Every one of the six public notebooks we downloaded for research
(`research/kaggle_notebooks/*/`) ends with an `augment_dataset` /
`MAX_COMPONENTS` / `FORKS` step that injects large numbers of **synthetic
nodes and edges with impossible coordinates** (`t=-1000`, `z=-10000`,
`z=-10001`, `x=-10000`, etc.) directly into the file that gets submitted to
the competition:

```python
MAX_COMPONENTS = 5000   # (varies per notebook, e.g. 1400 in biohub-solution)
FORKS = 15              # (varies per notebook, e.g. 5 in biohub-solution)

def row(dataset, row_type, node_id=-1, t=-1, z=-1, y=-1, x=-1, ...): ...

def augment_dataset(group):
    ...
    new_nodes = [row(dataset, "node", hub_id, -1000, -10000, -10000, -10000)]
    ...  # more nodes with t=-999..-985, z=-10000/-10001, x=-10000
```

In the run we downloaded, this added **132,846 of 265,328 total submission
rows** (roughly half the file) as fabricated out-of-bounds nodes/edges,
structured as synthetic parent→children "division" forks per connected
component. The notebook family names make the intent explicit:
`biohub-metric-hack-last-call`, `improved-metric-hack-last-call`,
`metric-hack-last-call`. This is designed to exploit the official grader's
local-window division-matching / connected-component bookkeeping (per
`kaggle-cell-tracking-competition/metrics.md`) to inflate the reported score,
not to submit genuine cell detections/tracks.

## Decision

**We did not submit this file.** Submitting a scorer exploit to gain an
artificially inflated leaderboard score, even one copied from a public
notebook, is a competition-integrity violation, not a legitimate modeling
improvement, and we will not do this regardless of how many public
leaderboard entries appear to be using the same trick.

This also answers the original question: the `~0.96` scores visible on the
public leaderboard from teams whose notebooks match this family are very
likely partly or fully inflated by this exploit, not a reflection of
real tracking quality. They are not a fair target to chase by copying the
same trick.

## What we keep from this notebook family (legitimately)

The *non-exploit* parts of these notebooks are legitimate prior art and
remain useful reference material (already summarized in
`research/kaggle_baseline_review.md`):
- TemporalUNet3D + node-transformer architecture for learned detection/edge
  scoring.
- `tracksdata` ILP-based graph solving with edge/appearance/disappearance/
  division weights.
- TTA (flip/rotate) for detection robustness.
- Physical-distance-based candidate edge gating (10-12 µm before final 7 µm
  metric matching).

Any future work adapting ideas from these notebooks must strip the
`augment_dataset` / `MAX_COMPONENTS` / `FORKS` synthetic-node injection
entirely before any submission is made.

## Status

- Private research kernel `aleixlopez/biohub-v6-ultra-best-fork` exists on
  Kaggle (not submitted, harmless) for our own inspection; it should not be
  used as-is for a real submission.
- Our real, submitted baseline remains `public-rule-control-est-budget`
  (kernel `aleixlopez/biohub-public-rule-control-est-budget-submit` v2),
  which scored a genuine **public score 0.251** with no synthetic nodes —
  see `docs/kaggle_submission_kernel.md` and `results/kaggle_lb/submissions.csv`.
