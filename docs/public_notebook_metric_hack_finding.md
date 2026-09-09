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

In the run we downloaded, this added **184 of 265,328 total submission
rows** as fabricated out-of-bounds nodes/edges, structured as synthetic
parent→children "division" forks per connected component.

**Correction (re-checked before resubmission):** an earlier version of this
document mis-reported this as "132,846 of 265,328 rows (roughly half the
file)". That figure was wrong — it came from naively filtering `t < 0`,
which also matches the file's *normal, legitimate* edge rows (edges don't
carry coordinates, so `t/z/y/x` are set to a harmless `-1` sentinel for
every edge row by design). Re-filtering on the actual exploit signature
(`z <= -9000` or `t <= -900`, matching the real injected coordinates like
`t=-1000, z=-10000`) shows only **184 rows** are the synthetic
hub/fork/divider nodes and their edges — a small, easily-isolated block at
the end of the pipeline, not half the file. This does not change the
conclusion (it is still a deliberate metric exploit and was correctly not
submitted), only the estimate of its scale. The notebook family names make
the intent explicit:
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

## Confirmed via official competition discussion: exploit is known and patched

Checked the competition's discussion forum directly
(`kaggle competitions topics list` / `topic-messages`). This exact exploit
is a known, publicly acknowledged issue, already patched by the organizers:

- Discussion topic **"Division Metric exploit and patch"**
  (id `727154`, 2026-07-18): a participant (`thibautgoldsborough`) publicly
  disclosed the exact mechanism — synthetic "hub" nodes and fork chains far
  outside the volume (e.g. `z=y=x=-10000`) merge all predicted tracks into
  one connected component and register as fake divisions, which the
  *pre-patch* division-Jaccard matcher (based on weak connectivity) scored
  as true positives, adding close to the full `+0.1` division-Jaccard
  weight to the final score with zero real tracking benefit.
- The competition organizers confirmed: *"a few of you might be aware that
  an exploit of our metrics was found, specifically in the division Jaccard
  score. We have made a patch and will re-score all submissions... The patch
  is already public at
  https://github.com/royerlab/kaggle-cell-tracking-competition."*
- Discussion topic **"COMPLETED: Rescore Underway"** (id `728324`,
  2026-07-22/23): organizers announced the metric patch went live and **all
  submissions were rescored**, with an explicit expectation that "most
  submissions will have at least a minor drop in score." A participant
  confirmed their own "hacking score" dropped after the rescore.
- The patch requires divisions to be genuine **strongly-connected**
  parent→two-daughter structures matched within the 7 µm radius, so the
  far-away synthetic forks now register as false positives instead of true
  positives — the exploit no longer works against the live scorer.

**Conclusion:** this is not a gray area — organizers explicitly identified
it as a metric bug, patched it, and rescored the whole leaderboard before
now. Any of the six downloaded public notebooks that still contain the
`augment_dataset`/`MAX_COMPONENTS`/`FORKS` injection code are stale
artifacts from before the July 2026 patch; running that code today against
the live (patched) scorer would just add false-positive division rows and
very likely *hurt* the score slightly rather than help it, in addition to
being explicitly against the spirit of the competition. There is no
indication in the rules or discussion that intentionally exploiting a
metric bug is condoned — quite the opposite, it was treated as a bug to fix.

This also answers the original question: the `~0.96` scores currently
visible on the public leaderboard are post-patch/post-rescore scores, so
they should **not** reflect this specific exploit anymore. The gap between
our 0.251 baseline and the leaderboard's ~0.96 entries is therefore a
genuine methodology gap (learned 3D detection + transformer edge scoring +
ILP graph solving + heavy post-processing vs. our simple rule-based
detector/linker), not a sign that we're missing a scoring trick we should
copy.

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

## Rules check: is reusing a public notebook itself allowed?

Before submitting anything derived from this notebook, we checked whether
this competition's rules restrict reuse of other participants' public code.
Kaggle's Foundational Competition Rules (which apply here, category
"Research", $60,000 prize pool) explicitly **require** that any
competition-related code shared publicly be shared on Kaggle.com (forums or
Notebooks), and this competition follows the standard Kaggle norm of
encouraging public notebook sharing for transparency/collaboration. There is
no rule against building on or forking a public notebook — that is the
intended use of "Public" notebooks. The exploit itself was never a
rules-sanctioned technique; it was a metric implementation bug, publicly
disclosed and patched (see above). So: **forking/reusing the legitimate
modeling pipeline is fine; submitting the leftover exploit-injection code is
not** (and, post-patch, would no longer even help the score).

## Resolution: cleaned submission

We removed the `augment_dataset`/`MAX_COMPONENTS`/`FORKS` cell entirely from
our fork. It turned out the notebook's own pipeline already writes a fully
clean, exploit-free intermediate file (`submission_clean.csv`, produced by
the legitimate "HYBRID POSTPROCESSING" cell) *before* the exploit cell runs.
We replaced the exploit cell with a no-op that promotes
`submission_clean.csv` to the required `submission.csv` output, with an
in-notebook assertion (`0 exploit rows`) guarding against regressions, and
added an attribution/modification note to the notebook.

Kernel `aleixlopez/biohub-v6-ultra-best-fork` v3 ran successfully on GPU and
produced a submission with:
- 259,564 rows (132,482 nodes / 127,082 edges), **0 exploit rows**.
- No NaNs, no duplicate node ids, no dangling edge references, all
  coordinates in-bounds (`t` 0-99, `z` 0-63 across all 4 test datasets).
- Byte-identical to `submission_clean.csv`.

Submitted via `kaggle competitions submit -c biohub-cell-tracking-during-development -k aleixlopez/biohub-v6-ultra-best-fork -v 3 -f submission.csv`
(submission id `56120358`). See `results/kaggle_lb/submissions.csv` for the
scored result.

## Scored result: 0.883, and why it doesn't match the ~0.96 leaderboard

Submission `56120358` scored a **public score of 0.883** — a genuine,
exploit-free result. This is legitimate, but noticeably below the ~0.95-0.97
entries currently visible near the top of the public leaderboard. Root
cause, confirmed from the kernel v3 run log and the attached support-pack
dataset's own manifest (not from anything we changed):

- The kernel log shows:
  ```
  Found 1 checkpoint splits: ['split_0']
  loaded weight: .../unet_transformer/split_0/edge_predictor_best.pth
  Ensemble mode: False
  ```
  `biohub-v6-ultra-best`'s main advantage over the earlier notebooks in the
  same family is *multi-checkpoint ensemble inference*, but the public
  dataset it depends on, `pilkwang/biohub-tracking-support-pack-50ep-v1`
  (checked via `kaggle datasets files` and its `ARTIFACT_MANIFEST.json`),
  only ships **one** checkpoint split (`split_0`, 3 weight files, ~33 MB
  total). With only one split available, the notebook's own ensembling logic
  has nothing to ensemble over and silently falls back to single-model
  inference. This is an inherent limitation of the *publicly shared* weights
  the notebook depends on, not a bug introduced by removing the exploit cell
  (the promoted `submission.csv` is byte-identical to the notebook's own
  `submission_clean.csv`, so nothing about the model/inference path was
  touched).
- The support pack is explicitly a "50-epoch" snapshot (per its dataset
  name/README); the author's actual best private checkpoints (more epochs
  and/or more cross-validation splits) were evidently never published
  alongside the notebook.
- **Correction on the "~0.966" reference score**: checking the *current*
  public leaderboard directly (`kaggle competitions leaderboard --download`)
  shows the notebook's author, `amanatar` (team "Aman Atar", team id
  `16458655`), sits at **rank 527 with a live public score of 0.942**, not
  0.966. The 0.966 score on the leaderboard belongs to a different,
  unrelated competitor. It's likely an earlier research note conflated the
  author's score with someone else's, or with a stale/cached score shown in
  the notebook's own output before the July division-metric patch and
  leaderboard rescore (which does not retroactively update a notebook's
  displayed "Best Score" widget).

**Conclusion:** 0.883 is the correct, legitimate score for running the
*exact* publicly shared code against the *exact* publicly shared weights,
after removing only the exploit-injection cell. Closing the remaining gap to
~0.94-0.97 would require either the author's un-shared, more complete
checkpoint set, or training additional checkpoint splits ourselves to
restore genuine ensembling — not a code or rules issue.

**Confirmation (2026-09-09, per user report):** the original public notebook
`amanatar/biohub-v6-ultra-best` itself displays a "Best Score" of **0.882**
on its Kaggle notebook page — essentially identical (within rounding/
run-to-run noise) to our reproduction's **0.883**. This is strong external
confirmation that:
- Our exploit-cell removal changed nothing about the model's real
  performance; running the *exact* same publicly shared code/weights as-is
  reproduces the *exact* same score the original author's own notebook page
  reports.
- The earlier "~0.966" figure was never this notebook's actual displayed
  score. It was likely a mix-up with a different competitor's leaderboard
  entry, and `amanatar`'s better live submissions (up to 0.942, per the
  competition leaderboard) come from a separate, more complete/private
  checkpoint set never bundled into the public `pilkwang` support-pack
  dataset that this specific published notebook depends on.
- There is now no unresolved "gap to explain" for *this notebook* — 0.882
  (original) vs. 0.883 (our clean fork) is the correct, consistent score for
  the publicly available code + weights combination.

**Clarification: this is not "private competition data", it is incomplete
shared model weights.** The gap between this notebook's ~0.88 score and
higher leaderboard scores is *not* evidence that top scorers are using
private ground-truth/test data or anything against the rules — the
competition data itself (train/test `.zarr` volumes) is identical for every
participant and openly accessible inside a Kaggle Notebook. What's missing
is simply that `amanatar` (and presumably other top scorers building on the
same family of notebooks) trained more model checkpoints locally than they
chose to publish in the `pilkwang/biohub-tracking-support-pack-50ep-v1`
dataset — the pack is explicitly named/versioned as a reduced "50-epoch"
snapshot with a single checkpoint split. Holding back one's best-trained
weights while still sharing the inference/training *code* publicly is
normal and permitted Kaggle practice (competitors are not obligated to
publish their best checkpoints, only their code if they choose to share at
all). So: no rule violation, no hidden competition data — just an inherent,
expected reproducibility ceiling when working from partially-shared
artifacts, closable only by training our own additional checkpoints.

## Status

- Private research kernel `aleixlopez/biohub-v6-ultra-best-fork` v3 (cleaned,
  exploit cell removed) is our submitted, legitimate public-notebook-derived
  entry — scored **public score 0.883** (submission `56120358`), see above.
- Our other submitted baseline is `public-rule-control-est-budget`
  (kernel `aleixlopez/biohub-public-rule-control-est-budget-submit` v2),
  which scored a genuine **public score 0.251** with no synthetic nodes —
  see `docs/kaggle_submission_kernel.md` and `results/kaggle_lb/submissions.csv`.
- `biohub-v6-ultra-best-fork` (0.883) is our current best legitimate
  submission, superseding the 0.251 rule-based baseline.
