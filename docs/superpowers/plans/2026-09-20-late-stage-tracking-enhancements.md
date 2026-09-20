# Late-Stage Tracking Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and unit-test spatiotemporal EMA Velocity projected tracking and DeepCenter image-space veto gating to push local CV tracking scores to peak efficiency.

**Architecture:** We will extend the existing Hungarian frame-to-frame tracker to optionally propagate and smooth trajectory velocity vectors using an Exponential Moving Average (EMA). We will also expose a decoupled, compliant `RefinementHook` factory that checks proposed post-processing gap midpoints against a spatial center-prior probability heatmap, rejecting those that fall below a threshold.

**Tech Stack:** Python 3.10+, NumPy, SciPy, Pytest

**Spec:** `docs/kaggle_late_stage_research_insights.md`

## Global Constraints
* No external heavy package imports (e.g. `torch`, `zarr`, `skimage`) are permitted outside local/lazy scopes.
* Coordinates must stay strictly in physical microns once detections enter tracking/metrics logic.
* All changes must be fully unit-tested on synthetic datasets before declaring completion.

## Review Focus
* Ensure `TrackerConfig` handles validation errors for invalid `ema_velocity_alpha` values.
* Ensure trajectory crossovers under fast-moving conditions are correctly resolved by velocity projection.
* Ensure veto hooks correctly distinguish boundary probability limits (`[0, 1]`) and handle edge cases gracefully.

---

### Task 1: EMA Velocity Projected Tracking

**Files:**
- Modify: `src/biohub_tracking/tracking/hungarian.py`
- Test: `tests/test_tracking.py`

**Interfaces:**
* Consumes: Existing `TrackerConfig`, `HungarianTracker.link_frames`, and `HungarianTracker.track`.
* Produces: Config parameters `use_ema_velocity_projection` and `ema_velocity_alpha`, and updated tracking logic that propagates and projects velocities.

- [ ] **Step 1: Write the failing tests**
  Add the following test cases in `tests/test_tracking.py`:
  ```python
  def test_tracker_config_validation_for_ema() -> None:
      with pytest.raises(ValueError, match="alpha"):
          TrackerConfig(use_ema_velocity_projection=True, ema_velocity_alpha=-0.1)
      with pytest.raises(ValueError, match="alpha"):
          TrackerConfig(use_ema_velocity_projection=True, ema_velocity_alpha=1.5)

  def test_velocity_projection_resolves_crossover_occlusions() -> None:
      # Two trajectories crossing paths at Frame 1:
      # Trajectory 1: moving right (+2.0 um/frame) from (0,0,0) -> (0,0,2) -> expects (0,0,4)
      # Trajectory 2: moving left (-2.0 um/frame) from (0,0,5) -> (0,0,3) -> expects (0,0,1)
      graph = TrackingGraph()
      # Frame 0
      graph.add_node(_det("s1", 0, 0, 0, 0))
      graph.add_node(_det("s2", 0, 0, 0, 5))
      # Frame 1
      graph.add_node(_det("m1", 1, 0, 0, 2))
      graph.add_node(_det("m2", 1, 0, 0, 3))
      # Frame 2
      graph.add_node(_det("e1", 2, 0, 0, 4))
      graph.add_node(_det("e2", 2, 0, 0, 1))

      # 1. Run tracking with projection disabled (causes crossover error matching m1->e2 and m2->e1)
      tracker_off = HungarianTracker(TrackerConfig(max_link_distance_um=5.0, use_ema_velocity_projection=False))
      tracked_off = tracker_off.track(graph)
      assert ("m1", "e2") in tracked_off.edges
      assert ("m2", "e1") in tracked_off.edges

      # 2. Run tracking with projection enabled (correctly matches m1->e1 and m2->e2)
      tracker_on = HungarianTracker(TrackerConfig(max_link_distance_um=5.0, use_ema_velocity_projection=True, ema_velocity_alpha=1.0))
      tracked_on = tracker_on.track(graph)
      assert ("m1", "e1") in tracked_on.edges
      assert ("m2", "e2") in tracked_on.edges
  ```

- [ ] **Step 2: Run tests to verify they fail**
  Run: `PYTHONPATH=src python3 -m pytest tests/test_tracking.py -k "ema or crossover" -v`
  Expected: FAIL (errors because fields don't exist and assertions fail)

- [ ] **Step 3: Modify `TrackerConfig` and `HungarianTracker`**
  Modify `src/biohub_tracking/tracking/hungarian.py` to add `use_ema_velocity_projection` and `ema_velocity_alpha` to config, and update `link_frames` and `track`.

  In `TrackerConfig`:
  ```python
  @dataclass(frozen=True)
  class TrackerConfig:
      # ... existing fields
      use_ema_velocity_projection: bool = False
      ema_velocity_alpha: float = 0.5
  ```
  Add the validation check inside `__post_init__`:
  ```python
          if not 0.0 < self.ema_velocity_alpha <= 1.0:
              raise ValueError("ema_velocity_alpha must be in (0, 1]")
  ```

  In `link_frames`:
  ```python
      def link_frames(
          self,
          sources: list[Detection],
          targets: list[Detection],
          projected_positions: dict[NodeId, tuple[float, float, float]] | None = None,
      ) -> list[tuple[NodeId, NodeId]]:
          # ... existing checks ...
          cfg = self.config
          src_pos = np.array(
              [
                  projected_positions.get(s.id, s.position)
                  if projected_positions is not None
                  else s.position
                  for s in sources
              ],
              dtype=float,
          )
          # ... rest of link_frames unchanged ...
  ```

  In `track`:
  ```python
      def track(self, graph: TrackingGraph) -> TrackingGraph:
          """Link every consecutive frame pair of a node-only graph.

          Any pre-existing edges in ``graph`` are discarded; this method is
          meant to (re)build the edge set from scratch given only detections.
          """
          by_frame = graph.nodes_by_frame()
          result = TrackingGraph(nodes=dict(graph.nodes))
          frames = sorted(by_frame)
          
          # Track active velocity vectors per trajectory: last_node_id -> velocity_vector_um
          active_tracks_velocity: dict[NodeId, tuple[float, float, float]] = {}
          cfg = self.config

          for t0, t1 in zip(frames, frames[1:]):
              sources = by_frame[t0]
              targets = by_frame[t1]
              dt = t1 - t0

              # Compute projected positions for active trajectories
              projected_positions: dict[NodeId, tuple[float, float, float]] = {}
              if cfg.use_ema_velocity_projection:
                  for s in sources:
                      if s.id in active_tracks_velocity:
                          v = active_tracks_velocity[s.id]
                          projected_positions[s.id] = (
                              s.z + v[0] * dt,
                              s.y + v[1] * dt,
                              s.x + v[2] * dt,
                          )

              # Link frames using projections
              links = self.link_frames(sources, targets, projected_positions=projected_positions if cfg.use_ema_velocity_projection else None)
              
              # Map targets and sources by ID for fast lookup
              targets_dict = {t.id: t for t in targets}
              sources_dict = {s.id: s for s in sources}

              # Propagate and update EMA velocities for successfully matched edges
              new_velocities: dict[NodeId, tuple[float, float, float]] = {}
              for s_id, t_id in links:
                  s_node = sources_dict[s_id]
                  t_node = targets_dict[t_id]
                  
                  v_current = (
                      (t_node.z - s_node.z) / dt,
                      (t_node.y - s_node.y) / dt,
                      (t_node.x - s_node.x) / dt,
                  )
                  
                  if s_id in active_tracks_velocity:
                      v_prev = active_tracks_velocity[s_id]
                      alpha = cfg.ema_velocity_alpha
                      v_new = (
                          alpha * v_current[0] + (1 - alpha) * v_prev[0],
                          alpha * v_current[1] + (1 - alpha) * v_prev[1],
                          alpha * v_current[2] + (1 - alpha) * v_prev[2],
                      )
                  else:
                      v_new = v_current
                      
                  new_velocities[t_id] = v_new
                  result.add_edge(s_id, t_id)
              
              # Update active velocities for the next frame
              active_tracks_velocity = new_velocities
              
          return result
  ```

- [ ] **Step 4: Run tests to verify they pass**
  Run: `PYTHONPATH=src python3 -m pytest tests/test_tracking.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add src/biohub_tracking/tracking/hungarian.py tests/test_tracking.py
  git commit -m "feat(tracking): implement EMA Velocity Projected tracking"
  ```

---

### Task 2: DeepCenter Heatmap Veto Gating

**Files:**
- Modify: `src/biohub_tracking/baselines/royerlab/postprocess.py`
- Test: `tests/test_royerlab_baseline.py`

**Interfaces:**
* Consumes: Compliance with `RefinementHook` callback signature.
* Produces: compliant hook creator `make_deepcenter_veto_hook`.

- [ ] **Step 1: Write the failing test**
  Add the following test case inside `tests/test_royerlab_baseline.py`:
  ```python
  def test_deepcenter_veto_hook_rejects_low_probability_interpolations() -> None:
      from biohub_tracking.baselines.royerlab.postprocess import (
          make_deepcenter_veto_hook,
          GapClosingConfig,
          close_one_frame_gaps,
      )
      graph = TrackingGraph()
      graph.add_node(_detection("before", 0, 0))
      graph.add_node(_detection("after", 2, 4))
      
      # Heatmap provider that returns 0.1 at (0.0, 0.0, 2.0)
      provider_low = lambda z, y, x: 0.1
      veto_hook_low = make_deepcenter_veto_hook(provider_low, threshold=0.25)
      
      # Run with veto hook: gap should be rejected since 0.1 < 0.25
      rejected = close_one_frame_gaps(
          graph, GapClosingConfig(max_distance_um=5), refinement_hook=veto_hook_low
      )
      assert len(rejected.nodes) == 2
      
      # Heatmap provider that returns 0.9 at (0.0, 0.0, 2.0)
      provider_high = lambda z, y, x: 0.9
      veto_hook_high = make_deepcenter_veto_hook(provider_high, threshold=0.25)
      
      # Run with veto hook: gap should be accepted since 0.9 >= 0.25
      accepted = close_one_frame_gaps(
          graph, GapClosingConfig(max_distance_um=5), refinement_hook=veto_hook_high
      )
      assert len(accepted.nodes) == 3
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `PYTHONPATH=src python3 -m pytest tests/test_royerlab_baseline.py -k "veto" -v`
  Expected: FAIL (ImportError because `make_deepcenter_veto_hook` is not defined)

- [ ] **Step 3: Implement `make_deepcenter_veto_hook`**
  Add `make_deepcenter_veto_hook` inside `src/biohub_tracking/baselines/royerlab/postprocess.py`:
  ```python
  def make_deepcenter_veto_hook(
      heatmap_provider: Callable[[float, float, float], float],
      threshold: float = 0.25,
  ) -> RefinementHook:
      """Create a RefinementHook that vetoes (rejects) proposals using a spatial prior heatmap.

      If the probability returned by ``heatmap_provider`` at the proposed midpoint
      is below ``threshold``, the proposal is rejected (returns ``None``).
      """
      if not 0 <= threshold <= 1:
          raise ValueError("threshold must be in [0, 1]")

      def veto_hook(
          _source: Detection,
          _target: Detection,
          midpoint: tuple[float, float, float],
      ) -> tuple[float, float, float] | None:
          prob = heatmap_provider(*midpoint)
          if prob < threshold:
              return None
          return midpoint

      return veto_hook
  ```

- [ ] **Step 4: Run tests to verify they pass**
  Run: `PYTHONPATH=src python3 -m pytest tests/test_royerlab_baseline.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add src/biohub_tracking/baselines/royerlab/postprocess.py tests/test_royerlab_baseline.py
  git commit -m "feat(postprocess): implement DeepCenter Heatmap veto hook"
  ```
