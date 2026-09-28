# Stage and Submit 0.956 SOTA Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage, audit, validate, push, and submit the 0.956 SOTA candidate (incorporating sub-voxel coordinate regression `v1284`, neighborhood-flow motion prior, DeepCenter vetoes, and runtime hardening) to Kaggle.

**Architecture:** Create a clean Kaggle kernel bundle under `scripts/kaggle_kernels/biohub_0_956_subvoxel_flow_harmonic/` derived from the verified 0.956 public solution (`anvithpothula/biohub-0-953-lb-original`), reconcile all GPU and dataset metadata, run offline static AST and integrity audits, push to Kaggle with Tesla T4 GPU enabled, and monitor execution.

**Tech Stack:** Python 3.10+, PyTorch (on Kaggle T4), tracksdata, geff, scikit-learn, scipy, Kaggle CLI.

**Spec:** `docs/kaggle_deep_dive_2026-09-27.md` and `docs/endgame_next_steps_and_blockers.md`.

## Global Constraints

- Never commit secrets or credentials.
- Retain submission `56132481` (0.946) as the locked baseline anchor in the ledger.
- Enforce strict submission schema: `dataset,row_type,node_id,t,z,y,x,source_id,target_id`.
- Ensure offline validation passes 100% before remote push.
- Reconcile `enable_gpu: true` in `kernel-metadata.json` and `isGpuEnabled: true` in notebook JSON metadata.

## Review Focus

1. Discrepancy between sidecar `enable_gpu` and notebook `isGpuEnabled` causing CPU fallback.
2. Missing or detached dataset sources (`v1284-head-s075`, `biohub-temporal-unet3d-seed314159-v1`, `biohub-deepcenter-unet3d-center-prior-v1`, `biohub-tracking-support-pack-50ep-v1`).
3. Accidental inclusion of obsolete metric exploit cells (must be 0 exploit rows, pure `submission.csv`).
4. Configuration guard assertion failures in cell 1.
5. Wall-clock timeout or frame-cache OOM on hidden test data.

---

### Task 1: Stage Kernel Candidate Directory and Reconcile Metadata

**Files:**
- Create: `scripts/kaggle_kernels/biohub_0_956_subvoxel_flow_harmonic/kernel-metadata.json`
- Create: `scripts/kaggle_kernels/biohub_0_956_subvoxel_flow_harmonic/biohub-0-956-subvoxel-flow-harmonic.ipynb`
- Test: `tests/test_0956_kernel_staging.py`

- [ ] **Step 1: Write test for kernel staging and metadata consistency**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Stage kernel and author synchronized metadata**
- [ ] **Step 4: Run test to verify it passes**

---

### Task 2: Static AST, Schema, and Safety Audit

**Files:**
- Test: `tests/test_0956_kernel_staging.py`

- [ ] **Step 1: Add AST parse test for all notebook cells**
- [ ] **Step 2: Add configuration guard parameter verification test**
- [ ] **Step 3: Run pytest to ensure all checks pass**

---

### Task 3: Push Kernel to Kaggle and Verify Initialization

**Files:**
- Command: `kaggle kernels push -p scripts/kaggle_kernels/biohub_0_956_subvoxel_flow_harmonic`
- Command: `kaggle kernels status aleixlopez/biohub-0-956-subvoxel-flow-harmonic`

- [ ] **Step 1: Push kernel to Kaggle**
- [ ] **Step 2: Verify kernel status transitions from queued to running**
- [ ] **Step 3: Document submission attempt in tracking documentation**

---
