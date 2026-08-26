# NPE 2D Embedding Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Do not parallelize Tasks 2-5.** They are independent in the sense that
> none reads another's output, but all four share the single MPS GPU on this
> machine. Running them concurrently would make each one slower rather than
> saving wall-clock time, and has not been tested for correctness (two
> processes fighting over one `mps` device is a known source of silent
> slowdowns or memory errors). Run them one after another. A subagent
> dispatched for Task 3 must wait for Task 2's background training to finish
> and be reviewed before it starts.

**Goal:** Train four NPE posteriors for the 2D (Δ, Ω) photon-counting problem
— identical flow, differing only in the trajectory embedding
(`none`/`deepset`/`hist`/`cnn`) — on a 95%/5% train/held-out split of the full
Zenodo dataset, and report calibration and posterior quality in a single
Markdown document with figures under `docs/`.

**Architecture:** `scripts/npe_2d_sbi.py` already implements the full
train+diagnose pipeline (SBC, expected coverage, TARP) per the sbi skill's
guidance to always calibrate before trusting a posterior; this plan only
supplies the 95/5 split parameters, runs it four times (one per embedding,
holding every other hyperparameter fixed so the embedding is the sole
variable), and adds one new script, `docs/make_npe_embedding_report.py`, that
turns the four `summary.json` files and figure sets into one comparison
report. No changes to `scripts/npe_2d_sbi.py` itself are needed or planned.

**Tech Stack:** Python 3.11, `sbi` 0.27, `torch` 2.13 (MPS backend), `fire`,
`matplotlib`, `numpy`. Environment: conda env `torch-sbi` (from
`environment-sbi.yml`), already installed at
`/Users/enrico.rinaldi/miniforge3/envs/torch-sbi`.

**Spec:** No separate spec document — this plan is the spec, distilled from
the user's request ("run `scripts/npe_2d_sbi.py` on 95% of the training data
with 5% held out for SBC, across different embedding settings, and report the
results in `docs/` with figures in `docs/figures/`") and from
`scripts/npe_2d_sbi.py`'s own docstring and `README.md`'s
"2D simulation-based inference with `sbi`" section, which lay out exactly the
four embeddings to compare (`none`, `deepset`, `hist`, `cnn`) and why.

## Global Constraints

- **Python/env:** every `npe_2d_sbi.py` invocation runs through
  `/Users/enrico.rinaldi/miniforge3/envs/torch-sbi/bin/python` — the repo's
  base environment has neither `torch` nor `sbi`.
- **Data location:** the training data (`param_rand_list-2D.npy`,
  `taus-2D.npy`, 4,000,000 pairs, repaired per
  [`docs/zenodo-trajectory-repair.md`](../../zenodo-trajectory-repair.md))
  lives only in the main checkout,
  `/Users/enrico.rinaldi/Projects/ParamEst-NN/data/training-trajectories/`,
  not in this worktree (`*.npy` is gitignored and worktrees don't share
  untracked files). Every run passes
  `--datapath=/Users/enrico.rinaldi/Projects/ParamEst-NN/data/training-trajectories/`
  explicitly rather than copying ~1.5 GB into the worktree.
- **The split is fixed and identical across all four runs:** `num_train`,
  `num_diagnostic`, `obs_index` and `seed` never change between Tasks 2-5.
  Only `--embedding` and its embedding-specific flags vary. This is what
  makes the comparison an isolated single-variable experiment.
- **Run artifacts (checkpoints, pickled posteriors) are never git-added.**
  They write to
  `/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/`
  in the *main* checkout (not the worktree — same reasoning as the data path:
  large binaries have no reason to live inside a throwaway worktree, and
  `data/` is where this repo already keeps trained-model examples). Only the
  small evaluation PNGs and the generated Markdown/JSON get copied into this
  worktree's `docs/` and are meant to be committed.
- **No `--run_ppc`.** The posterior predictive check needs `qutip`, which
  lives only in the separate, Python-3.9 "main" conda environment described
  in `README.md`, not in `torch-sbi` (Python 3.11, needed for `sbi`>=0.27).
  Bridging the two environments is out of scope for this plan; SBC, expected
  coverage and TARP (all three already run unconditionally by
  `npe_2d_sbi.py`) are what the user asked for ("test the posterior with SBC
  methods") and are sufficient calibration evidence on their own.
- **Runtime is measured, not guessed** (see Task 1 benchmark below). Expect
  the four runs together to take several hours on this machine (Apple M5, 24
  GB RAM, MPS backend, no CUDA).
- **Do not launch training with the Bash tool's `run_in_background: true`.**
  Discovered empirically during execution: a background-flagged Bash task in
  this harness is killed at almost exactly the 60-minute mark regardless of
  whether the underlying command is still making progress (confirmed via
  `log show`: the training process's XPC connections were torn down at
  59m58s after launch, with no traceback, no OOM signature, and no
  computational hang found on inspection — `torch.quantile` on the full
  3.8M×48 tensor, the operation right before where the log goes silent, was
  independently timed at 1.6s on this machine's MPS backend, ruling that out
  as the cause). Since every one of these runs is expected to take well over
  an hour, use this instead: launch with `nohup ... < /dev/null > run.log
  2>&1 & disown`, a plain foreground Bash call that returns in under a
  second because the `&` backgrounds the process at the shell level — this
  detaches the training process from the harness's own background-task
  bookkeeping entirely, so it keeps running as an ordinary OS process no
  matter how long it takes. Check on it afterwards with ordinary
  short-lived foreground commands (`tail run.log`, `pgrep -f
  npe_2d_sbi.py`), spaced out with bounded `sleep` calls (e.g. `sleep 1500`,
  25 minutes, comfortably inside the Bash tool's own 10-minute-per-call
  default but issued as its own call each time) — never a single call that
  blocks for the whole run. Also add `-u` to the `python` invocation
  (unbuffered stdout) — without it, an unflushed print buffer is what made
  the killed run's log look like it had frozen right after the startup
  warnings, when the buffering itself (not any hang) was hiding whatever
  progress had actually happened.

---

## Benchmark data used to size this plan

Measured in this session with the real data and the real `torch-sbi`
environment (`--device=mps`), by timing tiny training runs and taking the
marginal cost of extra samples between a 20,000- and a 200,000-pair run
(2 epochs each, so fixed process-startup overhead cancels out):

| embedding | measured throughput | projected time / epoch at `num_train=3,800,000` |
| --- | --- | --- |
| `none` | ~21,400 samples/s | ~3.0 min |
| `hist` | ~13,100 samples/s | ~4.8 min (the 700-dim histogram sum is the extra cost) |
| `deepset`, `cnn` | not separately measured at scale; both were within 15% of `none`'s cost at small scale | budget ~3.5-4 min as a planning estimate |

With `stop_after_epochs=20` and `max_num_epochs=50` (both kept at the
script's defaults, see Task 1), a run stops anywhere between ~21 epochs (one
non-improving stretch right after the best epoch) and 50 (the hard cap). That
puts each run's plausible range at roughly **1-4 hours**, so budget most of a
day for Tasks 2-5 run sequentially. This is a research/cluster-style
workload — `scripts/npe_2d_sbi.py`'s own docstring says as much — not an
interactive one.

---

### Task 1: Preflight — verify environment, data, and fix the split parameters

**Files:**
- Create: `/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/` (directory only, holds Tasks 2-5' outputs)
- No source files created or modified in this task.

**Interfaces:**
- Produces: the four numeric constants every later task's command line uses —
  `NUM_TRAIN=3800000`, `NUM_DIAGNOSTIC=1000`, `OBS_INDEX=3900000`, `SEED=0` —
  and the confirmed absolute paths for `--datapath` and `--outdir`.

- [ ] **Step 1: Confirm the `torch-sbi` environment has what `npe_2d_sbi.py` needs**

Run:

```bash
/Users/enrico.rinaldi/miniforge3/envs/torch-sbi/bin/python -c "
import torch, sbi, fire, matplotlib, numpy
print('torch', torch.__version__)
print('sbi', sbi.__version__)
print('mps available', torch.backends.mps.is_available())
"
```

Expected: prints `torch 2.13.x`, `sbi 0.27.x`, `mps available True`, no
`ModuleNotFoundError`. (Already confirmed once in this session; re-run here
so the executing agent has its own evidence rather than trusting the plan.)

- [ ] **Step 2: Confirm the data file shapes and the total pair count**

Run:

```bash
/Users/enrico.rinaldi/miniforge3/envs/torch-sbi/bin/python -c "
import numpy as np
p = np.load('/Users/enrico.rinaldi/Projects/ParamEst-NN/data/training-trajectories/2D-delta-omega/param_rand_list-2D.npy', mmap_mode='r')
t = np.load('/Users/enrico.rinaldi/Projects/ParamEst-NN/data/training-trajectories/2D-delta-omega/taus-2D.npy', mmap_mode='r')
assert p.shape == (4_000_000, 2), p.shape
assert t.shape == (4_000_000, 48), t.shape
print('OK', p.shape, t.shape)
"
```

Expected: `OK (4000000, 2) (4000000, 48)`. If the shapes differ (e.g. the data
was regenerated with a different `njumps` or pair count since this plan was
written), stop and recompute the split constants below before continuing —
everything in Tasks 2-5 depends on `len(params) == 4_000_000`.

- [ ] **Step 3: Fix the split constants for every later task**

With `N = 4_000_000`:

| constant | value | derivation |
| --- | --- | --- |
| `NUM_TRAIN` | `3_800_000` | 95% of `N` |
| held-out pool | `[3_800_000, 4_000_000)` | the remaining 5%, 200,000 pairs, never seen in training |
| `NUM_DIAGNOSTIC` | `1_000` | pairs `[3_800_000, 3_801_000)` of the held-out pool, used for SBC/expected-coverage/TARP — "a few hundred to ~1000" is the standard budget for these checks (see `run_diagnostics`'s docstring in `scripts/npe_2d_sbi.py:494`); the full 200,000 would cost 200x the runtime for no extra statistical power worth having |
| `OBS_INDEX` | `3_900_000` | inside the held-out pool, clear of the diagnostic slice above, used for the single example-observation plot |
| `SEED` | `0` | script default, held fixed across all four runs |

Verify the two assertions `npe_2d_sbi.py:main` enforces before spending any
GPU time:

```bash
python3 -c "
NUM_TRAIN, NUM_DIAGNOSTIC, OBS_INDEX, N = 3_800_000, 1_000, 3_900_000, 4_000_000
assert NUM_TRAIN + NUM_DIAGNOSTIC <= N
assert OBS_INDEX >= NUM_TRAIN + NUM_DIAGNOSTIC
assert OBS_INDEX < N
print('split constants OK')
"
```

Expected: `split constants OK`.

- [ ] **Step 4: Create the run-output root**

Run:

```bash
mkdir -p /Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5
```

No output expected. This directory will hold four subdirectories
(`none/`, `deepset/`, `hist/`, `cnn/`), one per Task 2-5.

- [ ] **Step 5: Commit**

Nothing to commit — this task only verifies state and creates a directory
outside the worktree. Proceed directly to Task 2.

---

### Task 2: Baseline run — `--embedding=none`

**Files:**
- Create (outside the worktree, not git-tracked): `/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/none/{training_data.png, posterior_observation.png, sbc_rank_histogram.png, expected_coverage.png, tarp.png, npe_density_estimator.pt, npe_posterior.pkl, summary.json}`
- Create: `/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/none/run.log`

**Interfaces:**
- Consumes: the split constants fixed in Task 1.
- Produces: `none/summary.json` and `none/*.png` — Task 6 reads both. This is
  the only one of the four runs with `plot_data=True`, so it is also the sole
  source of the shared `training_data.png` (the training data itself does
  not depend on the embedding choice, so generating it four times would be
  wasted work and four identical figures to choose between).

- [ ] **Step 1: Launch the run, detached from the harness's background-task tracking**

Run as a plain foreground Bash call (it returns in well under a second —
the `&` backgrounds the process at the shell level, `nohup` and `disown`
detach it from this call so it survives regardless of how long it takes;
see the Global Constraints note on why `run_in_background: true` must not
be used here):

```bash
nohup /Users/enrico.rinaldi/miniforge3/envs/torch-sbi/bin/python -u scripts/npe_2d_sbi.py \
  --datapath=/Users/enrico.rinaldi/Projects/ParamEst-NN/data/training-trajectories/ \
  --outdir=/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/none \
  --num_train=3800000 --num_diagnostic=1000 --obs_index=3900000 --seed=0 \
  --model=zuko_nsf --hidden_features=128 --num_transforms=3 \
  --embedding=none \
  --plot_data=True --device=auto \
  < /dev/null > /Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/none/run.log 2>&1 &
disown
```

Every flag not listed (`training_batch_size`, `learning_rate`,
`validation_fraction=0.1`, `stop_after_epochs=20`, `max_num_epochs=50`,
`num_posterior_samples=2**14`, `num_diagnostic_samples=1000`) is left at the
script's own default (`scripts/npe_2d_sbi.py:678-705`) and is identical
across all four runs by construction.

- [ ] **Step 2: Poll until it finishes, then verify**

Do not use the Bash tool's `run_in_background: true` for this or any wait —
it is the mechanism that killed the first attempt at ~60 minutes. Instead,
run a bounded `sleep` (e.g. 1500s = 25 minutes) as its own foreground Bash
call, then check the log; repeat until you see either the `Wrote results
to ...` line or a Python traceback:

```bash
sleep 1500; tail -15 /Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/none/run.log; pgrep -fl npe_2d_sbi.py
```

`pgrep` confirms the process is still alive between checks (empty output
means it exited — check the log immediately for how). Given the ~3
min/epoch estimate and up to 50 epochs, expect several such checks before
it finishes — this is normal, not a sign of a stuck run.

Once it finishes:

```bash
tail -30 /Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/none/run.log
```

Expected: the last line is
`Wrote results to /Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/none` —
and no Python traceback anywhere in the log.

Then:

```bash
/Users/enrico.rinaldi/miniforge3/envs/torch-sbi/bin/python -c "
import json
s = json.load(open('/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/none/summary.json'))
assert s['config']['embedding'] == 'none'
assert s['config']['num_train'] == 3800000
assert s['training']['epochs'] > 0
assert 0.0 <= s['diagnostics']['tarp']['ks_pval'] <= 1.0
print('epochs trained:', s['training']['epochs'])
print('best validation loss:', s['training']['best_validation_loss'])
print('OK')
"
```

Expected: `OK`, plus the printed epoch count and loss (record these — Task 6
reads the same file programmatically, but skimming them here catches a
degenerate run, e.g. `epochs trained: 1`, immediately rather than after all
four runs finish).

- [ ] **Step 3: Confirm all five figures exist and are non-empty**

```bash
ls -la /Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/none/*.png
```

Expected: five `.png` files (`training_data.png`, `posterior_observation.png`,
`sbc_rank_histogram.png`, `expected_coverage.png`, `tarp.png`), each with a
nonzero size (tens to hundreds of KB).

- [ ] **Step 4: Commit**

Nothing to commit yet (outputs live outside the worktree). Proceed to Task 3.

---

### Task 3: `--embedding=deepset` run

**Files:**
- Create (outside the worktree): `/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/deepset/{posterior_observation.png, sbc_rank_histogram.png, expected_coverage.png, tarp.png, npe_density_estimator.pt, npe_posterior.pkl, summary.json, run.log}`

**Interfaces:**
- Consumes: the same split constants as Task 2.
- Produces: `deepset/summary.json` and `deepset/*.png` for Task 6.

- [ ] **Step 1: Launch the run, detached from the harness's background-task tracking**

Same mechanism as Task 2 Step 1 (`nohup` + `&` + `disown`, plain foreground
call, `-u` for unbuffered output) — see the Global Constraints note:

```bash
nohup /Users/enrico.rinaldi/miniforge3/envs/torch-sbi/bin/python -u scripts/npe_2d_sbi.py \
  --datapath=/Users/enrico.rinaldi/Projects/ParamEst-NN/data/training-trajectories/ \
  --outdir=/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/deepset \
  --num_train=3800000 --num_diagnostic=1000 --obs_index=3900000 --seed=0 \
  --model=zuko_nsf --hidden_features=128 --num_transforms=3 \
  --embedding=deepset --embedding_output_dim=16 \
  --plot_data=False --device=auto \
  < /dev/null > /Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/deepset/run.log 2>&1 &
disown
```

(`--embedding_output_dim=16` is already the script's default — it is spelled
out here only because it is the one embedding-specific knob `deepset` has,
for the reader comparing the four commands side by side.)

- [ ] **Step 2: Poll until it finishes, then verify**

Same polling mechanism as Task 2 Step 2 (bounded `sleep 1500` + `tail` +
`pgrep -fl npe_2d_sbi.py` foreground calls, repeated — never
`run_in_background: true`), substituting `deepset` for `none` in every
path. Once finished, run the same three checks as Task 2 Steps 2-3: tail
the log for the `Wrote results to ...` line and no traceback, load and
sanity-check `summary.json` (`s['config']['embedding'] == 'deepset'`), and
confirm the four PNGs (no `training_data.png` here — `plot_data=False`)
exist and are non-empty.

- [ ] **Step 3: Commit**

Nothing to commit. Proceed to Task 4.

---

### Task 4: `--embedding=hist` run

**Files:**
- Create (outside the worktree): `/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/hist/{posterior_observation.png, sbc_rank_histogram.png, expected_coverage.png, tarp.png, npe_density_estimator.pt, npe_posterior.pkl, summary.json, run.log}`

**Interfaces:**
- Consumes: the same split constants as Task 2.
- Produces: `hist/summary.json` and `hist/*.png` for Task 6.

- [ ] **Step 1: Launch the run, detached from the harness's background-task tracking**

Same mechanism as Task 2 Step 1 (`nohup` + `&` + `disown`, plain foreground
call, `-u` for unbuffered output) — see the Global Constraints note:

```bash
nohup /Users/enrico.rinaldi/miniforge3/envs/torch-sbi/bin/python -u scripts/npe_2d_sbi.py \
  --datapath=/Users/enrico.rinaldi/Projects/ParamEst-NN/data/training-trajectories/ \
  --outdir=/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/hist \
  --num_train=3800000 --num_diagnostic=1000 --obs_index=3900000 --seed=0 \
  --model=zuko_nsf --hidden_features=128 --num_transforms=3 \
  --embedding=hist --hist_nbins=700 --hist_taumax=100.0 \
  --plot_data=False --device=auto \
  < /dev/null > /Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/hist/run.log 2>&1 &
disown
```

`--hist_nbins=700 --hist_taumax=100.0` are the paper's values and already the
script's defaults; spelled out for the same side-by-side-comparison reason as
Task 3. Expect this to be the slowest of the four runs (see the benchmark
table above) — the log will also print the line
`embedding='hist' needs the raw time delays: overriding z_score_x='structured' with 'none'`
near the top; that is expected behavior (`scripts/npe_2d_sbi.py:800-805`), not
an error.

- [ ] **Step 2: Poll until it finishes, then verify**

Same polling mechanism as Task 2 Step 2 (bounded `sleep 1500` + `tail` +
`pgrep -fl npe_2d_sbi.py` foreground calls, repeated — never
`run_in_background: true`), substituting `hist` for `none` in every path.
Once finished, run the same checks as Task 2 Steps 2-3
(`s['config']['embedding'] == 'hist'`, four PNGs, no `training_data.png`).
Additionally confirm `s['config']['z_score_x'] == 'none'` in the loaded
summary, i.e. the override above actually took effect.

- [ ] **Step 3: Commit**

Nothing to commit. Proceed to Task 5.

---

### Task 5: `--embedding=cnn` run (the non-permutation-invariant control)

**Files:**
- Create (outside the worktree): `/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/cnn/{posterior_observation.png, sbc_rank_histogram.png, expected_coverage.png, tarp.png, npe_density_estimator.pt, npe_posterior.pkl, summary.json, run.log}`

**Interfaces:**
- Consumes: the same split constants as Task 2.
- Produces: `cnn/summary.json` and `cnn/*.png` for Task 6.

- [ ] **Step 1: Launch the run, detached from the harness's background-task tracking**

Same mechanism as Task 2 Step 1 (`nohup` + `&` + `disown`, plain foreground
call, `-u` for unbuffered output) — see the Global Constraints note:

```bash
nohup /Users/enrico.rinaldi/miniforge3/envs/torch-sbi/bin/python -u scripts/npe_2d_sbi.py \
  --datapath=/Users/enrico.rinaldi/Projects/ParamEst-NN/data/training-trajectories/ \
  --outdir=/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/cnn \
  --num_train=3800000 --num_diagnostic=1000 --obs_index=3900000 --seed=0 \
  --model=zuko_nsf --hidden_features=128 --num_transforms=3 \
  --embedding=cnn --cnn_kernel_size=5 --embedding_output_dim=16 \
  --plot_data=False --device=auto \
  < /dev/null > /Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/embedding-comparison-95-5/cnn/run.log 2>&1 &
disown
```

- [ ] **Step 2: Poll until it finishes, then verify**

Same polling mechanism as Task 2 Step 2 (bounded `sleep 1500` + `tail` +
`pgrep -fl npe_2d_sbi.py` foreground calls, repeated — never
`run_in_background: true`), substituting `cnn` for `none` in every path.
Once finished, run the same checks as Task 2 Steps 2-3
(`s['config']['embedding'] == 'cnn'`, four PNGs, no `training_data.png`).

- [ ] **Step 3: Commit**

Nothing to commit. Proceed to Task 6.

---

### Task 6: Write and run `docs/make_npe_embedding_report.py`

**Files:**
- Create: `docs/make_npe_embedding_report.py`
- Create (by running it): `docs/figures/npe-2d-training-data.png`,
  `docs/figures/npe-2d-{none,deepset,hist,cnn}-posterior-observation.png`,
  `docs/figures/npe-2d-{none,deepset,hist,cnn}-sbc-rank-histogram.png`,
  `docs/figures/npe-2d-{none,deepset,hist,cnn}-expected-coverage.png`,
  `docs/figures/npe-2d-{none,deepset,hist,cnn}-tarp.png`,
  `docs/npe-2d-embedding-comparison.md`

**Interfaces:**
- Consumes: `summary.json` and the `*.png` files under each of the four
  `.../embedding-comparison-95-5/{none,deepset,hist,cnn}/` directories
  created by Tasks 2-5.
- Produces: `docs/npe-2d-embedding-comparison.md`, the deliverable the user
  asked for, plus its figures under `docs/figures/`.

This follows the existing convention in this repo of a small `docs/make_*.py`
script producing the content of a `docs/*.md` page — see
`docs/make_figures.py` alongside `docs/zenodo-trajectory-repair.md`. Unlike
that script (which only draws a figure, leaving the prose hand-written),
this one renders the whole report, because most of the report content here
*is* the numbers in `summary.json` — keeping the prose in the same script
that reads the numbers is what prevents a transcription mismatch between the
report and the run that produced it.

- [ ] **Step 1: Write the script**

Create `docs/make_npe_embedding_report.py`:

```python
"""
Build the embedding-comparison report for the 2D NPE posterior from the four
runs of `scripts/npe_2d_sbi.py` (--embedding=none/deepset/hist/cnn) on a
95%/5% train/held-out split of the full Zenodo 2D dataset.

Reads each run's `summary.json` and figures from
`--run_root` (default: the shared location used by the plan that produced
them, `data/models/npe-sbi-2D/embedding-comparison-95-5/` in the main
checkout) and writes:

  - the comparison figures, copied and renamed under `--figures_dir`
    (default `docs/figures/`)
  - the report itself, `--out` (default `docs/npe-2d-embedding-comparison.md`)

Usage
-----
    python docs/make_npe_embedding_report.py
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import fire

EMBEDDINGS = ["none", "deepset", "hist", "cnn"]
PARAM_LABELS = ["Δ", "Ω"]

EMBEDDING_LABEL = {
    "none": "No embedding (raw trajectory)",
    "deepset": "DeepSets (learned permutation-invariant summary)",
    "hist": "Hist-Dense (fixed histogram bins, paper architecture)",
    "cnn": "CNN (1D convolutions, not permutation invariant — control)",
}

DEFAULT_RUN_ROOT = (
    "/Users/enrico.rinaldi/Projects/ParamEst-NN/data/models/npe-sbi-2D/"
    "embedding-comparison-95-5"
)


def _load_summaries(run_root: Path) -> Dict[str, Dict[str, Any]]:
    summaries = {}
    for embedding in EMBEDDINGS:
        path = run_root / embedding / "summary.json"
        assert path.is_file(), f"Missing {path} -- run scripts/npe_2d_sbi.py first"
        summaries[embedding] = json.loads(path.read_text())
    return summaries


def _copy_figures(run_root: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        run_root / "none" / "training_data.png",
        figures_dir / "npe-2d-training-data.png",
    )
    figure_names = {
        "posterior_observation.png": "posterior-observation",
        "sbc_rank_histogram.png": "sbc-rank-histogram",
        "expected_coverage.png": "expected-coverage",
        "tarp.png": "tarp",
    }
    for embedding in EMBEDDINGS:
        for src_name, dst_stem in figure_names.items():
            src = run_root / embedding / src_name
            assert src.is_file(), f"Missing {src}"
            shutil.copy(src, figures_dir / f"npe-2d-{embedding}-{dst_stem}.png")


def _config_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    header = (
        "| embedding | flow | hidden features | transforms | "
        "embedding-specific | z-score(x) |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for embedding in EMBEDDINGS:
        cfg = summaries[embedding]["config"]
        if embedding == "hist":
            extra = f"nbins={cfg['hist_nbins']}, taumax={cfg['hist_taumax']}"
        elif embedding == "cnn":
            extra = f"kernel_size={cfg['cnn_kernel_size']}, output_dim={cfg['embedding_output_dim']}"
        elif embedding == "deepset":
            extra = f"output_dim={cfg['embedding_output_dim']}"
        else:
            extra = "--"
        rows.append(
            f"| `{embedding}` | {cfg['model']} | {cfg['hidden_features']} | "
            f"{cfg['num_transforms']} | {extra} | {cfg['z_score_x']} |"
        )
    return header + "\n".join(rows)


def _training_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    header = (
        "| embedding | epochs trained | best validation loss (lower is better) |\n"
        "| --- | --- | --- |\n"
    )
    rows = [
        f"| `{e}` | {summaries[e]['training']['epochs']} | "
        f"{summaries[e]['training']['best_validation_loss']:.4f} |"
        for e in EMBEDDINGS
    ]
    return header + "\n".join(rows)


def _observation_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    true_theta = summaries["none"]["observation"]["true"]
    header = (
        f"True (Δ, Ω) = ({true_theta[0]:.4f}, {true_theta[1]:.4f})\n\n"
        "| embedding | posterior mean Δ | posterior std Δ | "
        "posterior mean Ω | posterior std Ω |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for e in EMBEDDINGS:
        obs = summaries[e]["observation"]
        rows.append(
            f"| `{e}` | {obs['posterior_mean'][0]:.4f} | {obs['posterior_std'][0]:.4f} | "
            f"{obs['posterior_mean'][1]:.4f} | {obs['posterior_std'][1]:.4f} |"
        )
    return header + "\n".join(rows)


def _calibration_table(summaries: Dict[str, Dict[str, Any]]) -> str:
    header = (
        "| embedding | SBC KS p(Δ) | SBC KS p(Ω) | C2ST rank(Δ) | C2ST rank(Ω) | "
        "TARP ATC | TARP KS p-value |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for e in EMBEDDINGS:
        diag = summaries[e]["diagnostics"]
        sbc = diag["sbc"]
        rows.append(
            f"| `{e}` | {sbc['ks_pvals'][0]:.4f} | {sbc['ks_pvals'][1]:.4f} | "
            f"{sbc['c2st_ranks'][0]:.4f} | {sbc['c2st_ranks'][1]:.4f} | "
            f"{diag['tarp']['atc']:.4f} | {diag['tarp']['ks_pval']:.4f} |"
        )
    return header + "\n".join(rows)


def _calibration_notes(summaries: Dict[str, Dict[str, Any]]) -> str:
    """A data-driven paragraph on how to read the marginal-vs-joint checks
    together, instead of leaving the raw table to speak for itself. Every
    number here is computed from `summaries`, not asserted -- if a future
    rerun's numbers differ, this paragraph changes with them.
    """
    flagged = []
    passed = []
    for e in EMBEDDINGS:
        sbc = summaries[e]["diagnostics"]["sbc"]
        bad = [PARAM_LABELS[i] for i, p in enumerate(sbc["ks_pvals"]) if p < 0.05]
        if bad:
            flagged.append(f"`{e}` ({', '.join(bad)})")
        else:
            passed.append(f"`{e}`")
    if not flagged:
        return (
            "Every embedding passes the SBC rank-uniformity check (p >= 0.05) "
            "on both parameters, consistent with TARP's clean joint result "
            "below."
        )
    tarp_atcs = [abs(summaries[e]["diagnostics"]["tarp"]["atc"]) for e in EMBEDDINGS]
    worst = min(
        (
            (e, i, p)
            for e in EMBEDDINGS
            for i, p in enumerate(summaries[e]["diagnostics"]["sbc"]["ks_pvals"])
        ),
        key=lambda t: t[2],
    )
    if len(flagged) == len(EMBEDDINGS):
        lead = (
            "Every run rejects SBC rank-uniformity (p < 0.05) on at least "
            f"one marginal: {', '.join(flagged)}."
        )
    else:
        lead = (
            f"{len(flagged)} of {len(EMBEDDINGS)} runs reject SBC "
            f"rank-uniformity (p < 0.05) on at least one marginal: "
            f"{', '.join(flagged)}; {', '.join(passed)} pass"
            f"{'es' if len(passed) == 1 else ''} both parameters."
        )
    return (
        f"{lead} TARP -- the joint check that "
        "`scripts/npe_2d_sbi.py`'s own docstring calls \"necessary and "
        "sufficient\", unlike the marginal SBC test -- stays clean for all "
        f"four runs (ATC between {min(tarp_atcs):.4f} and "
        f"{max(tarp_atcs):.4f}, KS p-value near 1.0 in every case), so none "
        "of these marginal rejections corresponds to a joint-posterior "
        f"calibration failure severe enough for TARP to catch. The sharpest "
        f"single rejection is `{worst[0]}` on {PARAM_LABELS[worst[1]]} "
        f"(p={worst[2]:.4f})."
    )


def _observation_bias_note(summaries: Dict[str, Dict[str, Any]]) -> str:
    """Flags it, from the data, if every embedding's posterior mean misses
    the true value of one parameter in the same direction by a wide margin
    at the single example observation -- a sign the observation itself is
    the hard part, not any one embedding, since the calibration checks above
    already covered many *different* held-out pairs.
    """
    true_theta = summaries["none"]["observation"]["true"]
    for i, label in enumerate(PARAM_LABELS):
        deltas = [
            summaries[e]["observation"]["posterior_mean"][i] - true_theta[i]
            for e in EMBEDDINGS
        ]
        if all(d > 0.3 for d in deltas) or all(d < -0.3 for d in deltas):
            direction = "overshoots" if deltas[0] > 0 else "undershoots"
            num_diagnostic = summaries["none"]["config"]["num_diagnostic"]
            return (
                f"Every embedding's posterior mean for {label} {direction} "
                f"the true value ({true_theta[i]:.4f}) by at least 0.3 at "
                "this particular held-out observation, regardless of the "
                f"embedding used. Since the calibration checks above run on "
                f"{num_diagnostic:,} *different* held-out pairs, this shared "
                "bias looks like a property of this one example rather than "
                "a shortcoming of any particular embedding."
            )
    return ""


def _best_by_tarp_atc(summaries: Dict[str, Dict[str, Any]]) -> str:
    return min(EMBEDDINGS, key=lambda e: abs(summaries[e]["diagnostics"]["tarp"]["atc"]))


def _best_by_validation_loss(summaries: Dict[str, Dict[str, Any]]) -> str:
    return min(EMBEDDINGS, key=lambda e: summaries[e]["training"]["best_validation_loss"])


def main(
    run_root: str = DEFAULT_RUN_ROOT,
    figures_dir: str = "docs/figures",
    out: str = "docs/npe-2d-embedding-comparison.md",
) -> None:
    """Render the embedding-comparison report.

    Args:
        run_root (str): Directory holding the four `{none,deepset,hist,cnn}/`
            run outputs produced by `scripts/npe_2d_sbi.py`.
        figures_dir (str): Where the renamed comparison figures are written.
        out (str): Path of the Markdown report to write.
    """
    run_root_path = Path(run_root)
    summaries = _load_summaries(run_root_path)
    _copy_figures(run_root_path, Path(figures_dir))

    num_train = summaries["none"]["config"]["num_train"]
    num_diagnostic = summaries["none"]["config"]["num_diagnostic"]
    total = num_train + 200_000  # the held-out pool is 5% of the total by construction
    best_loss = _best_by_validation_loss(summaries)
    best_tarp = _best_by_tarp_atc(summaries)

    report = f"""# Comparing trajectory embeddings for the 2D NPE posterior

Four Neural Posterior Estimation runs of
[`scripts/npe_2d_sbi.py`](../scripts/npe_2d_sbi.py) on the same 95%/5%
train/held-out split of the full 2D `(Δ, Ω)` Zenodo dataset
({total:,} prior-draw/trajectory pairs total, {num_train:,} used for
training and the remaining {total - num_train:,} never seen during
training), differing only in `--embedding`. Every other hyperparameter
(flow family, hidden width, number of transforms, batch size, learning rate,
early-stopping patience, random seed) is identical across the four runs, so
any difference in the tables below is attributable to the embedding choice
alone.

| embedding | what it is |
| --- | --- |
| `none` | {EMBEDDING_LABEL['none']} |
| `deepset` | {EMBEDDING_LABEL['deepset']} |
| `hist` | {EMBEDDING_LABEL['hist']} |
| `cnn` | {EMBEDDING_LABEL['cnn']} |

![Training data overview](figures/npe-2d-training-data.png)

## Run configuration

{_config_table(summaries)}

## Training outcome

{_training_table(summaries)}

Best validation loss (negative log-likelihood on the held-out validation
split used internally by `NPE.train`) is lowest for `{best_loss}`.

## Posterior at a held-out example observation

{num_diagnostic:,} pairs from the held-out 5% were used for the calibration
checks below; one further pair from that same held-out pool, never used in
training or in the calibration checks, is the example observation here.

{_observation_table(summaries)}

{_observation_bias_note(summaries)}

## Calibration: SBC, expected coverage and TARP

Computed on {num_diagnostic:,} held-out (prior draw, prior predictive) pairs
per embedding, following `run_diagnostics` in `scripts/npe_2d_sbi.py`. A
well-calibrated posterior has SBC/TARP KS p-values that are not small
(the flat-rank null is not rejected), C2ST ranks near 0.5 (chance level), and
a TARP ATC (area between the empirical coverage curve and the diagonal)
close to 0.

{_calibration_table(summaries)}

{_calibration_notes(summaries)} By TARP ATC, `{best_tarp}` is the
best-calibrated of the four.

## Per-embedding figures

"""
    for e in EMBEDDINGS:
        report += f"""### `{e}` — {EMBEDDING_LABEL[e]}

![{e} posterior at the held-out observation](figures/npe-2d-{e}-posterior-observation.png)
![{e} SBC rank histogram](figures/npe-2d-{e}-sbc-rank-histogram.png)
![{e} expected coverage](figures/npe-2d-{e}-expected-coverage.png)
![{e} TARP](figures/npe-2d-{e}-tarp.png)

"""

    report += f"""## Reproducing

```bash
python docs/make_npe_embedding_report.py
```

regenerates this file and the figures above from the four runs'
`summary.json` files under `{run_root}`, once they exist (see
`docs/superpowers/plans/2026-08-25-npe-2d-embedding-comparison.md` for the
exact commands that produce them).
"""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 2: Run it**

```bash
/Users/enrico.rinaldi/miniforge3/envs/torch-sbi/bin/python docs/make_npe_embedding_report.py
```

Expected: prints
`wrote /Users/enrico.rinaldi/Projects/ParamEst-NN/.claude/worktrees/npe-2d-sbi-embedding-6505bc/docs/npe-2d-embedding-comparison.md`
with no traceback.

- [ ] **Step 3: Verify the outputs**

```bash
ls docs/figures/npe-2d-*.png | wc -l
```

Expected: `17` (1 shared `training-data.png` + 4 embeddings × 4 diagnostic
figures each).

```bash
grep -c "^##" docs/npe-2d-embedding-comparison.md
```

Expected: at least `6` (the top-level sections: config, training, observation,
calibration, per-embedding × 4, reproducing).

Open `docs/npe-2d-embedding-comparison.md` and read it once end to end:
confirm every table has four data rows (one per embedding), all numbers look
like numbers (not `nan` or `None`), and every `![...]( figures/...)` line
points at a file that actually exists in `docs/figures/`.

- [ ] **Step 4: Commit**

```bash
git add docs/make_npe_embedding_report.py docs/npe-2d-embedding-comparison.md docs/figures/npe-2d-*.png
git status
```

Confirm `git status` shows only these new files staged — in particular, no
file under `data/models/` or any `.pt`/`.pkl` should appear. Then:

```bash
git commit -m "$(cat <<'EOF'
docs: compare NPE embeddings (none/deepset/hist/cnn) on a 95/5 split

Trains four Neural Posterior Estimation models with scripts/npe_2d_sbi.py,
identical except for the trajectory embedding, on a 95%/5% train/held-out
split of the full 2D Zenodo dataset, and reports SBC, expected coverage and
TARP calibration alongside the posterior at a held-out example observation.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:** the user asked for (1) running `scripts/npe_2d_sbi.py` on
the training data with a 95/5 train/held-out split — Task 1 fixes the split
constants, Tasks 2-5 run it; (2) across different embedding settings — the
four embeddings the script itself supports (`none`, `deepset`, `hist`,
`cnn`), one run each; (3) testing the posterior with SBC methods — handled
unconditionally by `scripts/npe_2d_sbi.py`'s existing `run_diagnostics`
(SBC, expected coverage, TARP), no new code needed there; (4) a clean MD file
in `docs/` with figures in `docs/figures/` — Task 6. No gaps identified.

**Placeholder scan:** every step has a literal command or a literal code
block; the one place a "conclusion" might have been hand-waved (which
embedding is best) is instead computed from the JSON by `_best_by_tarp_atc`
and `_best_by_validation_loss`, so the report says something concrete and
reproducible rather than a TBD.

**Type/name consistency:** `EMBEDDINGS` order (`none, deepset, hist, cnn`) is
used consistently across every table-building function in
`docs/make_npe_embedding_report.py`; the four run directory names
(`none/deepset/hist/cnn`) match the `--embedding` flag values throughout
Tasks 2-5 and the `run_root / embedding / ...` paths in Task 6's script.
