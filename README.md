# GATraj-RiskMotion

A lightweight experimental extension of [GATraj](https://github.com/mengmengliu1998/GATraj)
for the five ETH/UCY pedestrian folds. It conditions GATraj's existing social
gate and attention on **relative velocity and anticipated closest approach**,
with confidence gating and optional adaptive neighbors. There are no new labels,
maps, external predictors, training stages, or loss terms.

**This is an unvalidated research hypothesis. No improved ADE/FDE results are
claimed.** The encoder, 20-mode decoder, Laplace/classification loss and
1,000-epoch cosine schedule are retained. A zero-initialized edge residual
starts the modified model at the shared corrected baseline.

The original MIT license and Git history are retained. Supplied upstream
checkpoints are excluded from this repository's current files: they are not
RiskMotion weights. See [the original README](docs/UPSTREAM_README.md) for paper
attribution and dataset sources. The code supports ETH/UCY, not nuScenes.

## Kaggle

Import [Kaggle_RiskMotion.ipynb](Kaggle_RiskMotion.ipynb), enable **Internet**
and a **GPU accelerator**, and run its cells. It trains one held-out fold in
100-epoch chunks with an unchanged 1,000-epoch schedule. Changing
`MOTION_MODULE` selects the baseline or modified model. The notebook resumes
from the latest numbered checkpoint present in its output directory.

Or use notebook cells directly:

```python
!git clone https://github.com/Kedar2002/GATraj-RiskMotion.git /kaggle/working/GATraj-RiskMotion
%cd /kaggle/working/GATraj-RiskMotion
!python -m pip install -q -r requirements.txt
```

```python
!python train.py \
  --test_set 1 --seed 1 --num_epochs 1000 --epochs_per_run 0 \
  --x_encoder_layers 3 --eta_min 1e-5 --batch_size 32 \
  --learning_rate 5e-4 --randomRotate True --final_mode 20 \
  --neighbor_thred 10 --using_cuda True --gpu 0 --clip 1 \
  --pass_time 2 --ifGaussian False --SR True --input_offset True \
  --fix_attention True --motion_module True \
  --relative_velocity True --closest_approach True \
  --confidence_gate True --adaptive_neighbors False \
  --save_base_dir /kaggle/working/runs
```

For the shared corrected baseline, change **only** `--motion_module False`.
Experiment directories are generated from flags and seed, so baseline,
component ablations and seeds do not overwrite one another. You can override
the directory name with `--train_model <unique-name>`; use unique names when
altering other hyperparameters such as `risk_radius`.

Fold IDs: **0=ETH, 1=Hotel, 2=Zara1, 3=Zara2, 4=University**. For each fold,
the other four scenes train an independent model. The bundled CSV files are
used directly, with 8 observed and 12 predicted steps.

## On/off switches

| Argument | Default | Effect |
|---|---|---|
| `--motion_module` | `False` | Master switch for the proposed edge residual |
| `--relative_velocity` | `True` | Include two relative-velocity features when the master switch is on |
| `--closest_approach` | `True` | Include time, separation, closing speed and risk features |
| `--confidence_gate` | `True` | Attenuate motion features when recent velocities are inconsistent |
| `--adaptive_neighbors` | `False` | Add approaching agents within a bounded 15 m radius |
| `--fix_attention` | `True` | Shared scalar-score/mask correction; keep identical in baseline and modification |

Motion settings are `velocity_window=3` observed displacements,
`velocity_scale=0.5` metres per sampled frame, `risk_radius=2.0` metres and
`adaptive_radius=15.0` metres. Times use sampled frames, not seconds.
Defaults are fixed starting hypotheses, not values optimized on test folds.
Keep adaptive expansion off for the primary comparison and measure it as
an additional ablation.

## Resume correctly

Keep `--num_epochs 1000` throughout. Add `--epochs_per_run 100` to stop after
100 further epochs and `--load_model 100` to continue at epoch 101. Repeat the
same architecture, seed and training arguments. `--epochs_per_run 0` runs to
the final epoch. The notebook automates this process.

Checkpoints contain weights, Adam moments, scheduler state, Python/NumPy/
PyTorch/CUDA RNG states, and the actual experiment configuration. An
incompatible configuration raises an error. Checkpoints are written through
a temporary file, then replaced. Batch caches include the fold, seed and
preprocessing settings in their identity.

Outputs live under `<save_base_dir>/<fold>/<experiment>/`. Kaggle's
`/kaggle/working` survives as output only when saved with the notebook version;
attach/download saved output and restore it to continue in a new runtime.
Automatic resume finds files already present; it does not recover an unsaved
runtime. Every epoch is checkpointed, so plan output storage accordingly.

## Evaluation and experimental controls

Use `evaluate.py` with the same training command arguments, replacing
`train.py` with `evaluate.py` and adding `--phase test --load_model 1000`.
`--epochs_per_run` has no effect during evaluation. The original
`train.py --phase test --load_model 1000` interface also works.

`evaluation_1000.json` contains:

- Independently minimized ADE and FDE across 20 modes, in metres.
- Per-agent results for sparse (0–2 base-graph neighbors) and crowded (>2)
  observed groups; these are window-based diagnostic subsets.
- Parameter count and synchronized forward latency after three warmup batches.
  Compare on the same GPU and batching; variable scene sizes affect latency.
- Collision counts for the label-free highest-probability mode of each agent,
  with a 0.2 m threshold and ground-truth pair counts for context. Modes are
  not a joint scene distribution; this is a diagnostic, not a safety metric.

Run the following controls on all five folds. See
[EXPERIMENTS.md](docs/EXPERIMENTS.md) for equations, limitations and reporting.

| Experiment | motion | velocity | closest approach | confidence | adaptive |
|---|---|---|---|---|---|
| Corrected baseline | False | ignored | ignored | ignored | ignored |
| Relative velocity | True | True | False | False | False |
| Closest approach | True | False | True | False | False |
| Both feature sets | True | True | True | False | False |
| Proposed primary model | True | True | True | True | False |
| Optional graph expansion | True | True | True | True | True |

Use seeds 1, 2 and 3 for the baseline and primary model and, where resources
permit, the component ablations. Compute the unweighted five-fold average per
seed, then mean and standard deviation across seeds. Keep optimizer, schedule,
batch size, augmentation, decoder mode count and checkpoint selection fixed.
Evaluate the final predetermined epoch; do not select the best test epoch.

There is a common baseline correction: upstream normalizes a scalar attention
score with LayerNorm, which removes its input dependence, then masks edges by
whether their score equals zero. This implementation uses the existing linear
score and an explicit neighbor mask, and preserves isolated agents. Compare
RiskMotion against **this same correction**, with `fix_attention=True` for
both. `--fix_attention False --motion_module False` restores the upstream
attention behavior for a separate control. That control still includes the
training-protocol fixes and is not a promise of reproducing paper numbers.

## Verification

```bash
python -m unittest discover -s tests -v
```

Checks cover closest-approach geometry, confidence, bounded neighbors,
attention masks, component ablations, singleton scenes, residual gradients,
future-label independence and exact CPU optimizer/scheduler/RNG continuation.
CPU startup/resume/evaluation on the real ETH/UCY loader is also recorded in
[EXPERIMENTS.md](docs/EXPERIMENTS.md). GPU convergence and the claimed
performance hypothesis require the full experiments.
