# Experimental specification

## Scope and implementation

The proposed system is GATraj with a confidence-gated motion edge residual.
The primary experiment changes only the social interaction representation.
The temporal encoder, decoder, 20 output modes, winner-takes-all Laplace loss,
soft-target classification loss and optimizer remain the GATraj implementation.
This is not an implementation of the published GATraj+PnS system or a claim
that velocity/closest-approach features are new to the literature.

For receiver i and sender j, use only the last observed position p and the
mean v of the last three observed displacements. Define r = p_j - p_i,
u = v_j - v_i and the future horizon H = 12 sampled steps:

    tau_raw = -(r dot u) / max(u dot u, 1e-8)
    tau = clamp(tau_raw, 0, H)
    d_closest = ||r + tau * u||
    closing = -(r dot u) / max(||r||, 1e-4)
    approaching = (r dot u < 0) and (u dot u > 1e-8)
    risk = exp(-0.5 * (d_closest / 2 m)^2) * approaching

Velocity features are tanh(u / 0.5). Closest-approach features are
[tau/H, min(d_closest/2 m, 4), tanh(closing/0.5), risk]. Positions are in
metres; velocities are metres per sampled frame. Two agents with identical
velocity have zero closing risk. Past closest approaches are clamped to the
present. This extrapolation is an interaction proxy and cannot anticipate
turns, yielding, scene constraints or unobserved people.

Estimate each agent's motion inconsistency as the RMS norm of the recent
displacements minus their mean. Confidence is

    q_i = 1 / (1 + inconsistency_i / (||v_i|| + 0.1))
    q_ij = q_i * q_j

An MLP (6 -> 64 -> 64 with SiLU for the full feature set) maps motion features
to a residual on the original relative-position embedding, multiplied by q_ij.
The existing motion gate and attention both receive the enriched embedding.
The MLP output layer is zero-initialized. It is shared across message-passing
rounds; it adds 4,608 parameters for hidden size 64 and all six features.
Confidence is a deterministic heuristic, not calibrated uncertainty.

The optional adaptive graph preserves original edges and adds only approaching
pairs with current separation <=15 m, closest separation <=2 m, closest time
within the 12-frame horizon, and pair confidence >=0.5. No self edges are added.
It operates inside existing scene/window splits, so it cannot recover agents
omitted by the original loader's grouping. Without confidence gating, the
confidence criterion is disabled. Adaptive neighbors are off in the primary
experiment because additional noisy interactions could harm performance.

## Common corrections and controls

1. Start fresh training at epoch 1 and finish at epoch 1000. The old inclusive
   0..1000 loop trained one extra epoch.
2. Keep the full cosine horizon fixed during segmented training; restore model,
   optimizer, scheduler and RNG state. NumPy RNG integers are serialized as
   int64 tensors for safe PyTorch checkpoint loading. Move the model to its
   device before constructing Adam.
3. Seed batch ordering and invalidate caches when preprocessing settings change.
   Baseline and modified variants reuse the same batch construction.
4. Fix scalar-normalized attention and value-based masks in both primary arms.
   Keep the original learned linear score weights and explicit boolean edges;
   do not transform a zero score into a missing edge. Isolated agents retain
   their original hidden/cell states. An upstream-attention switch provides a
   separate control.
5. Disable autograd during evaluation and remove forced CUDA_LAUNCH_BLOCKING
   from model files; synchronize only when timing. These are common execution
   changes, not part of the proposed research contribution.

Always label results 'corrected implementation baseline' when these fixes are
enabled. Paper numbers and the authors' pretrained results are reported
references, not baseline results reproduced by this repository.

## Reporting

Use the README's six ablations and the additional upstream-attention control.
For each seed, train five independent leave-one-scene-out models with the
same 8/12 horizon, 20 modes, batch size 32, 1,000 epochs, learning rate 5e-4,
eta_min 1e-5, clip 1 and random rotations. Start from scratch for baseline and
modified arms; do not compare old changing-horizon runs against new runs.
Use the same seed set and data for paired comparisons. Fixed defaults should
be used across folds; choose any further hyperparameters using training scenes,
not the held-out test scene. The existing random-window validation option
can have overlapping trajectory windows and is not a substitute for a clearly
documented scene/track-separated validation design.

Report each fold's minADE20/minFDE20, unweighted mean across the five folds,
and mean/standard deviation across seeds. Report absolute and percentage
differences versus the corrected baseline. Distinguish individual seed
variation from evidence of a consistent gain. Do not average all agents across
folds and label that the paper's five-fold average.

The evaluator computes per-agent minADE and minFDE independently across the
same K=20 output means. Its groups use the original observed neighbor graph
for comparable density subsets. They count trajectory windows, not unique
pedestrians. Top-1 collision counts consider unordered pairs within each
loader scene/window at any of 12 discrete steps. They can count the same
physical interaction in overlapping windows and do not measure continuous-time
collisions. Independently selected best-of-20 paths are not jointly plausible
scene forecasts and are not used for the collision diagnostic.

Measure label-free forward latency with the same GPU, batch size, dataset,
PyTorch version and flags. Exclude three warmup batches and synchronize CUDA
before/after each timed forward. Variable-size batches preclude treating
milliseconds per batch as a fixed-agent real-time guarantee. Different GPU
models can train separate folds, but speed comparisons need the same device;
floating-point behavior means exact cross-GPU repeatability is not guaranteed.

## Verification record

The CPU check suite exercises analytical approaching/stationary geometry,
noisy confidence, optional neighbor caps, no self edges, zero-score masks,
isolated agents, component combinations, nonzero residual/attention gradients,
unchanged predicted means when future labels change, and exact continuation
of model/Adam/scheduler/RNG state across a saved checkpoint. The residual-on
model matches the corrected baseline at initialization when expansion is off.

A real-data startup check trained the modified Hotel-held-out fold for one
epoch with the 1,000-epoch horizon and resumed for another epoch (96 training
batches per epoch). Both the original test interface and the new evaluator
ran on 13 test batches / 1,075 agent-windows and agreed on ADE/FDE. All nine
focused CPU checks passed. These runs exercise startup and serialization; their errors are
not convergence results. The verification environment uses CPU PyTorch 2.6;
Kaggle GPU operation and full-seed performance must be measured by running the
notebook. No trained modified weights or performance claim are published.
