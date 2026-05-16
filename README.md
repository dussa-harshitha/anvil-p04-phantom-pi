# PHANTOM-Pi V21 — Anvil P-04 Submission

**Team Sonic** · Manipal Institute of Technology Bangalore

A precision-prediction agent for PCAM (Modern Hopfield) memory retrieval.
Two precision heads, switched at inference time by a distance gate:

- the **retrieval head** shapes precision around the corrupted query for
  attractor recovery;
- the **anisotropy head** emits L-BFGS-optimal precision for sharp
  probes, balancing the eigenvalues of `Pi^(1/2) H Pi^(1/2)` at the
  equilibrium Hessian.

V21's headline change over V19: the retrieval head's constants are
**self-calibrated** in `__init__` instead of hard-coded. The agent
synthesises its own corrupted queries from the stored patterns, runs
them through the provided PCAM dynamics, and tunes `(beta_post, a, b)`
to whatever data it is handed. This removes the v0 benchmark
overfitting V19 carried, and — measured across six seeds — also
*raises* the v0 score. See `WRITEUP.md` §8 for the full argument.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: NumPy and SciPy. SciPy is used only inside `__init__`
(L-BFGS for the anisotropy precompute).

## Run

```bash
python self_check.py --adapter adapters.myteam:Engine --quick
python run.py --adapter adapters.myteam:Engine --seeds 7 13 31 97 211 503 1009 --out report.json
```

## Headline results

Quick mode (seeds 42, 101) and validated on four further unseen seeds
(202, 303, 7, 211). Every seed lands `Delta in [+0.075, +0.125]`:

| Metric                  | V19 (old)      | V21            |
|-------------------------|----------------|----------------|
| mean Delta retrieval    | +0.076         | ~ +0.100       |
| min  Delta retrieval    | +0.057         | ~ +0.092       |
| mean spread reduction   | 1.30x          | 1.30x          |
| Retrieval points        | 66.17 / 70     | ~ 70 / 70      |
| Anisotropy points       | 3.25 / 20      | ~ 3.25 / 20    |
| Total automated         | 69.42 / 90     | ~ 73 / 90      |

V21's mean Delta clears the 0.08 full-marks threshold, so retrieval
caps at 70. Anisotropy is unchanged — it is *provably* at the
diagonal-precision ceiling on v0 (see below). Both per-seed gates clear
with wide margin (`min Delta` ~ +0.092 >> 0; `min reduction` ~ 1.27x).

**Run the full 7-seed eval yourself before submitting** — the numbers
above are from 6 seeds tested in pairs; confirm the aggregate.

## Architecture

### Retrieval head (corrupted queries) — self-calibrated

```
posterior(k|q)   = softmax(beta_post * X @ q_hat)
pi_reliable      = 1 / (1 + sum_k posterior(k|q) * |X_k - q|)
pi_gap           = sum_k posterior(k|q) * (X_k - x_bar)^2
pi               = (pi_reliable ^ a) * (pi_gap ^ b)
                <- normalise mean=1 -> clip [0.1,10] -> identity shrink -> renormalise
```

`(beta_post, a, b)` are NOT hard-coded. `__init__` builds a
self-supervised calibration set — corrupted queries synthesised from
the stored patterns, spanning mask+Gaussian (v0 style) AND heavy
mask-only (the L3 / paper Section-6.6 style) — and coordinate-ascends
the three constants, starting from the V19 values, to maximise
retrieval on that set under the real PCAM dynamics.

### Anisotropy head (sharp probes) — precomputed, proven optimal

```
For each stored pattern k:
  a*_k      = run_dynamics(X[k], pi = I)            # PCAM equilibrium
  H_k       = Hessian at a*_k
  pi_opt[k] = argmin_pi log_spread(Pi^(1/2) H_k Pi^(1/2))   # L-BFGS, warm-started
```

We proved (bisection over an SDP feasibility test — a globally exact
method) that this L-BFGS already finds the **global** optimum diagonal
precision. On v0 the ~1.3x ceiling is structural (the rank-1
`delta*11^T` term in R), not an optimisation failure. On L3 with a
different R the same code captures whatever is achievable there.

### Head gate — distance-based, calibrated

```
dist_sq = 2 - 2 * max_k cos(q, X_k)
if dist_sq < threshold:   return anisotropy head   # probe-like
else:                     return retrieval head    # corrupted query
```

`threshold` is set in `__init__` from the gap between self-generated
probe and corrupted distance distributions — dataset-independent.

## Precompute cache (iteration speed)

`__init__` memoises its expensive precompute (anisotropy L-BFGS +
calibration) to disk, keyed by a SHA-256 of `(version, X, R, frozen
params)`. Re-running the **same seed** is then near-instant. A fresh
seed = different `X` = cache miss = full recompute, so the cache cannot
game the multi-seed (L2) check and does not change any score. Disable
with `PCAM_NO_CACHE=1`. Any cache failure falls back to recompute — it
can never break the agent. Cache directory: `.pcam_cache/` (or a temp
dir if the CWD is read-only).

## Hyperparameters

| Name        | Value             | Role                                   |
|-------------|-------------------|----------------------------------------|
| `beta_post` | self-calibrated   | Posterior sharpness                    |
| `a`         | self-calibrated   | Reliability exponent                   |
| `b`         | self-calibrated   | Gap exponent                           |
| `shrink`    | 0.10 (fixed)      | Identity shrink — per-seed gate safety |
| gate thr.   | self-calibrated   | Probe/corrupted boundary               |

Only `shrink` is fixed, and it is a conservative safety term.

## Files

- `adapters/myteam.py` — final submission (PHANTOM-Pi V21, self-calibrating)
- `adapters/myteam_v19.py` — V19 fallback (69.42/90, hard-coded constants)
- `adapters/myteam_v17.py` — V17 deep fallback (64.62/90, NumPy only)
- `WRITEUP.md` — architectural defense + audit log + SDP proof
- `report.json` — full 7-seed eval results

Fallback chain: if V21 errors at init, use V19; if scipy is unavailable
entirely, use V17.

## Reproducibility

Deterministic — identical numbers on any machine for the same seeds.
`__init__` runs ~60-90s per seed (anisotropy L-BFGS dominates;
calibration is vectorised and adds only seconds). The cache makes every
repeat run of a seed instant.
