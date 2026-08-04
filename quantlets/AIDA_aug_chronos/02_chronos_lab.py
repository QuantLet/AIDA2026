"""Stage 02 — Chronos forecasts for the lab, both heads.

Two pipelines, because they answer the same question with different machinery and the
difference is the point of the exercise.

CHRONOS-BOLT reads a quantile head. It is fast enough to run live in the classroom
(the whole 500-day span for five assets takes seconds on a CPU) and it was trained on
the quantile grid 0.1, 0.2, ..., 0.9. Ask it for 0.05 and it does not extrapolate: it
returns the 0.1 value, having emitted a warning. Verified on 2026-07-31 with
chronos-forecasting 2.2.2 -- requesting [0.05, 0.1, 0.5, 0.9, 0.95] returned a vector
whose first two entries were bit-identical and whose last two were bit-identical. This
script records the full requested grid precisely so the notebook can put the clamp on
screen instead of describing it.

CHRONOS-T5 samples paths, so any quantile can be read off the sample distribution.
That is what makes a genuine 5% or 1% VaR available. It costs about 1.8 seconds per
forecast on a CPU at NUM_SAMPLES = 500, hence the precomputation here.

PARAMETERISATION follows aida-risk/src/02c_chronos.py: the model is given the last
CONTEXT closing PRICES and asked for the next one. Feeding it the return series was
tried in that pipeline and is not a fair test -- returns are near-zero-mean noise and
the model collapses to a predictive standard deviation of 0.14 against a realised 1.22.
Implied one-day returns are recovered as r = 100 * log(P_hat / P_{t-1}), a monotone
transform, so a price quantile maps to the return quantile of the same level.

The context ends at t-1. Nothing from day t enters any forecast.

Outputs into precomputed/:
    chronos_bolt_<ASSET>.csv    date, level, var, q_price, clamped
    chronos_t5_<ASSET>.csv      date, level, var, es

Usage (must use the interpreter with a working chronos install):
    ../aida-ensemble/venv/bin/python src/02_chronos_lab.py bolt SPX DAX N225 GOLD BTC
    ../aida-ensemble/venv/bin/python src/02_chronos_lab.py t5 SPX BTC
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labcommon import (ALPHAS, CONTEXT, LAB, NUM_SAMPLES, PRECOMP,  # noqa: E402
                       SEED)

BOLT = "amazon/chronos-bolt-tiny"     # ~9M parameters, runs anywhere
T5 = "amazon/chronos-t5-mini"         # sampling head, matches the certified run
BOLT_GRID = [0.01, 0.05, 0.10, 0.50, 0.90]
BATCH = 32


def _load(asset):
    rets = pd.read_csv(LAB / f"returns_{asset}.csv", index_col=0, parse_dates=True)
    bench = pd.read_csv(LAB / f"bench_{asset}.csv", parse_dates=["date"])
    test_dates = pd.DatetimeIndex(sorted(bench["date"].unique()))
    pos = {d: i for i, d in enumerate(rets.index)}
    idx = [pos[d] for d in test_dates]
    assert min(idx) >= CONTEXT, f"{asset}: not enough history for a {CONTEXT}-day context"
    return rets, test_dates, np.array(idx)


def run_bolt(asset):
    import torch
    from chronos import BaseChronosPipeline

    rets, dates, idx = _load(asset)
    px = rets["close"].values.astype(float)
    pipe = BaseChronosPipeline.from_pretrained(BOLT, device_map="cpu", dtype=torch.float32)

    rows, t0 = [], time.time()
    for i in range(0, len(idx), BATCH):
        chunk = idx[i:i + BATCH]
        ctx = torch.tensor(np.stack([px[t - CONTEXT:t] for t in chunk]), dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # the clamp warning fires on every batch
            with torch.no_grad():
                q, _ = pipe.predict_quantiles(ctx, prediction_length=1,
                                              quantile_levels=BOLT_GRID)
        qp = q[:, 0, :].numpy()                 # [batch, len(BOLT_GRID)] price quantiles
        for j, t in enumerate(chunk):
            r_q = 100.0 * np.log(np.maximum(qp[j], 1e-8) / px[t - 1])
            for k, a in enumerate(BOLT_GRID):
                # A level is flagged clamped when its price quantile is bit-identical to
                # the one at the model's lowest trained level. This is recorded, never
                # repaired: the notebook shows it.
                clamped = bool(a < 0.10 and np.isclose(qp[j, k], qp[j, BOLT_GRID.index(0.10)],
                                                       rtol=0, atol=0))
                rows.append((dates[i + j], a, -r_q[k], qp[j, k], clamped))

    out = pd.DataFrame(rows, columns=["date", "level", "var", "q_price", "clamped"])
    path = PRECOMP / f"chronos_bolt_{asset}.csv"
    out.to_csv(path, index=False)
    n_cl = int(out.groupby("level")["clamped"].mean().get(0.05, 0.0) * 100)
    print(f"{asset:<5} bolt  {len(dates)} dates in {time.time() - t0:5.1f}s  "
          f"5% clamped on {n_cl}% of days  mean VaR(5%) "
          f"{out[np.isclose(out.level, 0.05)]['var'].mean():.2f}  -> {path.name}")


def run_t5(asset, n_samples=NUM_SAMPLES):
    """Sampling head. n_samples sets the quantile noise floor: at alpha = 1% it is
    n_samples / 100 draws below the quantile, so it is part of the identity of a run
    and not a tuning knob. Anything other than the certified NUM_SAMPLES writes to its
    own checkpoint and its own output file, so the two cannot be mixed."""
    import torch
    from chronos import BaseChronosPipeline

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rets, dates, idx = _load(asset)
    px = rets["close"].values.astype(float)
    pipe = BaseChronosPipeline.from_pretrained(T5, device_map="cpu", dtype=torch.float32)

    tag = "" if n_samples == NUM_SAMPLES else f"_N{n_samples}"
    ckpt = PRECOMP / f"_t5_raw_{asset}{tag}.csv"
    done = set()
    if ckpt.exists():
        done = set(pd.read_csv(ckpt)["i"].tolist())
        print(f"  resuming: {len(done)} forecasts already computed")
    todo = [(i, t) for i, t in enumerate(idx) if i not in done]

    rows, t0, B = [], time.time(), 4
    for k in range(0, len(todo), B):
        chunk = todo[k:k + B]
        ctx = torch.tensor(np.stack([px[t - CONTEXT:t] for _, t in chunk]),
                           dtype=torch.float32)
        with torch.no_grad():
            samp = pipe.predict(ctx, 1, num_samples=n_samples)[:, :, 0].numpy()
        for j, (i, t) in enumerate(chunk):
            rs = 100.0 * np.log(np.maximum(samp[j], 1e-8) / px[t - 1])
            for a in ALPHAS:
                q = np.quantile(rs, a)
                tail = rs[rs <= q]
                var = -q
                rows.append((i, a, var, max(-tail.mean() if tail.size else var, var)))
        if len(rows) >= 300:
            _flush(rows, ckpt)
            rows = []
        if (k // B) % 20 == 0:
            frac = (k + len(chunk)) / max(len(todo), 1)
            el = time.time() - t0
            print(f"  {k + len(chunk)}/{len(todo)}  {frac * 100:5.1f}%  "
                  f"eta {el / max(frac, 1e-9) * (1 - frac) / 60:.1f}m", flush=True)
    _flush(rows, ckpt)

    raw = pd.read_csv(ckpt).drop_duplicates(subset=["i", "level"])
    raw["date"] = dates[raw["i"].values]
    out = raw[["date", "level", "var", "es"]].sort_values(["level", "date"])
    path = PRECOMP / f"chronos_t5_{asset}{tag}.csv"
    out.to_csv(path, index=False)
    print(f"{asset:<5} t5    {len(dates)} dates, {n_samples} samples, "
          f"{(time.time() - t0) / 60:.1f}m  mean VaR(5%) "
          f"{out[np.isclose(out.level, 0.05)]['var'].mean():.2f}  -> {path.name}")


def _flush(rows, ckpt):
    if not rows:
        return
    pd.DataFrame(rows, columns=["i", "level", "var", "es"]).to_csv(
        ckpt, mode="a", header=not ckpt.exists(), index=False)


if __name__ == "__main__":
    # usage: 02_chronos_lab.py {bolt|t5} [ASSET ...] [--samples N]
    argv = sys.argv[1:]
    n_samples = NUM_SAMPLES
    if "--samples" in argv:
        k = argv.index("--samples")
        n_samples = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    head = argv[0] if argv else "bolt"
    assets = argv[1:] or ["SPX"]
    for a in assets:
        run_bolt(a) if head == "bolt" else run_t5(a, n_samples)
