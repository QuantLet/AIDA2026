"""Stage 02b — Chronos-T5 fed RETURNS instead of prices, on the lab's own 500 days.

Why this exists. Chronos rescales every context by its own mean and scale before
tokenising it. A price path has a level and a trend to rescale; a daily return series is
already near zero mean with a tiny scale, so the rescaling has almost nothing to work
with and the model's predictive spread collapses. The deck used to quote this failure
from a longer companion run. It is measured here instead, on the same 500 scored days
and the same weights as every other Chronos number in the course, so the slide does not
have to send anyone to another study.

Protocol matches src/02_chronos_lab.py exactly, one thing changed: the context is the
last CONTEXT daily returns rather than the last CONTEXT closing prices, and the sampled
values are read as returns directly instead of being converted from a price.

Output: precomputed/chronos_t5_returns_<ASSET>.csv   date, level, var, es
        precomputed/chronos_t5_returns_<ASSET>_facts.csv   the two numbers the slide uses

Usage:  python src/02b_chronos_returns.py [ASSET] [--samples N]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "notebook"))
from labcommon import ALPHAS, CONTEXT, LAB, NUM_SAMPLES, PRECOMP, SEED  # noqa: E402

T5 = "amazon/chronos-t5-mini"


def main():
    asset = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
             else "SPX").upper()
    n_samples = NUM_SAMPLES
    if "--samples" in sys.argv:
        n_samples = int(sys.argv[sys.argv.index("--samples") + 1])

    import torch
    from chronos import BaseChronosPipeline

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rets = pd.read_csv(LAB / f"returns_{asset}.csv", parse_dates=["date"])
    bench = pd.read_csv(LAB / f"bench_{asset}.csv", parse_dates=["date"])
    test_dates = sorted(bench["date"].unique())
    r = rets.set_index("date")["ret"].astype(float)
    pos = {d: i for i, d in enumerate(r.index)}
    idx = [pos[pd.Timestamp(d)] for d in test_dates]
    assert min(idx) >= CONTEXT, "not enough history for the context"

    rv = r.values
    pipe = BaseChronosPipeline.from_pretrained(T5, device_map="cpu", dtype=torch.float32)

    # the draw count is part of the identity of a run, so it is part of the file name
    tag = "" if n_samples == NUM_SAMPLES else f"_N{n_samples}"
    ckpt = PRECOMP / f"_t5_returns_raw_{asset}{tag}.csv"
    done = set()
    if ckpt.exists():
        done = set(pd.read_csv(ckpt)["i"].tolist())
        print(f"  resuming: {len(done)} forecasts already computed", flush=True)
    todo = [(i, t) for i, t in enumerate(idx) if i not in done]

    rows, t0, B = [], time.time(), 4
    for k in range(0, len(todo), B):
        chunk = todo[k:k + B]
        ctx = torch.tensor(np.stack([rv[t - CONTEXT:t] for _, t in chunk]),
                           dtype=torch.float32)
        with torch.no_grad():
            samp = pipe.predict(ctx, 1, num_samples=n_samples)[:, :, 0].numpy()
        for j, (i, t) in enumerate(chunk):
            rs = samp[j]                       # already on the return scale
            for a in ALPHAS:
                q = float(np.quantile(rs, a))
                tail = rs[rs <= q]
                var = -q
                rows.append((i, a, var,
                             max(-float(tail.mean()) if tail.size else var, var),
                             float(rs.std(ddof=1))))
        if len(rows) >= 300:
            _flush(rows, ckpt)
            rows = []
        if (k // B) % 20 == 0:
            frac = (k + len(chunk)) / max(len(todo), 1)
            el = time.time() - t0
            print(f"  {k + len(chunk)}/{len(todo)}  {frac * 100:5.1f}%  "
                  f"eta {el / max(frac, 1e-9) * (1 - frac) / 60:.1f}m", flush=True)
    if rows:
        _flush(rows, ckpt)

    raw = pd.read_csv(ckpt).drop_duplicates(subset=["i", "level"])
    raw["date"] = [test_dates[i] for i in raw["i"]]
    out = raw[["date", "level", "var", "es"]].sort_values(["date", "level"])
    out.to_csv(PRECOMP / f"chronos_t5_returns_{asset}.csv", index=False)

    # The two numbers the slide quotes, both on this 500-day span.
    pred_sd = float(raw.groupby("i")["pred_sd"].first().mean())
    realised = float(r.loc[test_dates[0]:test_dates[-1]].std(ddof=1))
    facts = pd.Series({"chronos_returns_pred_sd": round(pred_sd, 2),
                       "chronos_returns_realised_sd": round(realised, 2),
                       "chronos_returns_n": len(test_dates),
                       "chronos_returns_samples": n_samples})
    facts.rename("value").rename_axis("fact").to_csv(
        PRECOMP / f"chronos_t5_returns_{asset}_facts.csv")
    print(f"\n  predictive sd {pred_sd:.2f} against a realised {realised:.2f} "
          f"over {len(test_dates)} days")


def _flush(rows, ckpt):
    df = pd.DataFrame(rows, columns=["i", "level", "var", "es", "pred_sd"])
    df.to_csv(ckpt, mode="a", header=not ckpt.exists(), index=False)


if __name__ == "__main__":
    main()
