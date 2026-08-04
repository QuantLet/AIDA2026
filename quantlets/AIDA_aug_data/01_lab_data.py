"""Stage 01 — carve the lab dataset out of the certified aida-risk run.

Nothing is downloaded and nothing is simulated. The price series are the raw Yahoo
Finance files already in aida-risk/data/raw (retrieved 2026-07-20 for the S&P, 2026-07-25
for the rest) and the classical forecasts are the ones the July deck's numbers trace to.
The lab therefore inherits a protocol it does not have time to re-run: rolling window
1000 days, refit every 20 days, burn-in 250, information up to t-1 only.

Outputs, per asset, into data/lab/:
    returns_<ASSET>.csv      date, close, ret          full sample
    bench_<ASSET>.csv        date, ret, model, level, var, es    over the test span only
    manifest.json            provenance, spans, checksums

Usage:  python src/01_lab_data.py
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labcommon import (ALPHAS, ASSETS, LAB, RISK, TEST_N,  # noqa: E402
                       risk_proc)

BENCH_MODELS = ["HS", "GARCH-t", "NN-t"]   # kept teachable: no model the lab cannot explain


def _sha(path, n=12):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]


def main():
    manifest = {
        "source": "aida-risk certified run; see aida-risk/README.md",
        "protocol": "rolling window 1000d, refit every 20d, burn-in 250d, info up to t-1",
        "test_n": TEST_N,
        "benchmark_models": BENCH_MODELS,
        "assets": {},
    }

    for asset in ASSETS:
        proc = risk_proc(asset)
        rets = pd.read_csv(proc / "returns.csv", index_col=0, parse_dates=True)
        fc = pd.read_csv(proc / "forecasts.csv", parse_dates=["date"])

        # The test span is defined on the forecast file, not on the return file: a day
        # with a realised return but no out-of-sample forecast cannot be scored, and
        # taking the span from the returns would silently shift the window by the
        # burn-in for every asset.
        dates = sorted(fc["date"].unique())
        span = dates[-TEST_N:]
        t0, t1 = pd.Timestamp(span[0]), pd.Timestamp(span[-1])

        bench = fc[(fc["date"] >= t0) & (fc["date"] <= t1) &
                   fc["model"].isin(BENCH_MODELS) &
                   fc["level"].isin(ALPHAS)].copy()
        bench = bench.sort_values(["model", "level", "date"])

        # The realised return must agree between the two files, or every breach count
        # downstream is scored against the wrong day.
        chk = bench.merge(rets["ret"].rename("ret_src"), left_on="date",
                          right_index=True, how="left")
        bad = (chk["ret"] - chk["ret_src"]).abs() > 1e-9
        assert not bad.any(), f"{asset}: return mismatch on {int(bad.sum())} rows"

        rpath = LAB / f"returns_{asset}.csv"
        bpath = LAB / f"bench_{asset}.csv"
        rets.to_csv(rpath)
        bench.to_csv(bpath, index=False)

        got = sorted(bench["model"].unique())
        manifest["assets"][asset] = {
            "ticker": ASSETS[asset]["ticker"],
            "label": ASSETS[asset]["label"],
            "n_obs": int(len(rets)),
            "sample": [str(rets.index[0].date()), str(rets.index[-1].date())],
            "test_span": [str(t0.date()), str(t1.date())],
            "benchmarks_present": got,
            "levels": sorted(float(x) for x in bench["level"].unique()),
            "sha256_returns": _sha(rpath),
            "sha256_bench": _sha(bpath),
        }
        missing = set(BENCH_MODELS) - set(got)
        flag = f"  MISSING {sorted(missing)}" if missing else ""
        print(f"{asset:<5} {len(rets):>5} obs  test {t0.date()} .. {t1.date()}  "
              f"bench {got}{flag}")

    (LAB / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  -> {LAB}/  ({len(ASSETS)} assets, {TEST_N}-day test span)")
    print(f"  raw provenance: {RISK / 'data' / 'raw'}")


if __name__ == "__main__":
    main()
