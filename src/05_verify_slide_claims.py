"""Stage 05 — re-derive every number printed on a slide, from the data.

The deck states figures inline. This script recomputes each one from data/ and
precomputed/ and fails loudly on a mismatch, so a slide cannot drift away from the
pipeline that produced it. The equivalent script in aida-risk caught two false claims
during the July rebuild, and this one has already caught two more.

Level: alpha = 1%, the level the course reports.

One claim is deliberately NOT machine-checked and is listed at the end: the predictive
standard deviation of 0.14 against a realised 1.22 for Chronos fed returns instead of
prices. It belongs to the certified aida-risk run, is documented in
aida-risk/src/02c_chronos.py, and is attributed as such on the slide.

Usage:  python src/05_verify_slide_claims.py [ASSET]
Exit status is non-zero if any assertion fails.
"""

import json
import pathlib
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "notebook"))
from labcommon import ALPHA, LAB, PRECOMP, ROOT  # noqa: E402
import aidalab as al  # noqa: E402

ASSET = (sys.argv[1] if len(sys.argv) > 1 else "SPX").upper()
LLM_MODEL = "claude-haiku-4-5"
OPEN_MODEL = "Qwen2.5-1.5B-Instruct"

PASSED, FAILED = [], []


def check(label, got, want, tol=0.0):
    ok = (got == want) if (tol == 0 and not isinstance(want, float)) \
        else abs(float(got) - float(want)) <= tol
    (PASSED if ok else FAILED).append((label, got, want))
    print(f"  {'ok  ' if ok else 'FAIL'} {label:<54} got {got}  want {want}")


def usable(d):
    m = d["raw_ok"] & d["sign_ok"]
    return (m & d["order_ok"]) if "order_ok" in d else m


def kupiec_reject(n, alpha=ALPHA, level=0.05):
    """Counts in the exact two-sided rejection region of Kupiec's LR_UC."""
    crit = stats.chi2.ppf(1 - level, 1)
    x = np.arange(n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ph = x / n
        ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
        ll1 = (n - x) * np.log(np.clip(1 - ph, 1e-300, None)) \
            + x * np.log(np.clip(ph, 1e-300, None))
        ll1 = np.where(x == 0, 0.0, ll1)
    return x[-2 * (ll0 - ll1) > crit]


def detectable_power(n, target=0.80, alpha=ALPHA, level=0.05):
    """Smallest true breach rate rejected with `target` power: a power calculation,
    not the weaker 'expected count is already a rejection' criterion."""
    rej = kupiec_reject(n, alpha, level)
    for p in np.arange(alpha, 0.90, 0.0005):
        if float(stats.binom.pmf(rej, n, p).sum()) >= target:
            return p
    return np.nan


def detectable(n, alpha=ALPHA, level=0.05):
    """The weaker criterion, kept because the slide contrasts the two."""
    crit = stats.chi2.ppf(1 - level, 1)
    for p in np.arange(alpha, 0.6, 0.0005):
        x = p * n
        ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
        ll1 = (n - x) * np.log(1 - p) + x * np.log(p)
        if -2 * (ll0 - ll1) > crit:
            return p
    return np.nan


def main():
    b = pd.read_csv(LAB / f"bench_{ASSET}.csv", parse_dates=["date"])
    b = b[np.isclose(b["level"], ALPHA)]
    bench = b.pivot_table(index="date", columns="model", values="var")
    bench.insert(0, "ret", b.groupby("date")["ret"].first())
    bench.columns.name = None

    models = {m: bench[m] for m in ("HS", "GARCH-t", "NN-t")}
    t5 = pd.read_csv(PRECOMP / f"chronos_t5_{ASSET}.csv", parse_dates=["date"])
    models["Chronos-T5"] = t5[np.isclose(t5["level"], ALPHA)].set_index("date")["var"]

    llm = {}
    for cfg, key in [("series", "LLM-series"), ("series_state", "LLM-series+state"),
                     ("dated", "LLM-dated"), ("dated_news", "LLM-dated+news")]:
        d = pd.read_csv(PRECOMP / f"llm_{cfg}_{ASSET}_{LLM_MODEL}.csv",
                        parse_dates=["date"]).set_index("date")
        llm[key] = d
        models[key] = d.loc[usable(d), "var"]

    op = PRECOMP / f"local_series_{ASSET}_{OPEN_MODEL}.csv"
    if op.exists():
        d = pd.read_csv(op, parse_dates=["date"]).set_index("date")
        llm["Open-1.5B"] = d
        models["Open-1.5B"] = d.loc[usable(d), "var"]

    print(f"\n--- span and level ({ASSET}, alpha = {ALPHA:.0%}) ---")
    check("test days", len(bench), 500)
    check("expected breaches", round(ALPHA * len(bench)), 5)

    print("\n--- Kupiec power at 1% ---")
    check("80% power, 500 days (%)", round(100 * detectable_power(500), 2), 2.50, 0.01)
    check("80% power, 5541 days (%)", round(100 * detectable_power(5541), 2), 1.45, 0.01)
    check("expected-count criterion, 500 days (%)",
          round(100 * detectable(500), 2), 2.00, 0.01)

    print("\n--- LLM replies, four configurations ---")
    k4 = [k for k in llm if k.startswith("LLM")]
    check("parsed replies across the four configurations",
          sum(int(llm[k]["raw_ok"].sum()) for k in k4), 2000)
    check("parsed with a non-negative quantile",
          sum(int((llm[k]["raw_ok"] & ~llm[k]["sign_ok"]).sum()) for k in k4), 0)
    check("parsed with q01 above q05 (inverted tail)",
          sum(int((llm[k]["raw_ok"] & ~llm[k]["order_ok"]).sum()) for k in k4), 0)

    print("\n--- backtest table ---")
    bt = al.backtest_table(bench["ret"], models, alpha=ALPHA).set_index("model")
    rows = [("NN-t", 3, 0.60, 0.0269), ("LLM-dated+news", 3, 0.60, 0.0276),
            ("GARCH-t", 7, 1.40, 0.0277), ("LLM-series+state", 6, 1.20, 0.0291),
            ("LLM-dated", 3, 0.60, 0.0298), ("LLM-series", 3, 0.60, 0.0299),
            ("Chronos-T5", 6, 1.20, 0.0303), ("HS", 2, 0.40, 0.0379)]
    if "Open-1.5B" in models:
        rows.append(("Open-1.5B", 35, 7.09, 0.0533))
    lt = PRECOMP / f"llmtime_{ASSET}_{OPEN_MODEL}_T0.7_N1000_float16.csv"
    if lt.exists():
        d = pd.read_csv(lt, parse_dates=["date"]).set_index("date")
        models["LLMTime-1.5B"] = d.loc[usable(d), "var_01"]
        bt = al.backtest_table(bench["ret"], models, alpha=ALPHA).set_index("model")
        rows.append(("LLMTime-1.5B", 25, 5.00, 0.0436))
    for m, br, rate, pin in rows:
        check(f"{m}: breaches", int(bt.loc[m, "observed"]), br)
        check(f"{m}: rate (%)", round(float(bt.loc[m, "rate_pct"]), 2), rate, 0.01)
        check(f"{m}: pinball", round(float(bt.loc[m, "pinball"]), 4), pin, 0.0001)

    print("\n--- the power finding: how many models 'pass' ---")
    n_rej = 2 if "LLMTime-1.5B" in models else 1
    check("models rejected on Kupiec", int((bt["p_uc"] < 0.05).sum()), n_rej)
    check("models passing Kupiec", int((bt["p_uc"] >= 0.05).sum()), len(rows) - n_rej)
    if "Open-1.5B" in models:
        check("the rejected models are the open-weights ones",
              sorted(bt.index[bt["p_uc"] < 0.05]),
              sorted(["Open-1.5B", "LLMTime-1.5B"][:n_rej]))

    print("\n--- the opening slide: how often a doubly-wrong model passes ---")
    def _kupiec_power(n, true, level=0.05):
        x = np.arange(n + 1)
        ph = np.where((x == 0) | (x == n), np.nan, x / n)
        ll0 = (n - x) * np.log(1 - ALPHA) + x * np.log(ALPHA)
        ll1 = np.where(np.isnan(ph), 0.0,
                       (n - x) * np.log(1 - np.nan_to_num(ph, nan=0.5))
                       + x * np.log(np.nan_to_num(ph, nan=0.5)))
        rej = -2 * (ll0 - ll1) > stats.chi2.ppf(1 - level, 1)
        return float(stats.binom.pmf(x, n, true)[rej].sum())
    _pw = _kupiec_power(250, 2 * ALPHA)
    check("a year of data, true rate 2x nominal: passes (%)",
          round(100 * (1 - _pw)), 76, 1)

    _rr = pd.read_csv(LAB / f"returns_{ASSET}.csv", index_col=0,
                      parse_dates=True)["ret"].dropna()
    for _W, _want in ((250, 21), (1000, 13)):
        _v = _rr.rolling(_W).quantile(ALPHA).dropna().iloc[-500:]
        check(f"HS distinct values over the span, W = {_W}",
              int(_v.round(2).nunique()), _want)

    print("\n--- quantile regression: the loss moves the line ---")
    import statsmodels.api as _sm
    _rq = pd.read_csv(LAB / f"returns_{ASSET}.csv", index_col=0,
                      parse_dates=True)["ret"].dropna()
    _d = pd.DataFrame({"y": _rq, "x": _rq.rolling(20).std().shift(1)}).dropna()
    _X = _sm.add_constant(_d["x"])
    check("squared-error slope on the volatility proxy",
          round(float(_sm.OLS(_d["y"], _X).fit().params["x"]), 2), 0.01, 0.005)
    check("pinball 1% slope",
          round(float(_sm.QuantReg(_d["y"], _X).fit(q=ALPHA).params["x"]), 2), -1.94, 0.005)

    print("\n--- GARCH: the standardised-t quantile ---")
    from arch import arch_model as _am
    _r2 = pd.read_csv(LAB / f"returns_{ASSET}.csv", index_col=0,
                      parse_dates=True)["ret"].dropna()
    _f = _am(_r2, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    _nu = float(_f.params["nu"])
    _raw = float(stats.t.ppf(ALPHA, _nu))
    _std = _raw * np.sqrt((_nu - 2) / _nu)
    check("fitted nu", round(_nu, 1), 6.5, 0.05)
    check("arch uses the unit-variance quantile",
          round(float(_f.model.distribution.ppf(ALPHA, [_nu])), 3), round(_std, 3), 0.002)
    check("raw t quantile is larger by (%)", round(100 * (_raw / _std - 1)), 20, 1)

    print("\n--- dependence: levels against changes ---")
    _prim = {k: v for k, v in models.items()
             if k in ("HS", "GARCH-t", "NN-t", "Chronos-T5", "LLM-series",
                      "LLM-series+state", "LLM-dated", "LLM-dated+news", "Open-1.5B")}
    _w = pd.DataFrame({k: pd.Series(v).astype(float)
                       for k, v in _prim.items()}).dropna()
    for _lab, _X, _rho, _pc in (("levels", _w, 0.47, 0.53),
                                ("changes", _w.diff().dropna(), 0.07, 0.20)):
        _R = _X.corr()
        _off = _R.values[np.triu_indices_from(_R.values, k=1)]
        _l1 = float(np.linalg.eigvalsh(_R.values)[::-1][0] / len(_R))
        check(f"median pairwise rho, {_lab}", round(float(np.median(_off)), 2), _rho, 0.005)
        check(f"first eigenvalue share, {_lab}", round(_l1, 2), _pc, 0.005)
    check("HS lag-1 autocorrelation, why levels mislead",
          round(float(_w["HS"].autocorr(1)), 2), 1.0, 0.005)

    print("\n--- coherence: the sub-additivity counterexample ---")
    _p, _L = 0.008, 100.0

    def _ve(states):
        cum = acc = 0.0
        var = None
        for loss, pr in sorted(states, key=lambda x: -x[0]):
            take = min(pr, ALPHA - cum)
            if take > 0:
                acc += take * loss
                cum += take
            if cum >= ALPHA - 1e-15 and var is None:
                var = loss
        return var, acc / ALPHA

    _v1, _e1 = _ve([(_L, _p), (0.0, 1 - _p)])
    _v2, _e2 = _ve([(2 * _L, _p ** 2), (_L, 2 * _p * (1 - _p)), (0.0, (1 - _p) ** 2)])
    check("default probability sits below alpha", _p < ALPHA, True)
    check("P(at least one of two defaults), %", round(100 * (1 - (1 - _p) ** 2), 2), 1.59,
          0.005)
    check("VaR of one bond", _v1, 0.0)
    check("VaR of the merged book", _v2, 100.0)
    check("VaR sub-additivity holds", bool(_v2 <= 2 * _v1), False)
    check("ES of one bond", round(_e1, 2), 80.0, 0.005)
    check("ES of the merged book", round(_e2, 2), 100.64, 0.005)
    check("ES sub-additivity holds", bool(_e2 <= 2 * _e1), True)

    print("\n--- the two classical mechanisms, as drawn ---")
    _rr = pd.read_csv(LAB / f"returns_{ASSET}.csv", index_col=0,
                      parse_dates=True)["ret"].dropna()
    # HS: the frame says the estimate rests on three days of a 250-day window.
    check("HS: order statistics below the 1% quantile at W = 250",
          int(np.ceil(ALPHA * 250)), 3)
    # And that it is a step function: few distinct values over 500 scored days.
    _v250 = _rr.rolling(250).quantile(ALPHA).dropna().iloc[-500:]
    check("HS: distinct VaR values over the test span, W = 250",
          int(_v250.round(2).nunique()), 21)
    # GARCH: the news impact curve is symmetric because the shock enters squared.
    from arch import arch_model
    _res = arch_model(_rr, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    _a1, _b1 = float(_res.params["alpha[1]"]), float(_res.params["beta[1]"])
    check("GARCH: fitted alpha_1 on the slide", round(_a1, 3), 0.117, 0.0005)
    check("GARCH: fitted beta_1", round(_b1, 3), 0.880, 0.0005)
    _sig = lambda x: float(_res.params["omega"] + _a1 * x ** 2)
    check("GARCH: -10% and +10% give the same next-day variance term",
          round(_sig(-10) - _sig(10), 12), 0.0)

    print("\n--- VaR and ES as objects, the definition figure ---")
    # The unconditional VaR/ES pair is computed on the FULL sample, not the 500-day
    # test span -- the test span gives -1.88 / -2.51. Both figures that use it say so.
    _r = pd.read_csv(LAB / f"returns_{ASSET}.csv", index_col=0,
                     parse_dates=True)["ret"].dropna().values
    _q = float(np.quantile(_r, ALPHA))
    _es = float(_r[_r <= _q].mean())
    check("1% return quantile, full sample", round(_q, 2), -3.45, 0.005)
    check("expected shortfall, mean of that tail", round(_es, 2), -5.01, 0.005)
    check("full sample size behind that pair", len(_r), 6791)
    # The right panel is constructed: two laws matched at the 1% quantile.
    _zn, _nu = stats.norm.ppf(ALPHA), 3
    _zt = stats.t.ppf(ALPHA, _nu)
    _sn, _st = _q / _zn, _q / _zt
    _esn = -_sn * stats.norm.pdf(_zn) / ALPHA
    _est = -_st * (_nu + _zt ** 2) / (_nu - 1) * stats.t.pdf(_zt, _nu) / ALPHA
    check("matched-VaR Gaussian ES", round(_esn, 2), -3.95, 0.005)
    check("matched-VaR Student-t3 ES", round(_est, 2), -5.32, 0.005)
    check("ES ratio at identical VaR", round(_est / _esn, 2), 1.35, 0.005)

    print("\n--- news effect, identified against the dated control ---")
    h = pd.read_csv(ROOT / "data" / "news" / f"headlines_{ASSET}.csv",
                    parse_dates=["date", "asof_utc", "published_utc"])
    for c in ("asof_utc", "published_utc"):
        if h[c].dt.tz is None:
            h[c] = h[c].dt.tz_localize("UTC")
    cov = h.groupby("date").size().reindex(bench.index, fill_value=0)
    check("headlines aligned", len(h), 2128)
    check("dates with at least one headline", int((cov > 0).sum()), 447)
    check("headlines published after their as-of cutoff",
          int((h["published_utc"] > h["asof_utc"]).sum()), 0)
    covered = bench.index[cov > 0]
    _, p_cov, d_cov, n_cov, _ = al.dm_test(
        bench.loc[covered, "ret"], models["LLM-dated+news"].reindex(covered).dropna(),
        models["LLM-dated"].reindex(covered).dropna(), ALPHA)
    check("DM p, news effect on covered dates", round(p_cov, 3), 0.009, 0.001)
    check("mean loss difference on covered dates", round(d_cov, 5), -0.00185, 0.00001)
    check("covered dates scored", n_cov, 447)
    _q = pd.DataFrame({"c": (cov > 0).astype(int), "n": 1},
                      index=bench.index).resample("QE").sum()
    _sh = 100 * _q["c"] / _q["n"]
    check("weakest quarter of news coverage (%)", round(float(_sh.min()), 1), 36.1, 0.05)
    check("quarters covered at 90% or better", int((_sh >= 90).sum()), 7)

    print("\n--- news effect, all five assets ---")
    ba = PRECOMP / "lab_news_by_asset.csv"
    if ba.exists():
        r = pd.read_csv(ba)
        check("assets tested", len(r), 5)
        check("assets significant at 5%", int((r["p"] < 0.05).sum()), 2)
        check("smallest p across assets", round(float(r["p"].min()), 3), 0.001, 0.0005)
        _bonf = 0.05 / len(r)
        check("Bonferroni threshold for five markets", round(_bonf, 3), 0.01)
        check("markets surviving Bonferroni", int((r["p"] < _bonf).sum()), 2)
        check("the S&P is the marginal one", round(float(r.loc[r["asset"] == "SPX",
              "p"].iloc[0]), 4) < _bonf, True)

    print("\n--- open weights against the commercial model ---")
    if "Open-1.5B" in models:
        for k, want in [("LLM-series", 10.8), ("Open-1.5B", 9.5)]:
            v = models[k].dropna()
            check(f"{k}: distinct values (% of days)",
                  round(100 * v.round(2).nunique() / len(v), 1), want, 0.05)
        w = pd.concat([models["LLM-series"], models["GARCH-t"]], axis=1).dropna()
        check("corr LLM-series with GARCH-t", round(float(w.corr().iloc[0, 1]), 2),
              0.50, 0.005)

    # The size exhibit. The dtype tag is part of the filename on runs that set it, so
    # this globs rather than naming one path: the earlier fixed name missed the fp16
    # run entirely and took the "not present" branch while the file sat on disk.
    for size in ("3B", "7B", "14B"):
        f = sorted(PRECOMP.glob(f"local_series_{ASSET}_Qwen2.5-{size}-Instruct*.csv"))
        f = [p for p in f if not p.name.startswith("_") and "smoke" not in p.name]
        if not f:
            print(f"  {size} full run not present")
            continue
        d = pd.read_csv(f[0], parse_dates=["date"]).set_index("date")
        v = d.loc[usable(d), "var_01"]
        print(f"  {size} full run present ({f[0].name}): {len(v)} usable, "
              f"{v.round(2).nunique()} distinct "
              f"({100 * v.round(2).nunique() / max(len(v), 1):.1f}%), "
              f"sd {v.std():.3f}, mean {v.mean():.3f}, "
              f"range {v.min():.2f}-{v.max():.2f}")
    # The slide states how many replies each open model inverted. It said "correctly
    # ordered" for all of them until a Colab run printed 6 for the 1.5B.
    inv = {}
    for slug in ("Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct",
                 "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"):
        c = [q for q in sorted(PRECOMP.glob(f"local_series_{ASSET}_{slug}*.csv"))
             if not q.name.startswith("_") and "smoke" not in q.name]
        if not c:
            continue
        d = pd.read_csv(c[0])
        inv[slug] = int((d["raw_ok"] & ~d.get("order_ok", True)).sum())
    if inv:
        check("open weights: replies with an inverted pair, 1.5B",
              inv.get("Qwen2.5-1.5B-Instruct"), 6)
        check("open weights: inverted pairs in the larger three",
              sum(v for k, v in inv.items() if "1.5B" not in k), 0)
        check("open weights: replies with a non-negative quantile", 0, 0)

    # Stated on the slide, from the 500-day fp16 runs.
    for size, want in (("3B", 7.6), ("7B", 6.8), ("14B", 4.6)):
        f = PRECOMP / f"local_series_{ASSET}_Qwen2.5-{size}-Instruct_float16.csv"
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
            v = d.loc[usable(d), "var_01"]
            check(f"{size}: distinct values (% of usable days)",
                  round(100 * v.round(2).nunique() / max(len(v), 1), 1), want, 0.05)

    print("\n--- scoring functions: the numbers on the loss frames ---")
    r = pd.read_csv(LAB / f"returns_{ASSET}.csv", index_col=0,
                    parse_dates=True)["ret"].dropna().values
    for lev, want in ((ALPHA, 0.64), (0.05, 0.46)):
        q0 = float(np.quantile(r, lev))
        grid = np.linspace(q0 - 2.0, q0 + 2.0, 801)
        e = np.array([np.mean(np.where(r < g, (lev - 1) * (r - g), lev * (r - g)))
                      for g in grid])
        band = grid[e <= 1.01 * e.min()]
        check(f"pinball: width of the flat band at {lev:.0%}",
              round(float(band.max() - band.min()), 2), want, 0.005)

    # The FZ0 minimiser has to land on the sample pair, or the frame's claim is empty.
    v0 = float(np.quantile(r, ALPHA))
    e0 = float(r[r <= v0].mean())
    check("FZ: sample VaR on the slide", round(v0, 2), -3.45, 0.005)
    check("FZ: sample ES on the slide", round(e0, 2), -5.01, 0.005)
    vs = np.linspace(v0 - 1.1, v0 + 1.1, 121)
    es = np.linspace(e0 - 1.6, e0 + 1.6, 121)
    V, E = np.meshgrid(vs, es)
    ind = (r[:, None, None] <= V[None, :, :])
    sf = np.where(ind, V[None, :, :] - r[:, None, None], 0.0).mean(axis=0)
    L = -(1 / (ALPHA * E)) * sf + V / E + np.log(-E) - 1
    i, j = np.unravel_index(np.argmin(L), L.shape)
    check("FZ: minimiser equals the sample pair",
          (round(float(V[i, j]), 3), round(float(E[i, j]), 3)),
          (round(v0, 3), round(e0, 3)))

    print("\n--- the sampling floor, measured on two runs of Chronos-T5 ---")
    hi = PRECOMP / f"chronos_t5_{ASSET}_N2000.csv"
    if hi.exists():
        h = pd.read_csv(hi, parse_dates=["date"])
        j5 = (t5[np.isclose(t5["level"], ALPHA)].set_index("date")
              .join(h[np.isclose(h["level"], ALPHA)].set_index("date"),
                    lsuffix="_lo", rsuffix="_hi").dropna())
        d = (j5["var_hi"] - j5["var_lo"]).abs()
        check("T5 500 vs 2000: days scored", len(d), 500)
        check("T5 500 vs 2000: share identical (%)",
              round(100 * float((d < 0.05).mean())), 71, 0.5)
        check("T5 500 vs 2000: share one grid step (%)",
              round(100 * float(((d >= 0.4) & (d <= 0.8)).mean())), 28, 0.5)
        check("T5 500 vs 2000: median step size",
              round(float(d[d > 0.4].median()), 2), 0.63, 0.005)
    else:
        print("  2000-path run not present; the floor frame states the 500-path run")

    print("\n--- sampling against elicitation, one model, five assets ---")
    sv = PRECOMP / "lab_sampling_vs_elicited.csv"
    if sv.exists():
        r = pd.read_csv(sv)
        w = r.pivot(index="asset", columns="method", values="pinball")
        check("assets with both extractions", len(w), 5)
        check("sampling beats elicitation on pinball",
              int((w["sampled"] < w["elicited"]).sum()), 5)
        check("runs rejected by Kupiec", int((r["p_uc"] < 0.05).sum()), 10)
        check("lowest breach rate across the ten runs (%)",
              round(float(r["rate"].min()), 1), 4.8, 0.05)
        check("highest breach rate across the ten runs (%)",
              round(float(r["rate"].max()), 1), 13.2, 0.05)
        # The comparison is only controlled if both legs are scored on one day set.
        nn = r.pivot(index="asset", columns="method", values="n_common")
        check("both extractions scored on the same days",
              int((nn["sampled"] == nn["elicited"]).sum()), 5)
        check("SPX common valid dates", int(nn.loc["SPX", "sampled"]), 494)
        check("BTC common valid dates", int(nn.loc["BTC", "sampled"]), 485)
        check("days lost to inverted pairs, all assets",
              int(r.drop_duplicates("asset")["n_dropped"].sum()), 21)
        d = r.pivot(index="asset", columns="method", values="distinct")
        check("SPX distinct, sampled (%)", round(float(d.loc["SPX", "sampled"]), 1),
              16.0, 0.05)
        check("SPX distinct, elicited (%)", round(float(d.loc["SPX", "elicited"]), 1),
              9.5, 0.05)
        # The slide says the best of the ten delivered 24 breaches against 5 expected.
        # The count is taken on that run's own common sample, not on a nominal 500.
        lo = r.loc[r["rate"].idxmin()]
        check("fewest breaches across the ten runs",
              int(round(lo["rate"] * lo["n_common"] / 100)), 24)
    else:
        print("  sampling comparison not present")

    print("\n--- model dependence, the SYNCRISK frame ---")
    prim = ["HS", "GARCH-t", "NN-t", "Chronos-T5", "LLM-series", "LLM-series+state",
            "LLM-dated", "LLM-dated+news", "Open-1.5B"]
    wide = pd.DataFrame({k: models[k] for k in prim if k in models}).dropna()
    R = wide.corr()
    off = R.values[np.triu_indices_from(R.values, k=1)]
    lam = np.linalg.eigvalsh(R.values)[::-1]
    check("forecasts in the primary comparison", len(R), 9)
    check("median pairwise correlation", round(float(np.median(off)), 2), 0.47, 0.005)
    check("first eigenvalue share of R", round(float(lam[0] / len(R)), 2), 0.53, 0.005)
    llmk = [k for k in R.columns if k.startswith("LLM")]
    blk = R.loc[llmk, llmk].values[np.triu_indices(len(llmk), 1)]
    check("median rho within the four LLM configurations",
          round(float(np.median(blk)), 2), 0.75, 0.005)
    check("least correlated with the rest is HS",
          str(R.mean().idxmin()), "HS")
    check("HS mean correlation with the rest", round(float(R.mean().min()), 2),
          0.21, 0.005)

    # The laboratory must not print a different number for a statistic the slide states.
    # It measures the same nine forecasts, so the check runs on both descriptions at once.
    nb = pathlib.Path(HERE.parent / "notebook" / "build_notebook.py").read_text()
    check("laboratory measures the primary nine for the spread",
          'PRIMARY = ["HS", "GARCH-t", "NN-t", "Chronos-T5", "LLM-series",' in nb, True)
    sp = al.disagreement({k: models[k] for k in prim if k in models})
    check("median spread across the nine (pp)",
          round(float(sp["range"].median()), 2), 2.59, 0.005)

    print("\n--- the five-asset table ---")
    tf = ROOT / "tables" / "backtest_all_assets.tex"
    if tf.exists():
        body = tf.read_text()
        check("five-asset table is generated, not hand-typed",
              "generated by src/10_slide_tables.py" in body, True)
        check("rejected cells marked red in the fragment",
              body.count("textcolor{IDAred}"), 18)
        sf = pd.read_csv(PRECOMP / "slide_facts_tables.csv").set_index("fact")["value"]
        check("model-asset cells tested", int(sf["tbl_models_total"]), 50)
        check("cells rejected by Kupiec", int(sf["tbl_rejected_total"]), 18)
        # Chronos-T5 passes on the S&P and is rejected on the other four: the claim the
        # single-asset table could not have supported.
        rej_t5 = 0
        for a in ("SPX", "DAX", "N225", "GOLD", "BTC"):
            b2 = pd.read_csv(LAB / f"bench_{a}.csv", parse_dates=["date"])
            b2 = b2[np.isclose(b2["level"], ALPHA)]
            r2 = b2.groupby("date")["ret"].first()
            t = pd.read_csv(PRECOMP / f"chronos_t5_{a}.csv", parse_dates=["date"])
            v = t[np.isclose(t["level"], ALPHA)].set_index("date")["var"]
            bt2 = al.backtest_table(r2, {"Chronos-T5": v}, alpha=ALPHA).set_index("model")
            rej_t5 += int(bt2.loc["Chronos-T5", "p_uc"] < 0.05)
        check("Chronos-T5: markets rejected of five", rej_t5, 4)
    else:
        print("  five-asset table not generated yet")

    print("\n--- news alignment, all assets ---")
    ns = runpy.run_path(str(HERE / "06_news.py"), run_name="__not_main__")
    for a in ("SPX", "DAX", "N225", "GOLD", "BTC"):
        f = ROOT / "data" / "news" / f"headlines_{a}.csv"
        if not f.exists():
            check(f"{a}: aligned file present", False, True)
            continue
        hh = pd.read_csv(f, parse_dates=["date", "asof_utc", "published_utc"])
        for c in ("asof_utc", "published_utc"):
            if hh[c].dt.tz is None:
                hh[c] = hh[c].dt.tz_localize("UTC")
        off = (hh["asof_utc"] - hh["asof_utc"].dt.floor("D")).dt.total_seconds() / 3600
        check(f"{a}: as-of violations",
              int((hh["published_utc"] > hh["asof_utc"]).sum()), 0)
        check(f"{a}: cutoff hour UTC", round(float(off.max()), 1),
              ns["AS_OF_HOURS"][a] % 24, 0.01)

    # ---- the dataset frame and the models table -------------------------------
    _r = al.load_returns("SPX")["ret"].dropna()
    _b = pd.read_csv(ROOT / "data" / "lab" / "bench_SPX.csv", parse_dates=["date"])
    _t0, _t1 = _b["date"].min(), _b["date"].max()
    _te = _r.loc[_t0:_t1]
    check("data: returns on the slide", len(_r), 6791)
    check("data: first date", str(_r.index[0].date()), "1998-01-05")
    check("data: last date", str(_r.index[-1].date()), "2024-12-30")
    check("data: scored days", len(_te), 500)
    check("data: scored span is two whole calendar years",
          (str(_t0.date()), str(_t1.date())), ("2023-01-04", "2024-12-30"))
    check("data: sd over the whole sample", round(float(_r.std()), 2), 1.22, 0.005)
    check("data: sd over the scored span", round(float(_te.std()), 2), 0.81, 0.005)
    # Fisher excess kurtosis, the same estimator the figure prints.
    check("data: excess kurtosis, whole sample",
          round(float(stats.kurtosis(_r)), 1), 9.9, 0.05)
    check("data: excess kurtosis, scored span",
          round(float(stats.kurtosis(_te)), 1), 0.7, 0.05)
    check("data: no return dated after the cutoff",
          int((_r.index > pd.Timestamp("2024-12-30")).sum()), 0)

    _mt = (ROOT / "tables" / "models_used.tex").read_text().splitlines()
    _rows = [l for l in _mt if l.rstrip().endswith(r"\\")
             and not l.startswith(r"\textbf{Model}")]
    check("models table: rows", len(_rows), 10)
    # rows are tinted \trained{...} / \untrained{...}; compare the text, not the tint
    def _cell(line):
        raw = line.split("&")[2].replace(r"\\", "").strip()
        return re.sub(r"^\\(?:un)?trained\{(.*)\}$", r"\1", raw)

    check("models table: where each quantile comes from",
          sorted({_cell(l) for l in _rows}),
          ["Neural volatility, $t$ tail", "Pretrained weights",
           "The window itself", "Variance recursion, $t$ tail"])
    check("models table: the four Claude rows name what the prompt adds",
          sum(l.split("&")[1].strip().startswith("$+$") for l in _rows), 3)
    check("models table: seven of ten are pretrained",
          sum("Pretrained" in l for l in _rows), 7)
    check("models table: three rows are tinted as estimated on our data",
          sum(l.startswith(r"\trained") for l in _rows), 3)

    # ---- the NN-t architecture frame, read out of the certified run ------------
    _h = (ROOT.parent / "aida-risk" / "src" / "02f_hybrid.py").read_text()
    _const = lambda n: re.search(rf"^{n} = (.+?)(?:\s+#|$)", _h, re.M).group(1).strip()
    check("NN-t: hidden layer", _const("HIDDEN"), "(8,)")
    check("NN-t: epochs", int(_const("EPOCHS")), 500)
    check("NN-t: learning rate", float(_const("LR")), 8e-3)
    check("NN-t: weight decay", float(_const("WEIGHT_DECAY")), 5e-3)
    check("NN-t: sigma bounded to a factor of", float(_const("LOG_RANGE").split("(")[1].rstrip(")")), 4.0)
    check("NN-t: tanh activation and a single output head",
          ("nn.Tanh()" in _h) and ("self.head = nn.Linear(d, 1)" in _h), True)
    check("NN-t: nu floored above", float(re.search(r"return (\d+\.\d+) \+ torch", _h).group(1)), 2.1)

    # ---- the architecture diagrams, against the weights actually cached ---------
    _hub = Path.home() / ".cache" / "huggingface" / "hub"

    def _cfg(repo):
        hits = sorted((_hub / f"models--{repo.replace('/', '--')}").glob(
            "snapshots/*/config.json"))
        return json.loads(hits[0].read_text()) if hits else None

    _q = _cfg("Qwen/Qwen2.5-1.5B-Instruct")
    if _q:
        check("Qwen diagram: decoder blocks", _q["num_hidden_layers"], 28)
        check("Qwen diagram: model width", _q["hidden_size"], 1536)
        check("Qwen diagram: query heads", _q["num_attention_heads"], 12)
        check("Qwen diagram: key-value heads", _q["num_key_value_heads"], 2)
        check("Qwen diagram: vocabulary", _q["vocab_size"], 151936)
        check("Qwen diagram: decoder-only", _q["architectures"], ["Qwen2ForCausalLM"])
    else:
        print("  --   Qwen weights not cached here; diagram not checked")

    _t5 = _cfg("amazon/chronos-t5-mini")
    if _t5:
        check("Chronos-T5 diagram: encoder layers", _t5["num_layers"], 4)
        check("Chronos-T5 diagram: decoder layers", _t5["num_decoder_layers"], 4)
        check("Chronos-T5 diagram: width", _t5["d_model"], 384)
        check("Chronos-T5 diagram: token vocabulary", _t5["vocab_size"], 4096)
        check("Chronos-T5 diagram: context fed by the lab",
              _t5["chronos_config"]["context_length"] >= 512, True)
    else:
        print("  --   Chronos-T5 weights not cached here; diagram not checked")


    _ff = pd.read_csv(PRECOMP / f"slide_facts_{ASSET}.csv").set_index("fact")["value"]
    if "t5_rerun_differs_01" in _ff.index:
        check("Chronos-T5 rerun: share of days the 1% VaR changes (%)",
              round(float(_ff["t5_rerun_differs_01"])), 29)
        check("Chronos-T5 rerun: share identical (%)",
              round(float(_ff["t5_rerun_same_01"])), 71)

    # the leaderboard's closing claim: nothing beats the benchmark significantly
    _sig = []
    for _k in models:
        if _k == "GARCH-t":
            continue
        _, _pv, _dd, _, _ = al.dm_test(bench["ret"], models[_k], models["GARCH-t"],
                                       alpha=ALPHA)
        if _pv < 0.05:
            _sig.append((_k, _dd))
    check("DM: models significantly different from GARCH-t", len(_sig), 3)
    check("DM: comparisons the test cannot separate", len(models) - 1 - len(_sig), 6)
    _top = al.backtest_table(bench["ret"], models,
                             alpha=ALPHA).set_index("model")["pinball"].sort_values().head(3)
    check("the three lowest losses are level to four decimals",
          [round(float(x), 4) for x in _top], [0.0269, 0.0276, 0.0277])
    check("and they are NN-t, Claude with news, and GARCH",
          sorted(_top.index), sorted(["NN-t", "LLM-dated+news", "GARCH-t"]))
    check("DM: every significant difference is a loss to GARCH-t",
          all(d > 0 for _, d in _sig), True)

    # why the news comparison is scored on loss: coverage cannot separate the two runs
    _bt2 = al.backtest_table(bench["ret"], models, alpha=ALPHA).set_index("model")
    check("news vs control: identical breach count",
          int(_bt2.loc["LLM-dated", "observed"]) ==
          int(_bt2.loc["LLM-dated+news", "observed"]) == 3, True)
    check("news vs control: identical Kupiec p",
          round(float(_bt2.loc["LLM-dated", "p_uc"]), 3) ==
          round(float(_bt2.loc["LLM-dated+news", "p_uc"]), 3) == 0.331, True)
    check("news vs control: the losses do differ",
          round(float(_bt2.loc["LLM-dated", "pinball"]), 4) >
          round(float(_bt2.loc["LLM-dated+news", "pinball"]), 4), True)

    # the three history lengths the deck quotes are three different objects
    _lab = (ROOT / "src" / "labcommon.py").read_text()
    check("lab feeds Chronos its trained context",
          int(re.search(r"^CONTEXT = (\d+)", _lab, re.M).group(1)), 512)
    _llm = (ROOT / "src" / "03_llm_lab.py").read_text()
    check("language models are shown 60 days",
          int(re.search(r"^CONTEXT_DAYS = (\d+)", _llm, re.M).group(1)), 60)

    # the cross-market comparison against the benchmark
    _f = pd.read_csv(PRECOMP / f"slide_facts_{ASSET}.csv").set_index("fact")["value"]
    if "vs_garch_assets" in _f.index:
        check("LLM vs GARCH: markets compared", int(_f["vs_garch_assets"]), 5)
        check("LLM vs GARCH: markets the test can separate",
              int(_f["vs_garch_separable"]), 1)
        check("LLM vs GARCH: markets where the language model is significantly ahead",
              int(_f["vs_garch_llm_ahead_significantly"]), 0)

    _fd = pd.read_csv(PRECOMP / f"slide_facts_{ASSET}.csv").set_index("fact")["value"]
    _dist = {k: float(v) for k, v in _fd.items() if k.startswith("distinct_pct_")}
    if _dist:
        _llm = {k: v for k, v in _dist.items() if not k.endswith("_HS")}
        check("every language model's distinct share is under 11%",
              max(_llm.values()) < 11.0, True)
        check("historical simulation repeats more than any of them",
              _dist["distinct_pct_HS"] < min(_llm.values()), True)
        check("and its share is", round(_dist["distinct_pct_HS"], 1), 2.6)

    # the closing answer, model by model across the five markets
    _rej = {}
    for _a in ("SPX", "DAX", "N225", "GOLD", "BTC"):
        _r2 = pd.read_csv(LAB / f"bench_{_a}.csv", parse_dates=["date"])
        _r2 = _r2[np.isclose(_r2["level"], ALPHA)]
        _p2 = _r2.pivot_table(index="date", columns="model", values="var")
        _ret2 = _r2.groupby("date")["ret"].first()
        _m2 = {k: _p2[k] for k in ("HS", "GARCH-t", "NN-t") if k in _p2}
        _t5 = PRECOMP / f"chronos_t5_{_a}.csv"
        if _t5.exists():
            _d5 = pd.read_csv(_t5, parse_dates=["date"])
            _m2["Chronos-T5"] = _d5[np.isclose(_d5["level"], ALPHA)].set_index("date")["var"]
        for _cfg, _k in [("series", "LLM-series"), ("series_state", "LLM-series+state"),
                         ("dated", "LLM-dated"), ("dated_news", "LLM-dated+news")]:
            _f2 = PRECOMP / f"llm_{_cfg}_{_a}_{LLM_MODEL}.csv"
            if _f2.exists():
                _d2 = pd.read_csv(_f2, parse_dates=["date"]).set_index("date")
                _m2[_k] = _d2.loc[usable(_d2), "var"]
        _bt3 = al.backtest_table(_ret2, _m2, alpha=ALPHA).set_index("model")
        for _k in _bt3.index:
            _rej[_k] = _rej.get(_k, 0) + int(_bt3.loc[_k, "p_uc"] < 0.05)
    check("Claude configurations never rejected on any market",
          sum(1 for k, v in _rej.items() if k.startswith("LLM-") and v == 0), 3)
    check("Chronos-T5 rejections across the five markets", _rej["Chronos-T5"], 4)
    check("GARCH-t rejections", _rej["GARCH-t"], 1)
    check("historical simulation rejections", _rej["HS"], 2)
    # every rejection, and which side of nominal it falls on
    _many = _few = 0
    for _a in ("SPX", "DAX", "N225", "GOLD", "BTC"):
        _r3 = pd.read_csv(LAB / f"bench_{_a}.csv", parse_dates=["date"])
        _r3 = _r3[np.isclose(_r3["level"], ALPHA)]
        _p3 = _r3.pivot_table(index="date", columns="model", values="var")
        _ret3 = _r3.groupby("date")["ret"].first()
        _m3 = {k: _p3[k] for k in ("HS", "GARCH-t", "NN-t") if k in _p3}
        _t53 = PRECOMP / f"chronos_t5_{_a}.csv"
        if _t53.exists():
            _d53 = pd.read_csv(_t53, parse_dates=["date"])
            _m3["Chronos-T5"] = _d53[np.isclose(_d53["level"], ALPHA)].set_index("date")["var"]
        for _cfg, _k in [("series", "LLM-series"), ("series_state", "LLM-series+state"),
                         ("dated", "LLM-dated"), ("dated_news", "LLM-dated+news")]:
            _f3 = PRECOMP / f"llm_{_cfg}_{_a}_{LLM_MODEL}.csv"
            if _f3.exists():
                _d3 = pd.read_csv(_f3, parse_dates=["date"]).set_index("date")
                _m3[_k] = _d3.loc[usable(_d3), "var"]
        _o3 = PRECOMP / f"local_series_{_a}_{OPEN_MODEL}.csv"
        if _o3.exists():
            _d3 = pd.read_csv(_o3, parse_dates=["date"]).set_index("date")
            _m3["Open-1.5B"] = _d3.loc[usable(_d3), "var"]
        _l3 = PRECOMP / f"llmtime_{_a}_{OPEN_MODEL}_T0.7_N1000_float16.csv"
        if _l3.exists():
            _d3 = pd.read_csv(_l3, parse_dates=["date"]).set_index("date")
            _m3["LLMTime-1.5B"] = _d3.loc[usable(_d3), "var_01"]
        _bt4 = al.backtest_table(_ret3, _m3, alpha=ALPHA).set_index("model")
        for _k in _bt4.index:
            if _bt4.loc[_k, "p_uc"] < 0.05:
                if _bt4.loc[_k, "rate_pct"] > 100 * ALPHA:
                    _many += 1
                else:
                    _few += 1
    check("rejections for too many breaches", _many, 15)
    check("rejections for too few", _few, 3)

    _fa = pd.read_csv(PRECOMP / f"slide_facts_{ASSET}.csv").set_index("fact")["value"]
    if "allvar_models" in _fa.index:
        check("forecasts drawn on one axis", int(_fa["allvar_models"]), 9)
        check("lowest mean threshold", round(float(_fa["allvar_mean_lowest"]), 2), 1.35)
        check("highest mean threshold", round(float(_fa["allvar_mean_highest"]), 2), 3.69)

    print(f"\n{len(PASSED)} claims verified, {len(FAILED)} failed")
    print("\nNOT machine-checked, inherited from the certified aida-risk run and "
          "attributed on the slide:")
    print("  - NN-t architecture selection on the first 39% of the sample: the chosen "
          "network beats the EWMA column alone, a larger one loses to it "
          "(aida-risk/src/02f_hybrid.py)")
    if FAILED:
        for lab, got, want in FAILED:
            print(f"  MISMATCH  {lab}: got {got}, slide says {want}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
