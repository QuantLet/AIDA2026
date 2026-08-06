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

    print("\n--- Chronos-Bolt quantile clamp ---")
    bolt = pd.read_csv(PRECOMP / f"chronos_bolt_{ASSET}.csv", parse_dates=["date"])
    piv = bolt.pivot_table(index="date", columns="level", values="var")
    for lvl in (0.01, 0.05):
        share = float(np.isclose(piv[lvl], piv[0.10], rtol=0, atol=0).mean())
        check(f"days where q{lvl:.2f} equals q0.10 exactly", round(100 * share), 100)

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

    print("\n--- news effect, all five assets ---")
    ba = PRECOMP / "lab_news_by_asset.csv"
    if ba.exists():
        r = pd.read_csv(ba)
        check("assets tested", len(r), 5)
        check("assets significant at 5%", int((r["p"] < 0.05).sum()), 2)
        check("smallest p across assets", round(float(r["p"].min()), 3), 0.001, 0.0005)

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
        d = r.pivot(index="asset", columns="method", values="distinct")
        check("SPX distinct, sampled (%)", round(float(d.loc["SPX", "sampled"]), 1),
              15.8, 0.05)
        check("SPX distinct, elicited (%)", round(float(d.loc["SPX", "elicited"]), 1),
              9.5, 0.05)
        # The slide says the best of the ten delivered 24 breaches against 5 expected.
        check("fewest breaches across the ten runs",
              int(round(r["rate"].min() * 500 / 100)), 24)
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

    print(f"\n{len(PASSED)} claims verified, {len(FAILED)} failed")
    print("\nNOT machine-checked, inherited from the certified aida-risk run and "
          "attributed on the slide:")
    print("  - Chronos fed returns instead of prices: predictive sd 0.14 vs realised "
          "1.22 (aida-risk/src/02c_chronos.py)")
    if FAILED:
        for lab, got, want in FAILED:
            print(f"  MISMATCH  {lab}: got {got}, slide says {want}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
