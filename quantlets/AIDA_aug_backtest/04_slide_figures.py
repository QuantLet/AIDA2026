"""Stage 04 — figures for the August lecture deck.

Presentation scale: larger fonts, transparent background (the slides sit on a gradient),
IDA colours from labcommon. Every figure is exported as PDF for LaTeX and 300 dpi PNG
for review, and every number in it comes from data/lab or precomputed.

A figure whose inputs are missing is skipped with a message rather than drawn from
partial data, so a half-finished run cannot put a half-finished chart on a slide.

Usage:  python src/04_slide_figures.py [ASSET]
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebook"))
from labcommon import (ALPHA, ALPHA_SECONDARY, ASSETS, FIGURES, HS_WINDOW,  # noqa: E402
                       LAB, MODEL_COLORS, MODEL_LABELS, PRECOMP, ROOT, SEED,
                       legend_below)
import aidalab as al  # noqa: E402

ASSET = (sys.argv[1] if len(sys.argv) > 1 else "SPX").upper()
LLM_MODEL = "claude-haiku-4-5"
GREY = "#6E6E6E"

mpl.rcParams.update({
    "figure.figsize": (10, 4.6),
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "savefig.bbox": "tight",
    "savefig.transparent": True,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
})

SLIDES = FIGURES / "slides"
SLIDES.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(SLIDES / f"{name}.pdf")
    fig.savefig(SLIDES / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  -> slides/{name}.pdf")


# ---------------------------------------------------------------------------
def load():
    rets = pd.read_csv(LAB / f"returns_{ASSET}.csv", index_col=0, parse_dates=True)
    b = pd.read_csv(LAB / f"bench_{ASSET}.csv", parse_dates=["date"])
    b = b[np.isclose(b["level"], ALPHA)]
    bench = b.pivot_table(index="date", columns="model", values="var")
    bench.insert(0, "ret", b.groupby("date")["ret"].first())
    bench.columns.name = None

    models = {m: bench[m] for m in ("HS", "GARCH-t", "NN-t") if m in bench}

    bolt = pd.read_csv(PRECOMP / f"chronos_bolt_{ASSET}.csv", parse_dates=["date"]) \
        if (PRECOMP / f"chronos_bolt_{ASSET}.csv").exists() else None

    p = PRECOMP / f"chronos_t5_{ASSET}.csv"
    if p.exists():
        t5 = pd.read_csv(p, parse_dates=["date"])
        models["Chronos-T5"] = t5[np.isclose(t5["level"], ALPHA)].set_index("date")["var"]

    for cfg, key in [("series", "LLM-series"), ("series_state", "LLM-series+state"),
                     ("dated", "LLM-dated"), ("dated_news", "LLM-dated+news")]:
        f = PRECOMP / f"llm_{cfg}_{ASSET}_{LLM_MODEL}.csv"
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
            models[key] = d.loc[_usable(d), "var"]

    # Open weights, the leg a student can run without a commercial key. Runs that set
    # --dtype carry it in the filename, so this globs: a fixed name silently missed the
    # fp16 size runs while the files sat in precomputed/.
    for size in ("1.5B", "3B", "7B", "14B"):
        f = _local_run(f"Qwen2.5-{size}-Instruct")
        if f:
            d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
            models[f"Open-{size}"] = d.loc[_usable(d), "var"]
    return rets, bench, models, bolt


def _local_run(slug):
    """The finished open-weights run for a model, dtype tag or not. Checkpoints
    (leading underscore) and smoke files are not runs."""
    c = [p for p in sorted(PRECOMP.glob(f"local_series_{ASSET}_{slug}*.csv"))
         if not p.name.startswith("_") and "smoke" not in p.name]
    return c[0] if c else None


def _usable(d):
    """Rows a forecast may be read from: parsed, correctly signed, correctly ordered."""
    m = d["raw_ok"] & d["sign_ok"]
    if "order_ok" in d:
        m = m & d["order_ok"]
    return m


# ---------------------------------------------------------------------------
def fig_power(bench):
    """Why the lab evaluates 500 days and not 50."""
    def detectable(n, alpha=ALPHA, level=0.05):
        crit = stats.chi2.ppf(1 - level, 1)
        for p in np.arange(alpha, 0.6, 0.001):
            x = p * n
            ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
            ll1 = (n - x) * np.log(1 - p) + x * np.log(p)
            if -2 * (ll0 - ll1) > crit:
                return p
        return np.nan

    ns = np.array([25, 50, 75, 100, 150, 250, 500, 1000, 2000])
    ys = np.array([detectable(int(n)) for n in ns])

    fig, ax = plt.subplots()
    ax.plot(ns, 100 * ys, "o-", color=al.MAIN_BLUE, lw=2, ms=6)
    ax.axhline(100 * ALPHA, color=al.FOREST, lw=1.4, ls="--",
               label=f"nominal {ALPHA:.0%}")
    ax.axvline(500, color=al.IDA_RED, lw=1.4, ls=":", label="lab test span, 500 days")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Out-of-sample days")
    ax.set_ylabel("Smallest detectable\nbreach rate (%)")
    ax.set_title("A short backtest cannot reject a bad model")
    legend_below(ax, ncol=2)
    save(fig, "s_power")
    return {"detect_50": ys[1], "detect_500": ys[6]}


def fig_clamp(bolt):
    """Chronos-Bolt returns the 10% quantile when asked for 1% or 5%."""
    if bolt is None:
        print("  skip s_clamp: no bolt file")
        return {}
    piv = bolt.pivot_table(index="date", columns="level", values="var")
    lv = sorted(piv.columns)
    low = [x for x in lv if x <= 0.10]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4),
                                 gridspec_kw={"width_ratios": [1.55, 1]})

    for x in low:
        a1.plot(piv.index, piv[x], lw=2.2 if x == 0.10 else 1.2,
                color={0.01: al.IDA_RED, 0.05: "#E67E22", 0.10: "#8E44AD"}[x],
                ls={0.01: "-", 0.05: "--", 0.10: ":"}[x],
                label=f"requested {x:.0%}")
    a1.set_ylabel("VaR (%)")
    a1.set_title("Three requested levels, one line")
    legend_below(a1, ncol=3, y=-0.18, fontsize=10)

    share = [float(np.isclose(piv[x], piv[0.10], rtol=0, atol=0).mean()) for x in low]
    a2.bar([f"{x:.0%}" for x in low], [100 * s for s in share],
           color=[al.IDA_RED, "#E67E22", "#8E44AD"], width=0.6)
    a2.set_ylim(0, 108)
    a2.set_ylabel("Days identical to the\n10% output (%)")
    a2.set_xlabel("Requested quantile level")
    a2.set_title("Trained grid: 0.1 to 0.9")
    a2.grid(axis="x", visible=False)
    for i, s in enumerate(share):
        a2.text(i, 100 * s + 2, f"{100 * s:.0f}%", ha="center", fontsize=12)

    fig.tight_layout()
    save(fig, "s_clamp")
    return {"clamp_share_05": share[low.index(0.05)] if 0.05 in low else np.nan,
            "n_dates": len(piv)}


def fig_thresholds(bench, models):
    """One asset, one test span, four thresholds."""
    keys = [k for k in ("HS", "GARCH-t", "Chronos-T5", "LLM-series") if k in models]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(bench.index, bench["ret"], lw=0.7, color="#BBBBBB",
            label="Realised return", zorder=1)
    for k in keys:
        v = models[k].dropna()
        ax.plot(v.index, -v.values, lw=1.6, color=MODEL_COLORS.get(k, GREY),
                label=MODEL_LABELS.get(k, k), zorder=2)
    ax.set_ylabel("Daily return (%)")
    ax.set_title(f"{ASSETS[ASSET]['label']}: the same question, four answers")
    legend_below(ax, ncol=3, y=-0.16, fontsize=10)
    save(fig, "s_thresholds")
    return {}


def fig_backtest(bench, models):
    """Breach counts against the acceptance band, and the pinball ranking."""
    bt = al.backtest_table(bench["ret"], models, alpha=ALPHA)
    lb = al.leaderboard(bt, alpha=ALPHA)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    b = bt.sort_values("observed")
    n = int(b["n"].max())
    exp, se = ALPHA * n, np.sqrt(ALPHA * (1 - ALPHA) * n)
    a1.axvspan(exp - 1.96 * se, exp + 1.96 * se, color=al.FOREST, alpha=0.13,
               label="95% acceptance band")
    a1.axvline(exp, color=al.FOREST, lw=1.5, label=f"expected ({exp:.0f})")
    a1.barh([MODEL_LABELS.get(m, m) for m in b["model"]], b["observed"],
            color=[MODEL_COLORS.get(m, GREY) for m in b["model"]], height=0.62)
    a1.set_xlabel(f"Breaches in {n} days at {ALPHA:.0%}")
    a1.grid(axis="y", visible=False)
    legend_below(a1, ncol=2, y=-0.16, fontsize=10)
    a1.set_title("Coverage")

    d = lb.sort_values("pinball", ascending=False)
    a2.barh([MODEL_LABELS.get(m, m) for m in d["model"]], d["pinball"],
            color=[MODEL_COLORS.get(m, GREY) for m in d["model"]],
            hatch=["//" if c == "reject" else "" for c in d["CC"]], height=0.62)
    a2.set_xlabel("Mean pinball loss (lower is better)")
    a2.grid(axis="y", visible=False)
    a2.set_title("Ranking; hatched = conditional coverage rejected")

    fig.tight_layout()
    save(fig, "s_backtest")

    bt.to_csv(PRECOMP / f"lab_backtest_{ASSET}.csv", index=False)
    best = lb.iloc[0]
    facts = {"best_model": best["model"], "best_pinball": best["pinball"],
             "n_reject_cc": int((lb["CC"] == "reject").sum()), "n_models": len(lb),
             "n_days": n}

    # A ranking without a test on the loss differential is an ordering, not a finding.
    top = best["model"]
    for m in lb["model"][1:]:
        t, p, d, nn, lag = al.dm_test(bench["ret"], models[top], models[m], ALPHA)
        facts[f"dm_p_{top}_vs_{m}"] = round(p, 4)
        facts[f"dm_diff_{top}_vs_{m}"] = round(d, 5)
    al.dm_matrix(bench["ret"], models, ALPHA).round(4).to_csv(
        PRECOMP / f"lab_dm_{ASSET}.csv")
    return facts


def fig_disagreement(models):
    """The band between the most and least conservative model."""
    sp = al.disagreement(models)
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.fill_between(sp.index, -sp["max"], -sp["min"], color=al.MAIN_BLUE, alpha=0.18,
                    label="Range across models")
    ax.plot(sp.index, -sp["mean"], color=al.MAIN_BLUE, lw=1.5, label="Mean forecast")
    ax.set_ylabel("VaR threshold, return axis (%)")
    ax.set_title(f"Median spread {sp['range'].median():.2f}pp, "
                 f"median max/min {sp['ratio'].median():.2f}x, "
                 f"widest {sp['range'].max():.2f}pp")
    legend_below(ax, ncol=2, y=-0.18)
    save(fig, "s_disagreement")
    return {"median_range": sp["range"].median(), "median_ratio": sp["ratio"].median(),
            "max_range": sp["range"].max(), "max_range_date": sp["range"].idxmax()}


def fig_llm(bench, models):
    """What the LLM's number is actually tracking."""
    keys = [k for k in models if k.startswith("LLM")]
    if not keys or "HS" not in models:
        print("  skip s_llm: no LLM run")
        return {}
    wide = pd.DataFrame({k: models[k] for k in
                         [*keys, "HS", "GARCH-t"] if k in models}).dropna()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    a1.plot(bench.index, bench["ret"], lw=0.6, color="#CCCCCC", label="Realised return")
    for k in keys:
        v = models[k].dropna()
        a1.plot(v.index, -v.values, lw=1.5, color=MODEL_COLORS.get(k, GREY),
                label=MODEL_LABELS.get(k, k))
    a1.plot(models["GARCH-t"].index, -models["GARCH-t"].values, lw=1.3,
            color=MODEL_COLORS["GARCH-t"], label=MODEL_LABELS["GARCH-t"])
    a1.set_ylabel("Daily return (%)")
    a1.set_title("The LLM states a threshold")
    legend_below(a1, ncol=2, y=-0.20, fontsize=9)

    c = wide.corr()
    im = a2.imshow(c.values, cmap="Blues", vmin=0, vmax=1)
    a2.set_xticks(range(len(c)))
    a2.set_xticklabels(c.columns, rotation=35, ha="right", fontsize=9)
    a2.set_yticks(range(len(c)))
    a2.set_yticklabels(c.columns, fontsize=9)
    a2.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            a2.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if c.values[i, j] > 0.6 else "#333333")
    a2.set_title("Correlation of daily forecasts")
    fig.colorbar(im, ax=a2, fraction=0.045)

    fig.tight_layout()
    save(fig, "s_llm")

    out = {}
    for k in keys:
        out[f"corr_{k}_HS"] = float(wide[k].corr(wide["HS"]))
        out[f"corr_{k}_GARCH"] = float(wide[k].corr(wide["GARCH-t"]))
        out[f"mean_{k}"] = float(models[k].mean())
    return out


def fig_news(bench, models):
    """Does real text move the forecast, once the period is already revealed?

    The comparison is dated+news against dated, not against the anonymised series
    configuration: headlines reveal the period on their own, so pairing them with an
    anonymised control would confound "the text carries information" with "the model
    now knows which period this is".

    It is also run on the COVERED subset. On a date with no headline in its window the
    two prompts are identical by construction, so those days contribute exactly zero to
    the differential and only dilute the test.
    """
    need = {"LLM-dated", "LLM-dated+news"}
    if not need <= set(models):
        print("  skip s_news: dated / dated+news not generated")
        return {}

    hp = ROOT / "data" / "news" / f"headlines_{ASSET}.csv"
    cov = pd.read_csv(hp, parse_dates=["date"]).groupby("date").size()
    cov = cov.reindex(bench.index, fill_value=0)
    covered = bench.index[cov > 0]

    a, b = models["LLM-dated+news"], models["LLM-dated"]
    t_all, p_all, d_all, n_all, _ = al.dm_test(bench["ret"], a, b, ALPHA)
    t_cov, p_cov, d_cov, n_cov, _ = al.dm_test(
        bench.loc[covered, "ret"], a.reindex(covered).dropna(),
        b.reindex(covered).dropna(), ALPHA)

    # per-day pinball differential on the covered dates, accumulated
    idx = bench.index.intersection(a.dropna().index).intersection(b.dropna().index)
    idx = idx.intersection(covered)
    r = bench.loc[idx, "ret"].values

    def loss(v):
        q = -v.loc[idx].values
        return (ALPHA - (r < q).astype(float)) * (r - q)

    diff = pd.Series(loss(a) - loss(b), index=idx)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.5),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    a1.bar(bench.index, cov.values, width=2.0, color=al.MAIN_BLUE, alpha=0.75)
    a1.set_ylabel("Headlines in the\nas-of window")
    a1.set_title(f"Coverage: {len(covered)} of {len(bench)} dates")

    a2.axhline(0, color=GREY, lw=1)
    a2.plot(diff.index, diff.cumsum().values, color=al.IDA_RED, lw=1.8)
    a2.set_ylabel("Cumulative pinball\nloss difference")
    a2.set_title(f"news minus no-news, {len(idx)} covered days: "
                 f"{d_cov:+.5f}/day, DM $p$ = {p_cov:.3f}")
    fig.tight_layout()
    save(fig, "s_news")

    return {"news_dates_covered": int(len(covered)),
            "news_headlines": int(cov.sum()),
            "news_mean_per_covered": round(float(cov[cov > 0].mean()), 2),
            "dm_p_news_all": round(p_all, 3), "dm_diff_news_all": round(d_all, 5),
            "dm_p_news_covered": round(p_cov, 3), "dm_diff_news_covered": round(d_cov, 5),
            "n_covered_scored": int(len(idx)),
            "mean_LLM-dated": round(float(models["LLM-dated"].mean()), 3),
            "mean_LLM-dated+news": round(float(models["LLM-dated+news"].mean()), 3)}


def fig_open(bench, models):
    """The open-weights alternative, and what it costs in fidelity.

    Two things are shown because they fail differently. Left: the forecasts through
    time, commercial against open. Right: how many DISTINCT values each model emits.
    A model can pass every validity check in the pipeline -- parses, correct sign,
    correct ordering of the two quantiles -- while emitting a handful of numbers, and
    no validity check can see that. Only counting can.
    """
    keys = [k for k in ("Open-1.5B", "Open-3B", "Open-7B", "Open-14B") if k in models]
    if not keys or "LLM-series" not in models:
        print("  skip s_open: open-weights runs missing")
        return {}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 4.8),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    a1.plot(bench.index, bench["ret"], lw=0.6, color="#CCCCCC", label="Realised return")
    # Left panel: the two legs the laboratory actually runs. The larger models are a
    # size sweep, and belong in the bar panel rather than as four more lines here.
    shown = [k for k in ("LLM-series", "Open-1.5B") if k in models]
    for k in shown:
        v = models[k].dropna()
        a1.plot(v.index, -v.values, lw=1.5, color=MODEL_COLORS.get(k, GREY),
                label=MODEL_LABELS.get(k, k))
    a1.set_ylabel("Daily return (%)")
    a1.set_title(f"Same prompt, {'two' if len(shown) == 2 else len(shown)} models")
    legend_below(a1, ncol=2, y=-0.20, fontsize=9)

    names = ["LLM-series", *keys]
    frac = [100 * models[k].round(2).nunique() / max(models[k].notna().sum(), 1)
            for k in names]
    a2.bar([MODEL_LABELS.get(k, k).split(" (")[0] for k in names], frac,
           color=[MODEL_COLORS.get(k, GREY) for k in names], width=0.6)
    a2.set_ylabel("Distinct VaR values\n(% of forecast days)")
    a2.set_ylim(0, 108)
    a2.set_title("How much of the output is actually varying?")
    a2.grid(axis="x", visible=False)
    a2.tick_params(axis="x", labelrotation=18, labelsize=9)
    for i, v in enumerate(frac):
        a2.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=11)

    fig.tight_layout()
    save(fig, "s_open")

    out = {}
    for k in names:
        v = models[k].dropna()
        out[f"distinct_pct_{k}"] = round(100 * v.round(2).nunique() / max(len(v), 1), 1)
        out[f"sd_{k}"] = round(float(v.std()), 3)
        out[f"corr_GARCH_{k}"] = round(float(
            pd.concat([v, models["GARCH-t"]], axis=1).dropna().corr().iloc[0, 1]), 3)
    return out


def fig_news_assets():
    """Does the news null replicate across markets, or is it one asset's accident?

    Five assets, each with its own market close, its own news feed and its own coverage.
    The single-asset null is worth little on its own; five of them, on independently
    fetched text, is the result.
    """
    rows = []
    for a in ASSETS:
        try:
            b = pd.read_csv(LAB / f"bench_{a}.csv", parse_dates=["date"])
            b = b[np.isclose(b["level"], ALPHA)]
            bb = b.pivot_table(index="date", columns="model", values="var")
            bb.insert(0, "ret", b.groupby("date")["ret"].first())
            mm = {}
            for cfg, k in [("dated", "d"), ("dated_news", "dn")]:
                d = pd.read_csv(PRECOMP / f"llm_{cfg}_{a}_{LLM_MODEL}.csv",
                                parse_dates=["date"]).set_index("date")
                mm[k] = d.loc[d["raw_ok"] & d["sign_ok"], "var"]
            h = pd.read_csv(ROOT / "data" / "news" / f"headlines_{a}.csv",
                            parse_dates=["date"])
            cov = h.groupby("date").size().reindex(bb.index, fill_value=0)
            cd = bb.index[cov > 0]
            _, p, d_, n, _ = al.dm_test(bb.loc[cd, "ret"], mm["dn"].reindex(cd).dropna(),
                                        mm["d"].reindex(cd).dropna(), ALPHA)
        except FileNotFoundError:
            continue
        rows.append({"asset": a, "label": ASSETS[a]["label"],
                     "covered": int((cov > 0).sum()), "n": n, "diff": d_, "p": p})
    if not rows:
        print("  skip s_news_assets: incomplete runs")
        return {}
    r = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    col = [al.FOREST if x < 0 else al.IDA_RED for x in r["diff"]]
    ax.barh(r["label"], r["diff"], color=col, height=0.6)
    ax.axvline(0, color=GREY, lw=1.2)
    for i, (d_, p, n) in enumerate(zip(r["diff"], r["p"], r["n"])):
        ax.text(d_ + (0.0002 if d_ >= 0 else -0.0002), i,
                f"p = {p:.3f}  (n = {n})", va="center",
                ha="left" if d_ >= 0 else "right", fontsize=11)
    ax.set_xlim(min(r["diff"].min() * 1.9, -0.0035), max(r["diff"].max() * 2.6, 0.005))
    ax.set_xlabel("Mean pinball loss, with headlines minus without "
                  "(negative = headlines helped)")
    ax.set_title("The news null replicates on every market tested")
    ax.grid(axis="y", visible=False)
    save(fig, "s_news_assets")

    r.round(5).to_csv(PRECOMP / "lab_news_by_asset.csv", index=False)
    out = {"news_assets_tested": len(r), "news_assets_significant": int((r["p"] < 0.05).sum()),
           "news_min_p": round(float(r["p"].min()), 3)}
    for _, x in r.iterrows():
        out[f"news_p_{x['asset']}"] = round(float(x["p"]), 3)
        out[f"news_diff_{x['asset']}"] = round(float(x["diff"]), 5)
    return out


def fig_sampling():
    """Sampling against elicitation, on the same weights and the same days.

    Everything else in the deck compares models. This compares two ways of getting a
    number out of ONE model: Qwen2.5-1.5B asked for the quantile in words, against the
    same Qwen2.5-1.5B sampled 1000 times with the quantile read off the draws. Same
    weights, same 500 days, same parser, same scoring. It is the only fully controlled
    experiment in the package, so it is the one place a difference can be attributed
    to the extraction method rather than to the model.

    Two panels because the answer is two-sided: sampling wins on loss, and neither
    method survives the coverage test.
    """
    rows = []
    for a in ASSETS:
        f = PRECOMP / f"llmtime_{a}_Qwen2.5-1.5B-Instruct_T0.7_N1000_float16.csv"
        g = PRECOMP / f"local_series_{a}_Qwen2.5-1.5B-Instruct.csv"
        if not (f.exists() and g.exists()):
            continue
        b = pd.read_csv(LAB / f"bench_{a}.csv", parse_dates=["date"])
        ret = b[np.isclose(b["level"], ALPHA)].groupby("date")["ret"].first()
        s = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        e = pd.read_csv(g, parse_dates=["date"]).set_index("date")
        m = {"sampled": s.loc[_usable(s), "var_01"], "elicited": e.loc[_usable(e), "var"]}
        bt = al.backtest_table(ret, m, alpha=ALPHA).set_index("model")
        for k in m:
            rows.append({"asset": a, "label": ASSETS[a]["label"], "method": k,
                         "distinct": 100 * m[k].round(2).nunique() / len(m[k]),
                         "pinball": float(bt.loc[k, "pinball"]),
                         "rate": float(bt.loc[k, "rate_pct"]),
                         "p_uc": float(bt.loc[k, "p_uc"])})
    if not rows:
        print("  skip s_sampling: llmtime or open-weights runs missing")
        return {}
    r = pd.DataFrame(rows)
    piv = {k: r[r["method"] == k].set_index("asset") for k in ("sampled", "elicited")}
    assets = [a for a in ASSETS if a in piv["sampled"].index]
    x, w = np.arange(len(assets)), 0.36
    col = {"sampled": al.MAIN_BLUE, "elicited": "#1B7837"}
    lab = {"sampled": "Sampled, 1000 draws", "elicited": "Asked for the quantile"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.4))
    for k, off in (("sampled", -0.5), ("elicited", 0.5)):
        a1.bar(x + off * w, piv[k].loc[assets, "pinball"], width=w, color=col[k],
               alpha=0.9, label=lab[k])
    a1.set_xticks(x)
    a1.set_xticklabels([ASSETS[a]["label"] for a in assets], fontsize=10)
    a1.set_ylabel("Mean pinball loss at 1%")
    a1.set_title("Sampling scores better on every asset")
    a1.grid(axis="x", visible=False)
    legend_below(a1, ncol=2, y=-0.20, fontsize=10)

    for k, off in (("sampled", -0.5), ("elicited", 0.5)):
        a2.bar(x + off * w, piv[k].loc[assets, "rate"], width=w, color=col[k], alpha=0.9,
               label=lab[k])
    a2.axhline(100 * ALPHA, color=al.IDA_RED, lw=1.6, ls="--",
               label=f"nominal {ALPHA:.0%}")
    a2.set_xticks(x)
    a2.set_xticklabels([ASSETS[a]["label"] for a in assets], fontsize=10)
    a2.set_ylabel("Breach rate (%)")
    a2.set_title("And neither one covers")
    a2.grid(axis="x", visible=False)
    legend_below(a2, ncol=3, y=-0.20, fontsize=10)

    fig.tight_layout()
    save(fig, "s_sampling")

    r.round(4).to_csv(PRECOMP / "lab_sampling_vs_elicited.csv", index=False)
    best = (piv["sampled"].loc[assets, "pinball"] < piv["elicited"].loc[assets, "pinball"])
    return {"samp_assets": len(assets),
            "samp_pinball_wins": int(best.sum()),
            "samp_rejected_both": int((r["p_uc"] < 0.05).sum()),
            "samp_distinct_SPX": round(float(piv["sampled"].loc["SPX", "distinct"]), 1),
            "elic_distinct_SPX": round(float(piv["elicited"].loc["SPX", "distinct"]), 1),
            "samp_rate_min": round(float(r["rate"].min()), 2),
            "samp_rate_max": round(float(r["rate"].max()), 2)}


def fig_pinball(rets):
    """What the pinball loss is, and why 1% is harder than 5% for the same estimator.

    Left: the loss itself. It is piecewise linear with a kink at zero error, and the
    two arms have slopes alpha and 1 - alpha, so at alpha = 1% an over-forecast costs
    99 times what an equally sized under-forecast costs.

    Right: the same loss in expectation over the realised return sample, as a function
    of the candidate quantile. This is the picture behind "strictly consistent": the
    curve is minimised at the sample quantile and nowhere else. It is also the picture
    behind the difficulty, and the two are the same picture -- around its minimum the
    1% curve is far flatter than the 5% curve, so a whole range of candidate quantiles
    is nearly indistinguishable on 500 days of data.
    """
    r = rets["ret"].dropna().values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.4))

    err = np.linspace(-3, 3, 601)
    for lev, c in ((ALPHA, al.IDA_RED), (ALPHA_SECONDARY, al.MAIN_BLUE)):
        a1.plot(err, np.where(err < 0, (lev - 1) * err, lev * err), lw=2, color=c,
                label=fr"$\alpha = {lev:.0%}$".replace("%", r"\%"))
    a1.axvline(0, color=GREY, lw=1, zorder=0)
    a1.annotate(r"slope $\alpha - 1$", xy=(-2.2, 2.15), fontsize=11, color=GREY)
    a1.annotate(r"slope $\alpha$", xy=(1.2, 0.30), fontsize=11, color=GREY)
    a1.set_xlabel(r"Forecast error $r_t - q_t$ (%)")
    a1.set_ylabel("Pinball loss")
    a1.set_title("At 1%, an over-forecast costs 99 times less")
    legend_below(a1, ncol=2, y=-0.22, fontsize=11)

    for lev, c in ((ALPHA, al.IDA_RED), (ALPHA_SECONDARY, al.MAIN_BLUE)):
        q0 = float(np.quantile(r, lev))
        grid = np.linspace(q0 - 1.6, q0 + 1.6, 321)
        e = np.array([np.mean(np.where(r < g, (lev - 1) * (r - g), lev * (r - g)))
                      for g in grid])
        a2.plot(grid, e / e.min(), lw=2, color=c,
                label=fr"$\alpha = {lev:.0%}$".replace("%", r"\%"))
        a2.plot([q0], [1.0], "o", color=c, ms=7)
    a2.set_xlabel("Candidate quantile (%)")
    a2.set_ylabel("Mean loss, relative to its minimum")
    a2.set_title("Minimised at the truth, and flat around it at 1%")
    legend_below(a2, ncol=2, y=-0.22, fontsize=11)

    fig.tight_layout()
    save(fig, "s_pinball")

    # How flat is flat: the width of the band of candidate quantiles within 1% of the
    # minimum mean loss. This is the number the slide states.
    out = {}
    for lev, tag in ((ALPHA, "01"), (ALPHA_SECONDARY, "05")):
        q0 = float(np.quantile(r, lev))
        grid = np.linspace(q0 - 2.0, q0 + 2.0, 801)
        e = np.array([np.mean(np.where(r < g, (lev - 1) * (r - g), lev * (r - g)))
                      for g in grid])
        band = grid[e <= 1.01 * e.min()]
        out[f"pinball_flat_band_{tag}"] = round(float(band.max() - band.min()), 3)
    return out


def fig_fz(rets):
    """Why one number cannot score a pair, and what does.

    The FZ0 loss of Patton, Ziegel and Chen scores a (VaR, ES) pair jointly: it is
    strictly consistent for the pair, and for the pair only -- neither coordinate is
    elicitable on its own at second order. The contour shows what that means in
    practice: the minimum sits at the sample (VaR, ES) in both coordinates at once,
    and the valley is far shallower along ES than along VaR, which is the same
    small-sample problem the rest of the lecture keeps meeting.
    """
    r = rets["ret"].dropna().values
    v0 = float(np.quantile(r, ALPHA))            # negative: the alpha-quantile
    e0 = float(r[r <= v0].mean())                # negative: the tail mean

    def fz0(v, e):
        """Patton, Ziegel and Chen (2019), FZ0; v, e < 0, homogeneous of degree zero."""
        ind = (r[:, None, None] <= v[None, :, :])
        shortfall = np.where(ind, v[None, :, :] - r[:, None, None], 0.0).mean(axis=0)
        return -(1 / (ALPHA * e)) * shortfall + v / e + np.log(-e) - 1

    vs = np.linspace(v0 - 1.1, v0 + 1.1, 121)
    es = np.linspace(e0 - 1.6, e0 + 1.6, 121)
    V, E = np.meshgrid(vs, es)
    L = fz0(V, E)
    i, j = np.unravel_index(np.argmin(L), L.shape)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    cs = ax.contourf(V, E, L, levels=28, cmap="Blues_r", alpha=0.9)
    ax.contour(V, E, L, levels=14, colors="white", linewidths=0.5, alpha=0.6)
    ax.plot(v0, e0, "o", color=al.IDA_RED, ms=9, label="Sample (VaR, ES)")
    ax.plot(V[i, j], E[i, j], "x", color="black", ms=10, mew=2,
            label="Minimiser of the FZ$_0$ loss")
    ax.set_xlabel(r"Candidate 1% quantile $v$ (%)")
    ax.set_ylabel(r"Candidate expected shortfall $e$ (%)")
    ax.set_title("One surface, one minimum, two coordinates")
    ax.grid(visible=False)
    fig.colorbar(cs, ax=ax, label="FZ$_0$ loss")
    legend_below(ax, ncol=2, y=-0.24, fontsize=11)
    fig.tight_layout()
    save(fig, "s_fz")

    return {"fz_var_truth": round(v0, 3), "fz_es_truth": round(e0, 3),
            "fz_var_argmin": round(float(V[i, j]), 3),
            "fz_es_argmin": round(float(E[i, j]), 3)}


def fig_floor():
    """The sampling floor, first in theory and then measured on this laboratory's runs.

    Left: the standard deviation of a sample quantile as a function of the number of
    draws, by simulation. At alpha = 1% the library default of 20 draws does not have
    an order statistic below the quantile at all, and 500 draws -- the certified
    setting here -- still leaves the estimate visibly noisy.

    Right: the same thing without simulation. Chronos-T5 was run twice on identical
    inputs, at 500 and at 2000 sample paths. The two runs are the same model on the
    same day and differ only in draw count, so the spread between them is the floor
    itself, measured.
    """
    rng = np.random.default_rng(SEED)
    reps, sizes = 3000, [20, 60, 120, 250, 500, 1000, 2000, 5000]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.4))

    facts = {"floor_mc_reps": reps}
    for lev, c, tag in ((ALPHA, al.IDA_RED, "01"), (ALPHA_SECONDARY, al.MAIN_BLUE, "05")):
        sd, mcse = [], []
        for n in sizes:
            q = np.quantile(stats.t.rvs(5, size=(reps, n), random_state=rng), lev, axis=1)
            sd.append(float(q.std(ddof=1)))
            mcse.append(float(q.std(ddof=1) / np.sqrt(2 * (reps - 1))))
        a1.errorbar(sizes, sd, yerr=mcse, lw=2, color=c, marker="o", ms=5, capsize=3,
                    label=fr"$\alpha = {lev:.0%}$".replace("%", r"\%"))
        facts[f"floor_sd_500_{tag}"] = round(sd[sizes.index(500)], 3)
        facts[f"floor_sd_2000_{tag}"] = round(sd[sizes.index(2000)], 3)
    a1.set_xscale("log")
    a1.set_xticks(sizes)
    a1.set_xticklabels([str(s) for s in sizes], fontsize=9)
    a1.set_xlabel("Sample paths drawn")
    a1.set_ylabel("SD of the estimated quantile\n(Student-$t_5$ draws)")
    a1.set_title(f"Simulation: {reps} replications per point")
    legend_below(a1, ncol=2, y=-0.22, fontsize=11)

    lo = PRECOMP / f"chronos_t5_{ASSET}.csv"
    hi = PRECOMP / f"chronos_t5_{ASSET}_N2000.csv"
    if lo.exists() and hi.exists():
        l = pd.read_csv(lo, parse_dates=["date"])
        h = pd.read_csv(hi, parse_dates=["date"])
        # The disagreement is not smooth noise. Chronos quantises values into a token
        # vocabulary, so the quantile either lands on the same token in both runs or
        # moves one step. Counting the two cases says more than a standard deviation.
        cats, w = ["Identical", "One grid step", "Something else"], 0.36
        for k, (lev, c, tag) in enumerate(((ALPHA, al.IDA_RED, "01"),
                                           (ALPHA_SECONDARY, al.MAIN_BLUE, "05"))):
            j = (l[np.isclose(l["level"], lev)].set_index("date")
                 .join(h[np.isclose(h["level"], lev)].set_index("date"),
                       lsuffix="_lo", rsuffix="_hi").dropna())
            d = (j["var_hi"] - j["var_lo"]).abs()
            step = float(d[d > 0.4].median()) if (d > 0.4).any() else np.nan
            share = [100 * float((d < 0.05).mean()),
                     100 * float(((d >= 0.4) & (d <= 0.8)).mean()),
                     100 * float(((d >= 0.05) & (d < 0.4)).mean()
                                 + (d > 0.8).mean())]
            a2.bar(np.arange(3) + (k - 0.5) * w, share, width=w, color=c, alpha=0.85,
                   label=fr"$\alpha = {lev:.0%}$".replace("%", r"\%"))
            for x, v in zip(np.arange(3) + (k - 0.5) * w, share):
                a2.text(x, v + 1.8, f"{v:.0f}%", ha="center", fontsize=10)
            facts[f"t5_rerun_sd_{tag}"] = round(float((j["var_hi"] - j["var_lo"]).std()), 3)
            facts[f"t5_rerun_same_{tag}"] = round(share[0], 1)
            facts[f"t5_rerun_step_{tag}"] = round(step, 3)
            facts[f"t5_rerun_n_{tag}"] = int(len(d))
        a2.set_xticks(np.arange(3))
        a2.set_xticklabels(cats)
        a2.set_ylim(0, 100)
        a2.set_ylabel("Share of the 500 days (%)")
        a2.set_title(f"Measured: {ASSETS[ASSET]['label']}, 500 paths against 2000")
        a2.grid(axis="x", visible=False)
        legend_below(a2, ncol=2, y=-0.22, fontsize=11)
    else:
        print("  s_floor: right panel skipped, no 2000-path run for this asset")

    fig.tight_layout()
    save(fig, "s_floor")
    return facts


# ---------------------------------------------------------------------------
def main():
    rets, bench, models, bolt = load()
    print(f"{ASSET}: {len(bench)} test days, models {sorted(models)}")

    facts = {}
    facts.update(fig_power(bench))
    facts.update(fig_clamp(bolt))
    facts.update(fig_thresholds(bench, models))
    facts.update(fig_backtest(bench, models))
    facts.update(fig_disagreement(models))
    facts.update(fig_llm(bench, models))
    facts.update(fig_news(bench, models))
    facts.update(fig_open(bench, models))
    facts.update(fig_news_assets())
    facts.update(fig_pinball(rets))
    facts.update(fig_fz(rets))
    facts.update(fig_floor())
    facts.update(fig_sampling())

    # Every inline number on a slide is written here first, so the deck can be checked
    # against the data instead of against memory.
    out = PRECOMP / f"slide_facts_{ASSET}.csv"
    pd.Series(facts, dtype=object).rename("value").rename_axis("fact").to_csv(out)
    print(f"\n  facts -> {out.name}")
    for k, v in facts.items():
        print(f"    {k:<28} {v}")


if __name__ == "__main__":
    main()
