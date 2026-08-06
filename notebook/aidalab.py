"""aidalab — helper module for the AIDA Summer School risk laboratory.

One file, four groups of functions:

    loading      load_returns, load_benchmarks, load_precomputed
    forecasting  hs_var, rolling_hs
    scoring      pinball_loss, kupiec_uc, christoffersen_ind, conditional_coverage,
                 brier_score, backtest_table, leaderboard
    plotting     plot_tail, plot_breaches, plot_disagreement, plot_leaderboard

Dependencies are deliberately limited to numpy, pandas, scipy and matplotlib, all of
which are preinstalled on Google Colab, so the notebook's first cell never has to
install anything before the first result appears.

SIGN CONVENTION. VaR and ES are positive losses:

    VaR_t(alpha) = -F_t^{-1}(alpha) > 0,   breach on day t  <=>  ret_t < -VaR_t(alpha)

PLOTTING CONVENTION. Figures are drawn on the RETURN axis with the tail on the LEFT.
A VaR of 4.7% appears as a threshold at a return of -4.7% and is labelled 4.7%. The two
conventions live in different places on purpose: a table quotes VaR as a positive number
because that is how a risk report quotes it, and a chart shows returns because that is
how a price series reads.

Both conventions are inherited from aida-risk/src/common.py so that a number computed in
this notebook can be compared directly with the certified pipeline.
"""

from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "SEED", "ALPHA", "ALPHA_SECONDARY", "HS_WINDOW", "MODEL_COLORS", "MODEL_LABELS",
    "set_seeds", "data_root", "load_returns", "load_benchmarks", "load_precomputed",
    "hs_var", "rolling_hs", "pinball_loss", "kupiec_uc", "christoffersen_ind",
    "conditional_coverage", "basel_zone", "brier_score", "backtest_table",
    "leaderboard", "dm_test", "dm_matrix", "disagreement", "legend_below", "plot_tail", "plot_breaches",
    "plot_disagreement", "plot_leaderboard", "save_fig",
]

SEED = 42
ALPHA = 0.01        # the level the course reports
ALPHA_SECONDARY = 0.05
HS_WINDOW = 250

MAIN_BLUE = "#003DA5"
IDA_RED = "#C8102E"
FOREST = "#228B22"
GREY = "#6E6E6E"

MODEL_COLORS = {
    "HS": GREY,
    "GARCH-t": MAIN_BLUE,
    "NN-t": "#16A085",
    "Chronos-Bolt": "#8E44AD",
    "Chronos-T5": "#5B2C6F",
    "LLM-series": "#E67E22",
    "LLM-series+state": "#B9770E",
    "LLM-dated": "#7D6608",
    "LLM-dated+news": IDA_RED,
    "LLM-ensemble": "#943126",
    "Open-1.5B": "#1B7837",
    "Open-3B": "#A6761D",
    "Open-7B": "#8C6D1F",
    "Open-14B": "#5E4A12",
}
MODEL_LABELS = {
    "HS": "Historical simulation",
    "GARCH-t": "GARCH(1,1)-t",
    "NN-t": "Neural volatility + t tail",
    "Chronos-Bolt": "Chronos-Bolt (zero-shot)",
    "Chronos-T5": "Chronos-T5 (sampled)",
    "LLM-series": "LLM, series only",
    "LLM-series+state": "LLM, series + state",
    "LLM-dated": "LLM, dated (control)",
    "LLM-dated+news": "LLM, dated + headlines",
    "LLM-ensemble": "LLM ensemble",
    "Open-1.5B": "Qwen2.5-1.5B (open weights)",
    "Open-3B": "Qwen2.5-3B (open weights)",
    "Open-7B": "Qwen2.5-7B (open weights)",
    "Open-14B": "Qwen2.5-14B (open weights)",
}

plt.rcParams.update({
    "figure.figsize": (9, 4.2),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    # House convention: a legend never sits on the chart, and a figure never carries a
    # background. Slides sit on a gradient and notebooks may be read in dark mode.
    "savefig.transparent": True,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
})


def set_seeds(seed: int = SEED) -> None:
    """Seed every RNG the lab can touch. Call this in the first cell."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# Set DATA_URL to the raw base of the published repository before the course. The
# loaders try, in order: an explicit path, a local checkout, that URL, and finally a
# live yfinance download. The chain exists so the lab survives a room with no network
# and a network with no repository, which are different failures.
DATA_URL = os.environ.get("AIDA_DATA_URL", "").rstrip("/")


def data_root(explicit: str | Path | None = None) -> Path | None:
    """Directory holding returns_*.csv, or None when only the URL path is available."""
    if explicit:
        p = Path(explicit)
        return p if p.is_dir() else None
    here = Path.cwd()
    for base in [here, *here.parents]:
        for cand in (base / "data" / "lab", base / "august-course" / "data" / "lab",
                     base / "aida_lab_data"):
            if (cand / "manifest.json").exists():
                return cand
    return None


def _read_csv(name: str, root: Path | None, **kw) -> pd.DataFrame:
    if root is not None and (root / name).exists():
        return pd.read_csv(root / name, **kw)
    if DATA_URL:
        with urllib.request.urlopen(f"{DATA_URL}/{name}", timeout=30) as fh:
            return pd.read_csv(io.BytesIO(fh.read()), **kw)
    raise FileNotFoundError(
        f"{name} not found. Upload aida_lab_data.zip to the Colab session, or set "
        f"AIDA_DATA_URL to the published data directory.")


def load_returns(asset: str = "SPX", root=None) -> pd.DataFrame:
    """Full daily sample for an asset: index date, columns close and ret (log %, x100)."""
    root = data_root(root)
    try:
        df = _read_csv(f"returns_{asset}.csv", root, index_col=0, parse_dates=True)
    except FileNotFoundError:
        df = _download_returns(asset)
    df.index.name = "date"
    # ret[t] must be a function of close[t] and close[t-1] only. A shifted return is
    # the single most common silent error in this kind of exercise and it inflates
    # every model's apparent skill, so it is checked on load rather than trusted.
    chk = 100.0 * np.log(df["close"] / df["close"].shift(1))
    assert np.allclose(chk.iloc[1:], df["ret"].iloc[1:], atol=1e-8), \
        "return alignment broken: ret[t] does not equal 100*log(close[t]/close[t-1])"
    return df


def _download_returns(asset: str) -> pd.DataFrame:
    """Last resort: pull the series live. Sample will differ from the shipped file."""
    tickers = {"SPX": "^GSPC", "DAX": "^GDAXI", "N225": "^N225",
               "GOLD": "GC=F", "BTC": "BTC-USD"}
    import yfinance as yf
    print(f"falling back to a live download of {tickers[asset]}; "
          "results will not match the shipped precomputed files")
    px = yf.download(tickers[asset], start="1998-01-01", end="2024-12-31",
                     auto_adjust=True, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    close = px["Close"].astype(float).dropna()
    return pd.DataFrame({"close": close,
                         "ret": 100.0 * np.log(close / close.shift(1))}).dropna()


def load_benchmarks(asset: str = "SPX", level: float = ALPHA, root=None) -> pd.DataFrame:
    """Certified aida-risk forecasts over the test span, wide by model.

    Returns a frame indexed by date with a `ret` column and one column per model
    holding VaR as a positive loss.
    """
    root = data_root(root)
    b = _read_csv(f"bench_{asset}.csv", root, parse_dates=["date"])
    b = b[np.isclose(b["level"], level)]
    wide = b.pivot_table(index="date", columns="model", values="var")
    wide.insert(0, "ret", b.groupby("date")["ret"].first())
    wide.columns.name = None
    return wide


def load_precomputed(name: str, root=None) -> pd.DataFrame | None:
    """Load a precomputed forecast file, or return None if it was never generated.

    Used for the Chronos and LLM legs, which are shipped so the lab runs without a
    model download or an API key. None is a legitimate answer and the notebook is
    written to continue without that model rather than fail.
    """
    root = data_root(root)
    pre = None if root is None else root.parent.parent / "precomputed"
    if pre is not None:
        cand = pre / name
        if cand.exists():
            return pd.read_csv(cand, parse_dates=["date"])
        # A run that set --dtype carries it in the filename, so
        # local_series_SPX_Qwen2.5-3B-Instruct.csv ships as ..._float16.csv. Match on
        # the stem: a fixed name reported the file as missing while it sat on disk.
        stem, ext = name.rsplit(".", 1)
        alt = [q for q in sorted(pre.glob(f"{stem}*.{ext}"))
               if not q.name.startswith("_") and "smoke" not in q.name]
        if alt:
            return pd.read_csv(alt[0], parse_dates=["date"])
    try:
        return _read_csv(f"../precomputed/{name}", root, parse_dates=["date"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Forecasting: the transparent benchmark
# ---------------------------------------------------------------------------
def hs_var(window_returns, alpha: float = ALPHA) -> float:
    """Historical simulation VaR: the empirical alpha-quantile of a return window.

    No model, no parameters, no distribution. Sort the window and read off the
    quantile. Returned as a positive loss.
    """
    return float(-np.quantile(np.asarray(window_returns, dtype=float), alpha))


def rolling_hs(returns: pd.Series, dates, window: int = HS_WINDOW,
               alpha: float = ALPHA) -> pd.Series:
    """One-day-ahead HS VaR for each date in `dates`, using data up to t-1 only.

    The window ends at the observation before the forecast date. Slicing with `.loc`
    up to and including the forecast date is the standard way this exercise leaks:
    the day being forecast then sits inside its own estimation window.
    """
    r = returns.astype(float)
    out = {}
    for d in pd.DatetimeIndex(dates):
        hist = r.loc[:d].iloc[:-1]         # drop day d itself
        if len(hist) < window:
            continue
        out[d] = hs_var(hist.iloc[-window:], alpha)
    return pd.Series(out, name="HS").rename_axis("date")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def pinball_loss(ret, var, alpha: float = ALPHA) -> float:
    """Mean pinball (quantile) loss at level alpha.

    Scored on the return axis: the forecast quantile is q = -VaR, the outcome is the
    realised return. Lower is better. The pinball loss is the strictly consistent
    scoring function for a quantile, which is why it is the right ranking criterion
    and the breach count is not.
    """
    r = np.asarray(ret, dtype=float)
    q = -np.asarray(var, dtype=float)
    return float(np.mean((alpha - (r < q).astype(float)) * (r - q)))


def kupiec_uc(breaches, alpha: float = ALPHA):
    """Kupiec unconditional coverage. H0: breach probability equals alpha.

    Returns (LR, p, n, x). Low p rejects the model.
    """
    b = np.asarray(breaches).astype(int)
    n, x = len(b), int(b.sum())
    if x == 0:
        lr = -2.0 * (n * np.log(1 - alpha))
    else:
        pi = x / n
        ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
        ll1 = (n - x) * np.log(1 - pi) + x * np.log(pi)
        lr = -2.0 * (ll0 - ll1)
    return lr, float(1.0 - stats.chi2.cdf(lr, 1)), n, x


def christoffersen_ind(breaches):
    """Christoffersen independence. H0: breaches do not cluster (order-1 Markov).

    Returns (LR, p, (n00, n01, n10, n11)).
    """
    b = np.asarray(breaches).astype(int)
    n00 = int(((b[:-1] == 0) & (b[1:] == 0)).sum())
    n01 = int(((b[:-1] == 0) & (b[1:] == 1)).sum())
    n10 = int(((b[:-1] == 1) & (b[1:] == 0)).sum())
    n11 = int(((b[:-1] == 1) & (b[1:] == 1)).sum())
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return 0.0, 1.0, (n00, n01, n10, n11)
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _ll(p, a, c):
        return 0.0 if p in (0.0, 1.0) else a * np.log(1 - p) + c * np.log(p)

    lr = -2.0 * (_ll(pi, n00 + n10, n01 + n11)
                 - (_ll(pi01, n00, n01) + _ll(pi11, n10, n11)))
    return lr, float(1.0 - stats.chi2.cdf(lr, 1)), (n00, n01, n10, n11)


def conditional_coverage(breaches, alpha: float = ALPHA):
    """Christoffersen CC = UC + IND, chi2 with 2 degrees of freedom."""
    lr_uc, _, _, _ = kupiec_uc(breaches, alpha)
    lr_ind, _, _ = christoffersen_ind(breaches)
    lr = lr_uc + lr_ind
    return lr, float(1.0 - stats.chi2.cdf(lr, 2))


def basel_zone(n_breach: int, n_days: int) -> str:
    """Basel traffic light for 1% VaR, scaled from its 250-day definition."""
    scale = n_days / 250.0
    if n_breach <= 4 * scale:
        return "green"
    if n_breach <= 9 * scale:
        return "yellow"
    return "red"


def brier_score(prob, outcome) -> float:
    """Mean squared error of a probability forecast against a 0/1 outcome."""
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    return float(np.mean((p - y) ** 2))


def backtest_table(ret, forecasts: dict, alpha: float = ALPHA) -> pd.DataFrame:
    """Run the full battery over a dict of {model: VaR series} sharing an index.

    Every model is scored on the dates it actually covers, and `n` is reported per
    row rather than assumed common: a foundation model that skipped a date must not
    be compared with a benchmark that did not.
    """
    ret = pd.Series(ret).astype(float)
    rows = []
    for model, var in forecasts.items():
        v = pd.Series(var).astype(float).dropna()
        idx = ret.index.intersection(v.index)
        if len(idx) == 0:
            continue
        r, v = ret.loc[idx], v.loc[idx]
        br = (r.values < -v.values).astype(int)
        lr_uc, p_uc, n, x = kupiec_uc(br, alpha)
        lr_ind, p_ind, _ = christoffersen_ind(br)
        lr_cc, p_cc = conditional_coverage(br, alpha)
        rows.append({
            "model": model, "n": n, "expected": alpha * n, "observed": x,
            "rate_pct": 100.0 * x / n,
            "pinball": pinball_loss(r.values, v.values, alpha),
            "mean_var": float(v.mean()),
            "p_uc": p_uc, "p_ind": p_ind, "p_cc": p_cc,
            "neg_var_days": int((v <= 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("pinball").reset_index(drop=True)


def leaderboard(bt: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Rank by pinball loss, then annotate what the ranking does not say.

    Rank order comes from the pinball loss because it is the consistent scoring rule
    for a quantile. The coverage verdict is reported beside it and not folded into
    the rank: a model can win the loss comparison and still be rejected on coverage,
    and collapsing the two hides exactly the case the lab is about.
    """
    out = bt.copy()
    out["rank"] = out["pinball"].rank(method="min").astype(int)
    out["UC"] = np.where(out["p_uc"] < 0.05, "reject", "pass")
    out["IND"] = np.where(out["p_ind"] < 0.05, "reject", "pass")
    out["CC"] = np.where(out["p_cc"] < 0.05, "reject", "pass")
    cols = ["rank", "model", "n", "observed", "expected", "rate_pct",
            "pinball", "UC", "IND", "CC", "neg_var_days"]
    return out[cols].sort_values("rank").reset_index(drop=True)


def dm_test(ret, var_a, var_b, alpha: float = ALPHA):
    """Diebold-Mariano test on the pinball-loss differential, HAC standard error.

    Answers the question a leaderboard cannot: is the gap between two models larger
    than the sampling noise? Returns (t, p, mean_diff, n, lag). A negative mean_diff
    means model A has the lower loss.

    The long-run variance uses a Bartlett kernel with the standard automatic lag
    floor(4 (n/100)^(2/9)); daily loss differentials are autocorrelated because
    volatility clusters, so the naive standard error is too small and would turn
    ordinary noise into a significant result.
    """
    ret = pd.Series(ret).astype(float)
    a = pd.Series(var_a).astype(float).dropna()
    b = pd.Series(var_b).astype(float).dropna()
    idx = ret.index.intersection(a.index).intersection(b.index)
    r = ret.loc[idx].values

    def _loss(v):
        q = -v.loc[idx].values
        return (alpha - (r < q).astype(float)) * (r - q)

    d = _loss(a) - _loss(b)
    n = len(d)
    if n < 20:
        return np.nan, np.nan, float(np.mean(d)) if n else np.nan, n, 0
    dbar = float(d.mean())
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    s = float(np.var(d, ddof=0))
    for k in range(1, lag + 1):
        gk = float(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
        s += 2.0 * (1.0 - k / (lag + 1)) * gk
    se = np.sqrt(max(s, 1e-12) / n)
    t = dbar / se
    return float(t), float(2 * (1 - stats.norm.cdf(abs(t)))), dbar, n, lag


def dm_matrix(ret, forecasts: dict, alpha: float = ALPHA) -> pd.DataFrame:
    """Pairwise DM p-values. Read it before believing a leaderboard ordering."""
    keys = list(forecasts)
    out = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for i in keys:
        for j in keys:
            out.loc[i, j] = np.nan if i == j else \
                dm_test(ret, forecasts[i], forecasts[j], alpha)[1]
    return out


def disagreement(forecasts: dict) -> pd.DataFrame:
    """Per-day spread across a set of VaR forecasts, plus the pairwise correlations.

    Two models can carry different names, different training corpora and different
    vendors and still be one forecast. The spread is the quantity that tells you
    whether an ensemble of them holds any diversification at all.
    """
    wide = pd.DataFrame({k: pd.Series(v).astype(float) for k, v in forecasts.items()}).dropna()
    spread = pd.DataFrame({
        "min": wide.min(axis=1),
        "max": wide.max(axis=1),
        "mean": wide.mean(axis=1),
    })
    spread["range"] = spread["max"] - spread["min"]
    spread["ratio"] = spread["max"] / spread["min"].replace(0, np.nan)
    return spread


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def legend_below(ax, ncol=3, y=-0.16, fontsize=9):
    """Legend outside the axes, centred below them.

    Not only cosmetic: the interesting part of a VaR chart is the lower-left tail,
    which is exactly where matplotlib prefers to drop a legend box.
    """
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     fontsize=fontsize, frameon=False)


def plot_tail(ret: pd.Series, forecasts: dict, title: str = "", ax=None,
              show_breaches: str | None = None):
    """Returns as a line, each VaR forecast as a threshold on the return axis.

    `show_breaches` names the model whose breaches are marked. Marking every model's
    breaches at once produces a chart nobody can read live.
    """
    ax = ax or plt.subplots()[1]
    ax.plot(ret.index, ret.values, lw=0.7, color="#B0B0B0", label="Realised return", zorder=1)
    for model, var in forecasts.items():
        v = pd.Series(var).astype(float).dropna()
        ax.plot(v.index, -v.values, lw=1.3, color=MODEL_COLORS.get(model, None),
                label=MODEL_LABELS.get(model, model), zorder=2)
    if show_breaches and show_breaches in forecasts:
        v = pd.Series(forecasts[show_breaches]).astype(float).dropna()
        idx = ret.index.intersection(v.index)
        br = idx[ret.loc[idx].values < -v.loc[idx].values]
        ax.scatter(br, ret.loc[br].values, s=22, color=IDA_RED, zorder=3,
                   label=f"{MODEL_LABELS.get(show_breaches, show_breaches)} breaches")
    ax.set_ylabel("Daily return (%)")
    ax.set_title(title)
    legend_below(ax, ncol=3, y=-0.14)
    return ax


def plot_breaches(bt: pd.DataFrame, alpha: float = ALPHA, ax=None):
    """Observed against expected breach count, with the 95% acceptance band."""
    ax = ax or plt.subplots()[1]
    b = bt.sort_values("observed")
    n = int(b["n"].iloc[0])
    exp = alpha * n
    se = np.sqrt(alpha * (1 - alpha) * n)
    ax.axvspan(exp - 1.96 * se, exp + 1.96 * se, color=FOREST, alpha=0.12,
               label="95% acceptance band")
    ax.axvline(exp, color=FOREST, lw=1.2, label=f"Expected ({exp:.0f})")
    ax.barh(b["model"], b["observed"],
            color=[MODEL_COLORS.get(m, GREY) for m in b["model"]], height=0.6)
    ax.set_xlabel(f"Breaches in {n} days at alpha = {alpha:.0%}")
    ax.grid(axis="y", visible=False)
    legend_below(ax, ncol=2, y=-0.18)
    return ax


def plot_disagreement(spread: pd.DataFrame, ax=None):
    """The band between the most and least conservative model, day by day."""
    ax = ax or plt.subplots()[1]
    ax.fill_between(spread.index, -spread["max"], -spread["min"],
                    color=MAIN_BLUE, alpha=0.18, label="Range across models")
    ax.plot(spread.index, -spread["mean"], color=MAIN_BLUE, lw=1.2, label="Mean forecast")
    ax.set_ylabel("VaR threshold, return axis (%)")
    ax.set_title(f"Median spread {spread['range'].median():.2f}pp, "
                 f"median max/min {spread['ratio'].median():.2f}x")
    legend_below(ax, ncol=2, y=-0.16)
    return ax


def plot_leaderboard(lb: pd.DataFrame, ax=None):
    """Pinball loss by model, with rejected models hatched."""
    ax = ax or plt.subplots()[1]
    d = lb.sort_values("pinball", ascending=False)
    ax.barh(d["model"], d["pinball"],
            color=[MODEL_COLORS.get(m, GREY) for m in d["model"]],
            hatch=["//" if c == "reject" else "" for c in d["CC"]], height=0.6)
    ax.set_xlabel("Mean pinball loss (lower is better); hatched = CC rejected")
    ax.grid(axis="y", visible=False)
    return ax


def save_fig(fig, name: str, outdir="figures", dpi=300):
    """Export to PDF and 300 dpi PNG, as everywhere else in this project."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight", transparent=True)
    return out / f"{name}.pdf"
