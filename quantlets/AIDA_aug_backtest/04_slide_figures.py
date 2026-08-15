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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
# The realised series is not a model, so it gets a colour no model uses. The old
# #CCCCCC vanished on a projector; charcoal stays legible while a thin line keeps it
# behind the forecasts it is there to be compared against.
RETURNS = "#3F4B57"

# The primary comparison. Qwen 3B, 7B and 14B are size diagnostics and are deliberately
# NOT in it: they answer "what does scale do to one family", not "which model forecasts
# this asset". Any figure that ranks or aggregates models uses this list.
PRIMARY = ("HS", "GARCH-t", "NN-t", "Chronos-T5", "LLM-series", "LLM-series+state",
           "LLM-dated", "LLM-dated+news", "Open-1.5B")

# Projection scale. A figure on a slide is read from the back of a room, not from a
# laptop: type is sized so a panel stays legible when the frame shrinks it.
#
# The canvas is what makes that work. The deck's text block is 409.72pt = 5.67in wide,
# so a figure included at width w*\textwidth is shrunk by w*5.67/W, where W is the
# canvas width in inches. Every canvas below is paired with an inclusion width in
# slides/ai_risk_course_2026-08.tex so that factor is 0.56 for ALL figures: 14pt ticks
# land at 7.8pt on the slide, 16pt labels at 9.0pt, 17pt titles at 9.5pt, against 9pt
# body type. Type at this size only fits if the canvas is wide enough to hold it --
# the earlier 7.8in canvases were not, which is why titles ran across their neighbour.
mpl.rcParams.update({
    "figure.figsize": (7.2, 3.6),
    "font.size": 12.5,
    "axes.titlesize": 13.5,
    "axes.labelsize": 12.5,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    # Constrained layout, not tight_layout: it reserves room for titles, colourbars AND
    # legends anchored outside the axes, and it does so per-panel, which is what keeps
    # two panels off each other. bbox is left alone so the saved canvas is exactly
    # figsize and the shrink factor above is the one that actually applies.
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 0.06,
    "figure.constrained_layout.h_pad": 0.04,
    "figure.constrained_layout.wspace": 0.06,
    # The saved canvas is exactly figsize, so the shrink factor above is the one that
    # applies. bbox="tight" would silently re-cut the canvas per figure -- and, with
    # constrained layout, re-cut it wrongly -- which is how the earlier build ended up
    # with clipped axis labels. save() warns instead, and the fix is to shorten the
    # label rather than to grow the canvas behind it.
    "savefig.bbox": None,
    "savefig.transparent": True,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
})

# Canvas sizes. TWO for side-by-side panels, WIDE for one panel that runs the full
# text block, ONE for a compact single panel, SQUARE for a panel with a colourbar.
# Every figure frame carries a bulleted note, which costs roughly 55pt of frame height
# and pushes the inclusion width down to about 0.77. The canvases are sized for THAT
# width, so the 0.56 shrink still holds and the type matches the 9pt body: a canvas
# scaled for 0.95 would render at 0.42 and be unreadable from the back of a room.
FIG_TWO = (8.2, 3.0)     # -> \includegraphics[width=0.77\textwidth]
FIG_WIDE = (8.2, 2.9)    # -> \includegraphics[width=0.77\textwidth]
FIG_ONE = (6.2, 2.9)     # -> \includegraphics[width=0.59\textwidth]
FIG_SQ = (5.8, 3.1)      # -> \includegraphics[width=0.55\textwidth]
# For a figure that shares its frame with a block of bullets: narrower canvas, so the
# smaller inclusion width still lands on the same 0.56 shrink and the type matches
# every other figure in the deck.
FIG_BLK = (7.6, 3.2)     # -> \includegraphics[width=0.75\textwidth]
# Same width as FIG_TWO, so the same type scale, but flatter: for frames whose note runs
# to three bullets and which therefore cannot afford a 3.0in canvas at full width.
FIG_FLAT = (8.2, 2.45)   # -> \includegraphics[width=0.795\textwidth]
FIG_DATA = (7.0, 2.15)
FIG_ARCH = (12.6, 4.3)   # -> \includegraphics[width=0.98\textwidth]
FIG_LLMARCH = (12.6, 3.1)  # -> \includegraphics[width=0.98\textwidth], one row
FIG_NN = (8.6, 3.3)      # -> \includegraphics[width=0.80\textwidth]   # -> \includegraphics[width=0.58\textwidth], caption is tall

# Bar and heatmap labels. The long forms in MODEL_LABELS are for tables and legends,
# where a line of text has the whole slide to run in; a categorical axis has a fifth
# of the panel, so it gets its own vocabulary.

# One short form per model. Tables and legends use the descriptive names in
# labcommon.MODEL_LABELS; everything drawn inside a panel uses these, so a model does
# not appear under three different names across the deck.
AXIS_LABELS = {
    "HS": "HS",
    "GARCH-t": "GARCH-$t$",
    "NN-t": "NN-$t$",
    "Chronos-Bolt": "Chronos-Bolt",
    "Chronos-T5": "Chronos-T5",
    "LLM-series": "Returns",
    "LLM-series+state": "$+$ stats",
    "LLM-dated": "$+$ date",
    "LLM-dated+news": "$+$ news",
    "LLM-ensemble": "LLM ens.",
    "Open-1.5B": "Qwen 1.5B",
    "Open-3B": "Qwen 3B",
    "Open-7B": "Qwen 7B",
    "Open-14B": "Qwen 14B",
}

SLIDES = FIGURES / "slides"
SLIDES.mkdir(parents=True, exist_ok=True)


LINE_LABELS = {          # for panels where a line is one model's forecast
    "LLM-series": "Claude Haiku 4.5",
    "LLM-series+state": "Claude, $+$ stats",
    "LLM-dated": "Claude, $+$ date",
    "LLM-dated+news": "Claude, $+$ news",
}


def line_label(m):
    """The model behind a plotted forecast. AXIS_LABELS names the PROMPT, which reads
    correctly as a bar category and wrongly as a line in a panel of thresholds."""
    return LINE_LABELS.get(m, axis_label(m))


def axis_label(m):
    return AXIS_LABELS.get(m, m)


def matrix_label(m):
    """Kept as a name for intent; a matrix uses the same short form as an axis."""
    return axis_label(m)


def datefmt(ax, years=1):
    """Year ticks, unrotated. A daily index over two years otherwise prints a dozen
    overlapping date strings at projection type size."""
    ax.xaxis.set_major_locator(mdates.YearLocator(years))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=0)


def fig_legend(fig, ax, ncol=2, fontsize=8.6):
    """One legend for the whole figure, below the panels.

    Two panels showing the same series do not need two legends: side by side they
    collide, and constrained layout pays for both by shrinking the axes. Pass a list of
    axes to merge their keys; duplicates are dropped, first occurrence wins. A legend
    inside a time-series panel sits on the data, which is why these go underneath.
    """
    axes = ax if isinstance(ax, (list, tuple)) else [ax]
    hh, ll = [], []
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in ll:
                hh.append(h); ll.append(l)
    return fig.legend(hh, ll, loc="outside lower center", ncol=ncol,
                      fontsize=fontsize, frameon=False)


def save(fig, name):
    # Constrained layout places everything inside the canvas -- unless a title or an
    # axis label is simply wider than the room it has, in which case matplotlib lets it
    # run off the edge and the PDF clips it. That is invisible until the slide is on a
    # projector, so it is checked here.
    #
    # fig.get_tightbbox() is NOT enough: it returns the CLIPPED extent, so a label
    # already cut off by the canvas edge measures as if it fit. Every text artist is
    # therefore measured directly.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w, h = fig.get_size_inches()
    inv = fig.dpi_scale_trans.inverted()
    worst, who = {"left": 0.0, "bottom": 0.0, "right": 0.0, "top": 0.0}, {}

    def note(bb, label):
        for edge, v in (("left", -bb.x0), ("bottom", -bb.y0),
                        ("right", bb.x1 - w), ("top", bb.y1 - h)):
            if v > worst[edge]:
                worst[edge], who[edge] = v, label

    note(fig.get_tightbbox(r), "layout")
    for ax in fig.axes:
        texts = [(ax.title, "title"), (ax.xaxis.label, "xlabel"),
                 (ax.yaxis.label, "ylabel")]
        # Only ticks inside the view interval are drawn; matplotlib keeps artists for
        # the rest, and measuring those reports overflow that never reaches the page.
        for axis, kind in ((ax.xaxis, "xtick"), (ax.yaxis, "ytick")):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    texts.append((tick.label1, kind))
        texts += [(t, "text") for t in ax.texts]
        for artist, kind in texts:
            if not artist.get_text():
                continue
            note(artist.get_window_extent(r).transformed(inv), f"{kind} {artist.get_text()[:22]!r}")
    for lg in list(fig.legends) + [a.get_legend() for a in fig.axes if a.get_legend()]:
        note(lg.get_window_extent(r).transformed(inv), "legend")

    bad = {k: (round(v, 2), who[k]) for k, v in worst.items() if v > 0.02}
    if bad:
        for edge, (amt, label) in bad.items():
            print(f"  !! {name}: {label} runs {amt} in off the {edge} edge "
                  f"-- shorten it, do not grow the canvas")

    # Neighbouring tick labels that run into each other are unreadable long before
    # anything leaves the canvas, so consecutive labels are compared pairwise.
    for ax in fig.axes:
        if hasattr(ax, "_colorbar"):
            continue
        for axis, kind in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            lo, hi = sorted(axis.get_view_interval())
            vis = [t for t in axis.get_major_ticks()
                   if lo - 1e-9 <= t.get_loc() <= hi + 1e-9 and t.label1.get_text()]
            bbs = [(t.label1.get_text(),
                    t.label1.get_window_extent(r).transformed(inv)) for t in vis]
            bbs.sort(key=lambda e: e[1].x0 if kind == "x" else e[1].y0)
            for (t1, b1), (t2, b2) in zip(bbs, bbs[1:]):
                gap = (b2.x0 - b1.x1) if kind == "x" else (b2.y0 - b1.y1)
                if gap < 0.005:
                    print(f"  !! {name}: {kind}-tick labels {t1!r} and {t2!r} collide "
                          f"({-gap:.2f} in) -- shorten or rotate them")
                    break

    # Fitting the canvas is not enough: a title wider than its own panel can still run
    # across the panel beside it, which is the defect this whole exercise started from.
    # Constrained layout does not prevent it, so it is measured.
    # A colourbar is attached to its panel by design, so its box legitimately touches.
    boxes = [(ax, ax.get_tightbbox(r).transformed(inv)) for ax in fig.axes
             if not hasattr(ax, "_colorbar")]
    for i, (ax_a, ba) in enumerate(boxes):
        for ax_b, bb in boxes[i + 1:]:
            ox = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            oy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if ox > 0.02 and oy > 0.02:
                ta = (ax_a.get_title() or ax_a.get_ylabel() or "panel")[:26]
                tb = (ax_b.get_title() or ax_b.get_ylabel() or "panel")[:26]
                print(f"  !! {name}: panels overlap by {ox:.2f} x {oy:.2f} in "
                      f"-- {ta!r} against {tb!r}")
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
def _kupiec_reject(n, alpha=ALPHA, level=0.05):
    """Breach counts in the rejection region of the two-sided Kupiec LR_UC test.

    Exact: the statistic is evaluated at every attainable count 0..n and compared with
    the chi-square(1) critical value, rather than inverted through a normal
    approximation. LR_UC is two-sided by construction -- too few breaches is a
    rejection as much as too many.
    """
    crit = stats.chi2.ppf(1 - level, 1)
    x = np.arange(n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ph = x / n
        ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
        ll1 = np.where(x == 0, n * np.log(1 - ph[0] if n else 1),
                       (n - x) * np.log(np.clip(1 - ph, 1e-300, None))
                       + x * np.log(np.clip(ph, 1e-300, None)))
        ll1 = np.where(x == 0, 0.0, ll1)
    return x[-2 * (ll0 - ll1) > crit]


def _detectable_power(n, target=0.80, alpha=ALPHA, level=0.05):
    """Smallest true breach probability the test rejects with `target` power.

    This is a power calculation: for each candidate p, the probability that a
    Binomial(n, p) count lands in the exact rejection region above.
    """
    rej = _kupiec_reject(n, alpha, level)
    if len(rej) == 0:
        return np.nan
    for p in np.arange(alpha, 0.90, 0.0005):
        if float(stats.binom.pmf(rej, n, p).sum()) >= target:
            return p
    return np.nan


def _detectable_expected(n, alpha=ALPHA, level=0.05):
    """Weaker criterion: the smallest p whose EXPECTED count is already a rejection.

    Kept alongside the power curve because it is the number people quote, and it is
    optimistic by roughly a factor of two: an expected count inside the rejection
    region still leaves the realised count outside it about half the time.
    """
    crit = stats.chi2.ppf(1 - level, 1)
    for p in np.arange(alpha, 0.90, 0.0005):
        x = p * n
        ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
        ll1 = (n - x) * np.log(1 - p) + x * np.log(p)
        if -2 * (ll0 - ll1) > crit:
            return p
    return np.nan


def fig_power(bench):
    """Why the lab evaluates 500 days and not 50, stated as an actual power curve."""
    ns = np.array([25, 50, 75, 100, 150, 250, 500, 1000, 2000])
    pw = np.array([_detectable_power(int(n)) for n in ns])
    ex = np.array([_detectable_expected(int(n)) for n in ns])

    fig, ax = plt.subplots(figsize=FIG_ONE)
    ax.plot(ns, 100 * pw, "o-", color=al.MAIN_BLUE, lw=2, ms=6,
            label="Detectable with 80% power")
    ax.plot(ns, 100 * ex, "s--", color=GREY, lw=1.6, ms=5,
            label="Expected count alone rejects")
    ax.axhline(100 * ALPHA, color=al.FOREST, lw=1.4, ls="--",
               label=f"nominal {ALPHA:.0%}")
    ax.axvline(500, color=al.IDA_RED, lw=1.4, ls=":", label="lab span, 500 days")
    ax.set_xscale("log")
    # Every point is drawn; only every other one is labelled, because nine log-spaced
    # date strings at projection type size do not fit under a 7in axis.
    shown = [25, 50, 100, 250, 500, 1000, 2000]
    ax.set_xticks(shown)
    ax.set_xticklabels([str(n) for n in shown])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Out-of-sample days")
    ax.set_ylabel("Detectable (%)")
    fig_legend(fig, ax, ncol=2)
    save(fig, "s_power")
    i500, i5541 = list(ns).index(500), None
    return {"detect_power_500": round(100 * pw[i500], 2),
            "detect_power_50": round(100 * pw[1], 2),
            "detect_expected_500": round(100 * ex[i500], 2),
            "detect_power_5541": round(100 * _detectable_power(5541), 2),
            "detect_expected_5541": round(100 * _detectable_expected(5541), 2)}


def fig_clamp(bolt, bench):
    """The clamp, shown as the finding rather than asserted.

    Three coincident lines are indistinguishable from one line, and three bars at 100%
    are one number drawn three times: the previous version of this figure could not
    show what it claimed. What separates a working quantile head from a clamped one is
    the SPREAD between requested levels, so that is what is drawn -- positive and moving
    for GARCH-t, exactly zero every day for Chronos-Bolt.

    The right panel gives the consequence a desk would actually meet: take the number
    returned for 1% and score it, and the realised breach rate is what it is.
    """
    if bolt is None:
        print("  skip s_clamp: no bolt file")
        return {}
    piv = bolt.pivot_table(index="date", columns="level", values="var")
    gar = bench.pivot_table(index="date", columns="level", values="var") \
        if "level" in getattr(bench, "columns", []) else None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_TWO,
                                 gridspec_kw={"width_ratios": [1.45, 1]})

    # --- left: the gap between the levels you asked for ---------------------------
    ref = pd.read_csv(LAB / f"bench_{ASSET}.csv", parse_dates=["date"])
    ref = ref[ref["model"] == "GARCH-t"].pivot_table(index="date", columns="level",
                                                     values="var")
    common = piv.index.intersection(ref.index)
    a1.plot(common, (ref.loc[common, 0.01] - ref.loc[common, 0.10]).values,
            color=MODEL_COLORS["GARCH-t"], lw=1.6,
            label=r"GARCH-$t$: a real quantile head")
    a1.plot(common, (piv.loc[common, 0.01] - piv.loc[common, 0.10]).values,
            color=al.IDA_RED, lw=2.2, label="Chronos-Bolt")
    a1.axhline(0, color=RETURNS, lw=1)
    a1.set_ylabel(r"VaR$(1\%)$ $-$ VaR$(10\%)$")
    a1.set_title("The gap between the levels you asked for", fontsize=13)
    datefmt(a1)

    # --- right: what you get if you take the number -------------------------------
    r = pd.read_csv(LAB / f"bench_{ASSET}.csv", parse_dates=["date"])
    r = r[np.isclose(r["level"], ALPHA)].groupby("date")["ret"].first()
    idx = piv.index.intersection(r.index)
    rate = 100 * float((r.reindex(idx).values < -piv.loc[idx, ALPHA].values).mean())
    a2.bar(["asked for", "got"], [100 * ALPHA, rate], width=0.55,
           color=[MODEL_COLORS["GARCH-t"], al.IDA_RED])
    for i, v in enumerate([100 * ALPHA, rate]):
        a2.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9.4)
    a2.set_ylim(0, rate * 1.3)
    a2.set_ylabel("Breach rate (%)")
    a2.set_title("Take the number and score it", fontsize=13)
    a2.grid(axis="x", visible=False)

    fig_legend(fig, a1, ncol=2)
    save(fig, "s_clamp")

    share = [float(np.isclose(piv[x], piv[0.10], rtol=0, atol=0).mean())
             for x in (0.01, 0.05, 0.10)]
    return {"clamp_share_05": share[1], "n_dates": len(piv),
            "clamp_gap_max": round(float((piv[0.01] - piv[0.10]).abs().max()), 6),
            "clamp_ref_gap_mean": round(
                float((ref.loc[common, 0.01] - ref.loc[common, 0.10]).mean()), 3),
            "clamp_realised_rate": round(rate, 1)}


def fig_thresholds(bench, models):
    """One asset, one test span, four thresholds."""
    keys = [k for k in ("HS", "GARCH-t", "Chronos-T5", "LLM-series") if k in models]
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(bench.index, bench["ret"], lw=0.6, color=RETURNS,
            label="Realised return", zorder=1)
    for k in keys:
        v = models[k].dropna()
        ax.plot(v.index, -v.values, lw=1.6, color=MODEL_COLORS.get(k, GREY),
                label=line_label(k), zorder=2)
    ax.set_ylabel("Return and\n$-$VaR$(1\%)$, %")
    datefmt(ax)
    # No axes title: the frame title already says what the panel shows, and a second
    # copy of it inside the figure only eats the panel.
    legend_below(ax, ncol=3, y=-0.16, fontsize=9.4)
    save(fig, "s_thresholds")
    return {}


def fig_backtest(bench, models):
    """Breach counts against the acceptance band, and the pinball ranking.

    PRIMARY only. The 3B, 7B and 14B runs are a size diagnostic for one family and the
    deck says so; putting them in a ranking chart makes the chart disagree with the
    headline table beside it, and with this file's own stated convention.
    """
    prim = {k: v for k, v in models.items() if k in PRIMARY}
    bt = al.backtest_table(bench["ret"], prim, alpha=ALPHA)
    lb = al.leaderboard(bt, alpha=ALPHA)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_TWO)
    b = bt.sort_values("observed")
    n = int(b["n"].max())
    exp, se = ALPHA * n, np.sqrt(ALPHA * (1 - ALPHA) * n)
    a1.axvspan(exp - 1.96 * se, exp + 1.96 * se, color=al.FOREST, alpha=0.13,
               label="95% band")
    a1.axvline(exp, color=al.FOREST, lw=1.5, label=f"expected ({exp:.0f})")
    a1.barh([axis_label(m) for m in b["model"]], b["observed"],
            color=[MODEL_COLORS.get(m, GREY) for m in b["model"]], height=0.62)
    a1.set_xlabel(f"Breaches in {n} days at {ALPHA:.0%}")
    a1.grid(axis="y", visible=False)
    for _a in (a1, a2):
        _a.tick_params(axis="y", labelsize=7)
    a1.set_title("Coverage")

    d = lb.sort_values("pinball", ascending=False)
    a2.barh([axis_label(m) for m in d["model"]], d["pinball"],
            color=[MODEL_COLORS.get(m, GREY) for m in d["model"]],
            hatch=["//" if c == "reject" else "" for c in d["CC"]], height=0.62)
    a2.set_xlabel("Mean pinball loss")
    a2.grid(axis="y", visible=False)
    a2.set_title("Ranking (hatched: rejected)", fontsize=10.1)

    fig_legend(fig, a1, ncol=2)
    save(fig, "s_backtest")

    bt.to_csv(PRECOMP / f"lab_backtest_{ASSET}.csv", index=False)
    best = lb.iloc[0]
    facts = {"best_model": best["model"], "best_pinball": best["pinball"],
             "n_reject_cc": int((lb["CC"] == "reject").sum()), "n_models": len(lb),
             "n_days": n}

    # A ranking without a test on the loss differential is an ordering, not a finding.
    top = best["model"]
    for m in lb["model"][1:]:
        t, p, d, nn, lag = al.dm_test(bench["ret"], prim[top], prim[m], ALPHA)
        facts[f"dm_p_{top}_vs_{m}"] = round(p, 4)
        facts[f"dm_diff_{top}_vs_{m}"] = round(d, 5)
    al.dm_matrix(bench["ret"], prim, ALPHA).round(4).to_csv(
        PRECOMP / f"lab_dm_{ASSET}.csv")
    return facts


def fig_disagreement(models):
    """Spread AND dependence, because a wide band is not evidence of either.

    The band alone cannot support the SYNCRISK argument: two forecasts can differ in
    level every day and still move together every day, which is the case that matters
    for concentration risk. The right panel therefore reports the pairwise correlation
    matrix of the daily forecasts and the share of its variance in the first principal
    component, which is the quantity the argument actually needs.
    """
    prim = {k: v for k, v in models.items() if k in PRIMARY}
    sp = al.disagreement(prim)
    wide = pd.DataFrame({k: pd.Series(v).astype(float)
                         for k, v in prim.items()}).dropna()
    R = wide.corr()
    off = R.values[np.triu_indices_from(R.values, k=1)]
    lam = np.linalg.eigvalsh(R.values)[::-1]
    pc1 = float(lam[0] / len(R))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_FLAT,
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    a1.fill_between(sp.index, -sp["max"], -sp["min"], color=al.MAIN_BLUE, alpha=0.18,
                    label="Range across models")
    a1.plot(sp.index, -sp["mean"], color=al.MAIN_BLUE, lw=1.5, label="Mean forecast")
    a1.set_ylabel(r"$-$VaR$(1\%)$, %")
    a1.set_title("Level: the band across nine models")
    datefmt(a1)

    # Levels AND first differences. Every VaR series is persistent -- HS has lag-1
    # autocorrelation 0.999 -- so a correlation of levels measures shared trend as much
    # as shared reaction. The honest comparison puts both side by side.
    dR = wide.diff().dropna().corr()
    doff = dR.values[np.triu_indices_from(dR.values, k=1)]
    dpc1 = float(np.linalg.eigvalsh(dR.values)[::-1][0] / len(dR))

    lbl = ["levels", "changes"]
    vals = [[float(np.median(off)), pc1], [float(np.median(doff)), dpc1]]
    x, w = np.arange(2), 0.34
    a2.bar(x - w / 2, [vals[0][0], vals[1][0]], width=w, color=al.MAIN_BLUE,
           label=r"median pairwise $\rho$")
    a2.bar(x + w / 2, [vals[0][1], vals[1][1]], width=w, color=al.IDA_RED,
           label=r"$\lambda_1(R)/N$")
    a2.axhline(1 / len(R), color=GREY, lw=1.6, ls="--",
               label=f"independence: {1 / len(R):.2f}")
    for xi, v in zip(x - w / 2, [vals[0][0], vals[1][0]]):
        a2.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9.4, color=al.MAIN_BLUE)
    for xi, v in zip(x + w / 2, [vals[0][1], vals[1][1]]):
        a2.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9.4, color=al.IDA_RED)
    a2.set_xticks(x)
    a2.set_xticklabels(lbl)
    a2.set_ylim(0, 0.68)
    a2.set_ylabel("Dependence")
    a2.set_title("Most of it is persistence", fontsize=10.1)
    a2.grid(axis="x", visible=False)

    fig_legend(fig, [a1, a2], ncol=3)
    save(fig, "s_disagreement")
    return {"median_rho_changes": round(float(np.median(doff)), 2),
            "pc1_share_changes": round(dpc1, 2),
            "median_pairwise_rho": round(float(np.median(off)), 2),
            "min_pairwise_rho": round(float(off.min()), 2),
            "pc1_share": round(pc1, 2),
            "disagreement_models": len(R),
            "median_range": sp["range"].median(), "median_ratio": sp["ratio"].median(),
            "max_range": sp["range"].max(), "max_range_date": sp["range"].idxmax()}


def fig_llm(bench, models):
    """What the LLM's number is actually tracking.

    Left: the threshold the model states, day by day, against the realised return and
    against GARCH-t. Only the anonymised configuration is drawn -- the frame's claim is
    about that one, and six lines on a panel this size are a tangle rather than an
    argument.

    Right: the claim itself, for all four configurations at once. Each is correlated far
    more with the GARCH scale than with historical simulation, which is what "it tracks
    volatility, the factor that has data" means. A 6x6 correlation matrix said the same
    thing in cell labels too small to read; the matrix that earns its place is the one in
    s_disagreement, where the question really is about every pair.
    """
    keys = [k for k in models if k.startswith("LLM")]
    if not keys or "HS" not in models:
        print("  skip s_llm: no LLM run")
        return {}
    wide = pd.DataFrame({k: models[k] for k in
                         [*keys, "HS", "GARCH-t"] if k in models}).dropna()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_TWO,
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    # --- left: the stated threshold, against the two references -------------------
    a1.plot(bench.index, bench["ret"], lw=0.6, color=RETURNS, label="Realised return")
    shown = "LLM-series" if "LLM-series" in models else keys[0]
    v = models[shown].dropna()
    a1.plot(v.index, -v.values, lw=1.6, color=MODEL_COLORS.get(shown, GREY),
            label=line_label(shown))
    a1.plot(models["GARCH-t"].index, -models["GARCH-t"].values, lw=1.4,
            color=MODEL_COLORS["GARCH-t"], label=axis_label("GARCH-t"))
    a1.set_ylabel("Return and\n$-$VaR$(1\%)$, %")
    a1.set_title("It moves, day by day", fontsize=10.1)
    datefmt(a1)

    # --- right: what each configuration is correlated with ------------------------
    order = [k for k in ("LLM-series", "LLM-series+state", "LLM-dated",
                         "LLM-dated+news") if k in wide.columns]
    rho_g = [float(wide[k].corr(wide["GARCH-t"])) for k in order]
    rho_h = [float(wide[k].corr(wide["HS"])) for k in order]
    x, w = np.arange(len(order)), 0.36
    a2.bar(x - w / 2, rho_g, width=w, color=MODEL_COLORS["GARCH-t"],
           label=r"$\rho$ with GARCH-$t$")
    a2.bar(x + w / 2, rho_h, width=w, color=GREY,
           label=r"$\rho$ with HS")
    a2.axhline(0, color=RETURNS, lw=1)
    a2.set_xticks(x)
    a2.set_xticklabels([matrix_label(k) for k in order], fontsize=8.6)
    a2.set_ylabel(r"Correlation $\rho$")
    a2.set_ylim(min(0, min(rho_h)) - 0.12, 1.0)
    a2.set_title("All four track the scale", fontsize=10.1)
    a2.grid(axis="x", visible=False)

    fig_legend(fig, [a1, a2], ncol=3)
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
    t_cov, p_cov, d_cov, n_cov, _lag_cov = al.dm_test(
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

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_FLAT,
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    # Coverage BY QUARTER, not day by day. 500 daily bars render as one solid block,
    # and the count per covered day is nearly constant because the feed caps at five.
    # What actually varies -- and what the as-of design has to live with -- is when the
    # covered days fall: one quarter is barely covered at all.
    q = pd.DataFrame({"cov": (cov > 0).astype(int), "n": 1},
                     index=bench.index).resample("QE").sum()
    share = 100 * q["cov"] / q["n"]
    lo_q = share.idxmin()
    a1.bar(range(len(share)), share.values, width=0.7,
           color=[al.IDA_RED if v < 60 else al.MAIN_BLUE for v in share.values])
    a1.set_xticks(range(len(share)))
    a1.set_xticklabels([f"{d.year}\nQ1" if d.quarter == 1 else f"Q{d.quarter}"
                        for d in share.index], fontsize=7.8)
    a1.set_ylim(0, 112)
    a1.set_ylabel("Trading days covered (%)")
    a1.set_title(f"{len(covered)} of {len(bench)}, unevenly", fontsize=10.1)
    a1.grid(axis="x", visible=False)
    a1.annotate(f"{share.min():.0f}%", xy=(list(share.index).index(lo_q), share.min()),
                xytext=(list(share.index).index(lo_q), share.min() + 16),
                fontsize=9.4, color=al.IDA_RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=al.IDA_RED, lw=1.3))

    a2.axhline(0, color=GREY, lw=1)
    a2.plot(diff.index, diff.cumsum().values, color=al.IDA_RED, lw=1.8)
    a2.set_ylabel("Cumulative pinball\nloss difference")
    # The differential and its p-value are in the slide caption; the title says which
    # comparison the curve is, and stops there.
    a2.set_title("With headlines minus without", fontsize=11.7)
    datefmt(a2)
    save(fig, "s_news")

    # A p-value says the gap is not noise; it does not say how big the gap is. The
    # loss differential is reported against the control's own mean loss, with the HAC
    # standard error the DM statistic already implies, so the slide can quote an
    # effect size and an interval instead of a verdict.
    # One day can carry a mean. The differential is recomputed with the largest
    # absolute daily difference removed, because a cumulative curve with a visible step
    # is exactly the case where the average is not the typical day.
    worst_day = diff.abs().idxmax()
    keep = diff.drop(worst_day).index
    _, p_drop, d_drop, n_drop, _ = al.dm_test(
        bench.loc[keep, "ret"], a.reindex(keep).dropna(), b.reindex(keep).dropna(), ALPHA)
    share_worst = float(abs(diff.loc[worst_day]) / abs(diff.sum()))

    base = float(loss(b).mean())
    se_cov = abs(d_cov / t_cov) if t_cov else np.nan
    rel = -100 * d_cov / base
    rel_lo = -100 * (d_cov + 1.96 * se_cov) / base
    rel_hi = -100 * (d_cov - 1.96 * se_cov) / base

    return {"news_worst_day": str(worst_day.date()),
            "news_worst_day_share": round(share_worst, 3),
            "news_diff_drop1": round(d_drop, 5),
            "news_p_drop1": round(p_drop, 3),
            "news_rel_drop1_pct": round(-100 * d_drop / float(loss(b).mean()), 1),
            "news_cov_min_quarter_pct": round(float(share.min()), 1),
            "news_cov_quarters_full": int((share >= 90).sum()),
            "news_control_loss": round(base, 5),
            "news_rel_pct": round(rel, 1),
            "news_rel_lo_pct": round(rel_lo, 1),
            "news_rel_hi_pct": round(rel_hi, 1),
            "news_dm_lag": int(_lag_cov),
            "news_dates_covered": int(len(covered)),
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

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_TWO,
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    a1.plot(bench.index, bench["ret"], lw=0.6, color=RETURNS, label="Realised return")
    # Left panel: the two legs the laboratory actually runs. The larger models are a
    # size sweep, and belong in the bar panel rather than as four more lines here.
    shown = [k for k in ("LLM-series", "Open-1.5B") if k in models]
    for k in shown:
        v = models[k].dropna()
        a1.plot(v.index, -v.values, lw=1.5, color=MODEL_COLORS.get(k, GREY),
                label=line_label(k))
    a1.set_ylabel("Return and\n$-$VaR$(1\%)$, %")
    a1.set_title(f"Same prompt, {'two' if len(shown) == 2 else len(shown)} models")
    datefmt(a1)

    names = ["HS", "LLM-series", *keys]
    frac = [100 * models[k].round(2).nunique() / max(models[k].notna().sum(), 1)
            for k in names]
    # Short bar labels: the long form is unreadable once the panel is on a slide.
    short = {"LLM-series": "Haiku", "Open-1.5B": "1.5B", "Open-3B": "3B",
             "Open-7B": "7B", "Open-14B": "14B"}
    a2.tick_params(axis="x", labelsize=7.8)
    a2.bar([short.get(k, k) for k in names], frac,
           color=[MODEL_COLORS.get(k, GREY) for k in names], width=0.6)
    a2.set_ylabel("Distinct VaR values\n(% of forecast days)")
    a2.set_ylim(0, 112)
    a2.set_title("How varied is the output?", fontsize=10.1)
    a2.grid(axis="x", visible=False)
    for i, v in enumerate(frac):
        a2.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=9.4)

    fig_legend(fig, a1, ncol=2, fontsize=8.6)
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

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    col = [al.FOREST if x < 0 else al.IDA_RED for x in r["diff"]]
    ax.barh(r["label"], r["diff"], color=col, height=0.6)
    ax.axvline(0, color=GREY, lw=1.2)
    # Every annotation starts to the right of zero, whichever way its bar points: a
    # label trailing left off a negative bar runs straight over the asset names.
    for i, (d_, p, n) in enumerate(zip(r["diff"], r["p"], r["n"])):
        ax.text(max(d_, 0) + 0.00025, i, f"p = {p:.3f}  (n = {n})",
                va="center", ha="left", fontsize=9.4)
    ax.set_xlim(min(r["diff"].min() * 1.9, -0.0035), max(r["diff"].max() * 2.6, 0.005))
    ax.set_xlabel("Mean pinball loss, with headlines minus without\n"
                  "(negative = headlines helped)")
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
    weights, same parser, same scoring. It is the only fully controlled experiment in
    the package, so it is the one place a difference can be attributed to the
    extraction method rather than to the model.

    THE DAY SET IS INTERSECTED, and that is what makes the experiment controlled. The
    elicited leg loses the days on which the model returned the two quantiles inverted
    -- six on the S&P, fifteen on Bitcoin, none elsewhere -- while the sampled leg
    never can, because a quantile read off sorted draws is ordered by construction.
    Scoring each leg on its own surviving days would compare 500 days against 494 and
    call it a controlled experiment. Both legs are therefore scored on the dates valid
    for BOTH, and n is reported per asset.

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
        sv, ev = s.loc[_usable(s), "var_01"], e.loc[_usable(e), "var"]
        common = sv.index.intersection(ev.index).intersection(ret.index)
        m = {"sampled": sv.reindex(common), "elicited": ev.reindex(common)}
        bt = al.backtest_table(ret.reindex(common), m, alpha=ALPHA).set_index("model")
        for k in m:
            rows.append({"asset": a, "label": ASSETS[a]["label"], "method": k,
                         "n_common": len(common),
                         "n_dropped": int(500 - len(common)),
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

    # Short tick labels: five full asset names side by side under a half-panel axis
    # overlap each other, and the legend already names the two methods.
    short_asset = {"SPX": "S&P", "DAX": "DAX", "N225": "Nikkei", "GOLD": "Gold",
                   "BTC": "BTC"}
    ticks = [f"{short_asset.get(a, a)}\n$n$ = {int(piv['sampled'].loc[a, 'n_common'])}"
             for a in assets]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_TWO)
    for k, off in (("sampled", -0.5), ("elicited", 0.5)):
        a1.bar(x + off * w, piv[k].loc[assets, "pinball"], width=w, color=col[k],
               alpha=0.9, label=lab[k])
    a1.set_xticks(x)
    a1.set_xticklabels(ticks, fontsize=7.8)
    a1.set_ylabel("Pinball loss at 1%")
    a1.set_title("Sampling scores better", fontsize=10.1)
    a1.grid(axis="x", visible=False)

    for k, off in (("sampled", -0.5), ("elicited", 0.5)):
        a2.bar(x + off * w, piv[k].loc[assets, "rate"], width=w, color=col[k], alpha=0.9,
               label=lab[k])
    a2.axhline(100 * ALPHA, color=al.IDA_RED, lw=1.6, ls="--",
               label=f"nominal {ALPHA:.0%}")
    a2.set_xticks(x)
    a2.set_xticklabels(ticks, fontsize=7.8)
    a2.set_ylabel("Breach rate (%)")
    a2.set_title("Both breach far more than 1%", fontsize=10.1)
    a2.grid(axis="x", visible=False)

    fig_legend(fig, a2, ncol=2)
    save(fig, "s_sampling")

    r.round(4).to_csv(PRECOMP / "lab_sampling_vs_elicited.csv", index=False)
    best = (piv["sampled"].loc[assets, "pinball"] < piv["elicited"].loc[assets, "pinball"])
    return {"samp_assets": len(assets),
            "samp_pinball_wins": int(best.sum()),
            "samp_n_common_SPX": int(piv["sampled"].loc["SPX", "n_common"]),
            "samp_n_common_min": int(r["n_common"].min()),
            "samp_n_dropped_total": int(piv["sampled"]["n_dropped"].sum()),
            "samp_rejected_both": int((r["p_uc"] < 0.05).sum()),
            "samp_distinct_SPX": round(float(piv["sampled"].loc["SPX", "distinct"]), 1),
            "elic_distinct_SPX": round(float(piv["elicited"].loc["SPX", "distinct"]), 1),
            "samp_rate_min": round(float(r["rate"].min()), 2),
            "samp_rate_max": round(float(r["rate"].max()), 2)}


def fig_coherent():
    """Why sub-additivity is the axiom VaR loses, on the smallest example that shows it.

    Two independent bonds, each losing L on default with probability p < alpha. Held
    alone, the default sits BEYOND the alpha-quantile, so VaR reports nothing. Held
    together, the chance that at least one defaults crosses alpha, the quantile lands on
    a default, and VaR jumps from zero to L. Merging the two books raised the measured
    risk, which is what sub-additivity forbids.

    ES reads the same two books correctly, because it averages the tail instead of
    reading its edge: it already saw the default in the single-bond book.

    Constructed, exact, and independent of any estimate -- the point is a property of
    the functional, not a finding about markets.
    """
    p_def, LOSS = 0.008, 100.0

    def var_es(states):
        """VaR = the alpha-quantile of the loss; ES = mean of the worst alpha."""
        cum, acc, var = 0.0, 0.0, None
        for loss, pr in sorted(states, key=lambda x: -x[0]):
            take = min(pr, ALPHA - cum)
            if take > 0:
                acc += take * loss
                cum += take
            if cum >= ALPHA - 1e-15 and var is None:
                var = loss
        return var, acc / ALPHA

    one = [(LOSS, p_def), (0.0, 1 - p_def)]
    two = [(2 * LOSS, p_def ** 2), (LOSS, 2 * p_def * (1 - p_def)),
           (0.0, (1 - p_def) ** 2)]
    v1, e1 = var_es(one)
    v2, e2 = var_es(two)
    p_any = 1 - (1 - p_def) ** 2

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_TWO)

    # --- left: the mechanism, one probability crossing alpha ----------------------
    bars = [100 * p_def, 100 * p_any]
    a1.bar(["One bond", "Two bonds"], bars, width=0.5,
           color=[al.FOREST, al.IDA_RED])
    a1.set_xlabel("P(at least one default)")
    a1.axhline(100 * ALPHA, color=al.MAIN_BLUE, lw=2, ls="--",
               label=fr"$\alpha = {ALPHA:.0%}$".replace("%", r"\%"))
    for i, v in enumerate(bars):
        a1.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9.4)
    a1.set_ylim(0, 100 * p_any * 1.35)
    a1.set_ylabel("Probability (%)")
    a1.set_title("Crossing $\\alpha$ is what moves VaR", fontsize=11.7)
    a1.grid(axis="x", visible=False)

    # --- right: the axiom, tested ------------------------------------------------
    x, w = np.arange(2), 0.34
    sums, ports = [2 * v1, 2 * e1], [v2, e2]
    a2.bar(x - w / 2, sums, width=w, color=GREY, label="Sum of the two books")
    a2.bar(x + w / 2, ports, width=w, color=al.MAIN_BLUE,
           label="The merged book")
    for xi, v in zip(x - w / 2, sums):
        a2.text(xi, v + 3, f"{v:.0f}", ha="center", fontsize=9.4, color=GREY)
    for xi, v in zip(x + w / 2, ports):
        a2.text(xi, v + 3, f"{v:.2f}", ha="center", fontsize=9.4, color=al.MAIN_BLUE)
    # The violation is the one thing a reader must not have to infer from bar heights.
    a2.annotate("merging raised it", xy=(w / 2, ports[0] * 0.72),
                xytext=(-0.12, ports[0] * 1.34), fontsize=9.4, color=al.IDA_RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=al.IDA_RED,
                                             lw=1.4, shrinkB=2))
    a2.set_xticks(x)
    a2.set_xticklabels([f"VaR at {ALPHA:.0%}", f"ES at {ALPHA:.0%}"])
    a2.set_ylim(0, max(sums + ports) * 1.22)
    a2.set_ylabel("Capital (loss units)")
    a2.set_title("Sub-additivity caps it at the sum", fontsize=10.1)
    a2.grid(axis="x", visible=False)

    fig_legend(fig, [a1, a2], ncol=2, fontsize=9.4)
    save(fig, "s_coherent")

    return {"coh_p_default": p_def, "coh_p_any": round(p_any, 6),
            "coh_var_one": v1, "coh_var_two": v2,
            "coh_es_one": round(e1, 2), "coh_es_two": round(e2, 2),
            "coh_var_subadditive": bool(v2 <= 2 * v1),
            "coh_es_subadditive": bool(e2 <= 2 * e1)}


def fig_data(rets, bench):
    """The dataset itself: what is estimated on, what is scored on, and how they differ.

    The deck scores 500 days. Those days are not a random sample of the history: their
    standard deviation is two thirds of the full sample's and their excess kurtosis is
    an order of magnitude smaller. Every verdict in Act V is a verdict on a calm span,
    and the figure says so before any of them is shown.
    """
    r = rets["ret"].dropna()
    t0, t1 = bench.index[0], bench.index[-1]
    test = r.loc[t0:t1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_DATA,
                                 gridspec_kw={"width_ratios": [1.55, 1]})

    # --- left: the whole series, with the scored span marked ----------------------
    a1.plot(r.index, r.values, lw=0.35, color=RETURNS)
    _pre = r.loc[:t0].iloc[:-1]                 # everything strictly before scoring
    a1.axvspan(r.index[0], t0, color=MODEL_COLORS["GARCH-t"], alpha=0.09,
               label=f"TRAIN  {_pre.index[0]:%Y-%m-%d} to {_pre.index[-1]:%Y-%m-%d}"
                     f"  ({len(_pre)} days)")
    a1.axvspan(t0, t1, color=al.IDA_RED, alpha=0.18,
               label=f"TEST  {t0:%Y-%m-%d} to {t1:%Y-%m-%d}  ({len(test)} days)")
    a1.set_ylabel("Return (%)")
    a1.set_title(f"{ASSETS[ASSET]['label']}, {r.index[0]:%Y}–{r.index[-1]:%Y}",
                 fontsize=12.5)
    a1.xaxis.set_major_locator(mdates.YearLocator(4))
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- right: the two distributions ---------------------------------------------
    grid = np.linspace(-6, 6, 400)
    for x, c, lab in ((r.values, RETURNS, "Full sample"),
                      (test.values, al.IDA_RED, "TEST span")):
        a2.plot(grid, stats.gaussian_kde(x)(grid), color=c, lw=2,
                label=f"{lab}: sd {x.std():.2f}")
    a2.set_yscale("log")
    a2.set_ylim(1e-4, 1)
    a2.set_xlabel("Daily return (%)")
    a2.set_ylabel("Density")
    a2.set_title("Calmer than history", fontsize=12.5)

    fig_legend(fig, [a1, a2], ncol=2)
    save(fig, "s_data")

    return {"data_n": len(r), "data_start": str(r.index[0].date()),
            "data_end": str(r.index[-1].date()),
            "test_start": str(t0.date()), "test_end": str(t1.date()),
            "sd_full": round(float(r.std()), 2), "sd_test": round(float(test.std()), 2),
            "kurt_full": round(float(stats.kurtosis(r)), 1),
            "kurt_test": round(float(stats.kurtosis(test)), 1)}


def fig_nnarch():
    """NN-t drawn: seven predictors into eight units, out to the scale only.

    The point of the picture is the asymmetry the previous frame argues for. The network
    produces one thing, the scale. The two tail parameters are fitted alongside it and
    never pass through it, so they are drawn off the graph entirely, and the two colours
    meet only in the quantile at the right.
    """
    fig, ax = plt.subplots(figsize=FIG_NN)
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    feats = [r"$r_{t-1}$", r"$|r_{t-1}|$", r"$\min_{22}$", r"$\mathrm{RV}^{(5)}$",
             r"$\mathrm{RV}^{(22)}$", r"$\mathrm{RV}^{(66)}$", r"EWMA"]
    xin, xh, xout = 1.35, 3.70, 5.85
    yin = np.linspace(3.45, 1.05, len(feats))
    yh = np.linspace(3.25, 1.25, 8)

    for a in yin:                                   # every input reaches every unit
        for b in yh:
            ax.plot([xin + 0.18, xh - 0.14], [a, b], color=GREY, lw=0.25, alpha=0.45,
                    zorder=1)
    for b in yh:
        ax.plot([xh + 0.14, xout - 0.18], [b, 2.25], color=GREY, lw=0.25, alpha=0.45,
                zorder=1)

    for a, lab in zip(yin, feats):
        ax.plot(xin, a, "o", ms=8, color="white", mec=al.MAIN_BLUE, mew=1.4, zorder=3)
        ax.text(xin - 0.32, a, lab, ha="right", va="center", fontsize=11)
    for b in yh:
        ax.plot(xh, b, "o", ms=8, color=al.MAIN_BLUE, mec=al.MAIN_BLUE, zorder=3)
    ax.plot(xout, 2.25, "o", ms=10, color=al.MAIN_BLUE, mec=al.MAIN_BLUE, zorder=3)

    ax.text(xin, 3.80, "7 predictors", ha="center", fontsize=11.5, color=al.MAIN_BLUE)
    ax.text(xh, 3.80, r"8 units, $\tanh$", ha="center", fontsize=11.5,
            color=al.MAIN_BLUE)

    # the head returns a bounded deviation from the window's own log sd
    ax.annotate("", xy=(7.15, 2.25), xytext=(xout + 0.22, 2.25),
                arrowprops=dict(arrowstyle="-|>", color=al.MAIN_BLUE, lw=1.6))
    ax.text(6.60, 2.60, "bounded", ha="center", fontsize=10.5, color=al.MAIN_BLUE)
    ax.text(6.60, 2.32, r"$s_{w}/4$ to $4s_{w}$", ha="center",
            fontsize=10.5, color=al.MAIN_BLUE)
    ax.text(7.30, 2.25, r"$\hat\sigma_{t}$   the scale", ha="left", va="center",
            fontsize=13, color=al.MAIN_BLUE)

    # the shape is fitted alongside the network, not through it
    ax.text(7.30, 0.72, r"$\hat\mu,\ \hat\nu$   the shape", ha="left", va="center",
            fontsize=13, color=al.IDA_RED)
    ax.text(7.30, 0.34, "two scalars, fitted jointly, outside the network", ha="left",
            va="center", fontsize=10, color=al.IDA_RED)

    # the two colours meet only here
    ax.annotate("", xy=(9.55, 1.72), xytext=(8.95, 2.10),
                arrowprops=dict(arrowstyle="-|>", color=al.MAIN_BLUE, lw=1.6))
    ax.annotate("", xy=(9.55, 1.28), xytext=(8.95, 0.88),
                arrowprops=dict(arrowstyle="-|>", color=al.IDA_RED, lw=1.6))
    ax.text(10.15, 1.50, r"$\hat q_{t}(\alpha)$" "\n" r"$=\hat\mu+\hat\sigma_{t}\,"
                         r"q_{\hat\nu}(\alpha)$",
            ha="center", va="center", fontsize=12, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.32", fc="#F2F4F7", ec=GREY, lw=0.7))

    save(fig, "s_nnarch")
    return {"nn_features": len(feats), "nn_hidden": len(yh)}


# Architecture diagrams. Every number in them is read out of the model's own config.json
# in the local HuggingFace cache by src/05_verify_slide_claims.py, so a diagram cannot
# describe a model the lab does not actually run.
def _box(ax, x, y, w, h, text, fc="white", ec=None, fontsize=10.5, tc=None, lw=1.2):
    ec = ec or GREY
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=tc or "black", zorder=3, linespacing=1.45)


def _arrow(ax, x0, x1, y, color=None):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color or GREY, lw=1.4))


def fig_chronosarch():
    """Chronos-T5 end to end: prices in, tokens, encoder-decoder, sampled paths out.

    The point of drawing it is that nothing in the chain is fitted here. The weights
    arrive trained, and the only thing the lab chooses is how many paths to draw, which
    is the knob the next frame measures.
    """
    fig, ax = plt.subplots(figsize=FIG_ARCH)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    xs = [1.35, 3.75, 6.30, 8.80, 11.25]
    w, y = 2.15, 1.35

    _box(ax, xs[0], y, w, 0.85, "512 closing\nprices")
    _box(ax, xs[1], y, w, 0.85, "scaled, binned into\n4096 tokens")
    _box(ax, xs[2], y, w, 0.85, "encoder-decoder\n4 + 4 layers, $d$ = 384",
         fc="#E8EEF7", ec=al.MAIN_BLUE)
    _box(ax, xs[3], y, w, 0.85, "sample 500 paths\none token at a time")
    _box(ax, xs[4], y, w, 0.85, "any level you ask\nfor", fc="#E8EEF7",
         ec=al.MAIN_BLUE, tc=al.MAIN_BLUE)
    for i in range(4):
        _arrow(ax, xs[i] + w / 2 + 0.04, xs[i + 1] - w / 2 - 0.04, y)

    ax.text(xs[2], 2.25, "Chronos-T5 mini, pretrained elsewhere", ha="center",
            fontsize=12, color=al.MAIN_BLUE, weight="bold")
    ax.text(xs[2], 0.42, "every parameter arrives fitted on other series", ha="center",
            fontsize=10.5, color=GREY)

    save(fig, "s_chronosarch")
    return {"arch_t5_layers": 4}

def fig_llmarch():
    """The language-model route to a quantile, drawn end to end.

    Every stage in the stack is general purpose. The number is a substring of generated
    text, and what makes it a quantile is the sentence asking for one.
    """
    fig, ax = plt.subplots(figsize=FIG_LLMARCH)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 3.1)
    ax.axis("off")

    xs = [1.35, 3.75, 6.30, 8.80, 11.25]
    w, y = 2.15, 1.95

    _box(ax, xs[0], y, w, 0.95, "60 daily returns\nwritten as text")
    _box(ax, xs[1], y, w, 0.95, "tokeniser\n151 936 symbols")
    _box(ax, xs[2], y, w, 0.95,
         "28 decoder blocks\n$d$ = 1536, 12 query\nheads, 2 key-value",
         fc="#E8EEF7", ec=al.MAIN_BLUE, fontsize=10)
    _box(ax, xs[3], y, w, 0.95, "next-token\ndistribution")
    _box(ax, xs[4], y, w, 0.95, "JSON reply, then\na parsed number",
         fc="#E8EEF7", ec=al.MAIN_BLUE)
    for i in range(4):
        _arrow(ax, xs[i] + w / 2 + 0.04, xs[i + 1] - w / 2 - 0.04, y)

    ax.text(xs[2], 2.80, "Qwen2.5-1.5B-Instruct, decoder-only", ha="center",
            fontsize=12, color=al.MAIN_BLUE, weight="bold")
    ax.text(xs[2], 1.06, "Every layer is general purpose. The request makes it a quantile.",
            ha="center", fontsize=11.5, color=al.IDA_RED)

    save(fig, "s_llmarch")
    return {"arch_qwen_layers": 28, "arch_qwen_dmodel": 1536}


def fig_vs_garch():
    """The best language-model configuration against the benchmark, on five markets.

    The leaderboard answers this for one asset. Five answers say whether the parity on
    the S&P is a property of the method or of that market. Bars are the loss difference,
    so left of zero is the language model ahead; the test is what decides whether a bar
    means anything.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tbl", ROOT / "src" / "10_slide_tables.py")
    tbl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tbl)

    rows = []
    for a in ASSETS:
        ret, mods = tbl.models_for(a)
        if "LLM-dated+news" not in mods or "GARCH-t" not in mods:
            continue
        bt = al.backtest_table(ret, mods, alpha=ALPHA).set_index("model")
        _, pv, dd, n, _ = al.dm_test(ret, mods["LLM-dated+news"], mods["GARCH-t"],
                                     alpha=ALPHA)
        rows.append({"label": ASSETS[a]["label"], "diff": dd, "p": pv, "n": n,
                     "g": bt.loc["GARCH-t", "pinball"],
                     "l": bt.loc["LLM-dated+news", "pinball"]})
    if not rows:
        print("  skip s_vs_garch: incomplete runs")
        return {}
    r = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    # colour carries the verdict, not the sign: grey where the test cannot separate
    col = [al.IDA_RED if p < 0.05 else GREY for p in r["p"]]
    ax.barh(r["label"], r["diff"], color=col, height=0.6)
    ax.axvline(0, color=RETURNS, lw=1.2)
    span = float(max(abs(r["diff"]))) or 1.0
    for y, (d_, pv) in enumerate(zip(r["diff"], r["p"])):
        ax.text(span * 1.12, y, f"p = {pv:.3f}", va="center", fontsize=10.5,
                color=al.IDA_RED if pv < 0.05 else RETURNS)
    ax.set_xlim(-span * 1.25, span * 1.95)
    ax.set_xlabel("Pinball loss at $1\\%$: Claude with headlines minus GARCH-$t$\n"
                  "(left of zero = the language model scored lower)")
    ax.grid(axis="y", visible=False)
    fig_legend(fig, ax, ncol=2)
    save(fig, "s_vs_garch")

    beaten = int((r["p"] < 0.05).sum())
    return {"vs_garch_assets": len(r), "vs_garch_separable": beaten,
            "vs_garch_llm_ahead_significantly": int(((r["p"] < 0.05) &
                                                     (r["diff"] < 0)).sum())}

def fig_allvar(bench, models):
    """Every scored forecast on one axis, so the spread is seen before it is measured.

    Ten lines is deliberately a lot. The point is that they answer the same question on
    the same days and do not agree, and that the disagreement is not noise around a
    common level: the open-weights runs sit an order of magnitude closer to zero, which
    is why they breach seven times too often.
    """
    keys = [k for k in ("HS", "GARCH-t", "NN-t", "Chronos-T5", "LLM-series",
                        "LLM-series+state", "LLM-dated", "LLM-dated+news",
                        "Open-1.5B", "LLMTime-1.5B") if k in models]
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(bench.index, bench["ret"], lw=0.5, color=RETURNS, alpha=0.55,
            label="Realised return", zorder=1)
    for k in keys:
        v = models[k].dropna()
        ax.plot(v.index, -v.values, lw=1.2, color=MODEL_COLORS.get(k, GREY),
                label=line_label(k), zorder=2)
    ax.set_ylabel("Return and\n$-$VaR$(1\\%)$, %")
    n = len(keys)
    ax.set_title(f"{ASSETS[ASSET]['label']}: {n} answers to one question, "
                 f"{len(bench)} days", fontsize=11)
    datefmt(ax)
    fig_legend(fig, ax, ncol=4, fontsize=8.6)
    save(fig, "s_allvar")

    lo = min(float(models[k].dropna().mean()) for k in keys)
    hi = max(float(models[k].dropna().mean()) for k in keys)
    return {"allvar_models": len(keys), "allvar_mean_lowest": round(lo, 2),
            "allvar_mean_highest": round(hi, 2)}

def fig_hs(rets, bench):
    """Historical simulation, drawn: sort the window, read off the quantile.

    Left is the estimator itself. The window is sorted and the alpha-quantile is the
    2.5th order statistic of 250 days, so the number the desk reports rests on the two
    or three worst observations in the window and on nothing else.

    Right is the consequence the frame claims: the estimate moves only when an extreme
    day enters or leaves, which makes the path a staircase, and where the steps fall
    depends on the window length rather than on the market.
    """
    r = rets["ret"].dropna()
    W_SHORT, W_LONG, SHOWN = 250, 1000, 40

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_TWO,
                                 gridspec_kw={"width_ratios": [1, 1.35]})

    # --- left: one window, sorted -------------------------------------------------
    win = r.loc[:bench.index[0]].iloc[-W_SHORT - 1:-1].values
    srt = np.sort(win)
    k = ALPHA * W_SHORT                       # 2.5 observations at 1% of 250 days
    var_hs = float(np.quantile(win, ALPHA))
    rank = np.arange(1, len(srt) + 1)
    a1.plot(rank, srt, color=RETURNS, lw=1.8, label="Sorted window")
    below = rank <= np.ceil(k)
    a1.plot(rank[below], srt[below], "o", color=al.IDA_RED, ms=7,
            label="Below the quantile")
    a1.axhline(var_hs, color=al.IDA_RED, lw=1.8, ls="--",
               label=f"$-$VaR $= {var_hs:.2f}$")
    a1.axvline(k, color=al.MAIN_BLUE, lw=1.4, ls=":",
               label=fr"rank $\alpha W = {k:.1f}$")
    a1.set_xlim(0.5, SHOWN + 0.5)
    # The curve runs over all W ranks; only the tail is on screen, so the vertical
    # range follows the tail rather than the whole window.
    lo, hi = float(srt[:SHOWN].min()), float(srt[:SHOWN].max())
    a1.set_ylim(lo - 0.35, hi + 0.35)
    a1.set_xlabel(f"Rank in the {W_SHORT}-day window")
    a1.set_ylabel("Daily return (%)")
    a1.set_title(f"The estimate rests on {int(np.ceil(k))} days", fontsize=11.7)

    # --- right: the same estimator at two window lengths ---------------------------
    for W, c, ls in ((W_SHORT, al.IDA_RED, "-"), (W_LONG, al.MAIN_BLUE, "--")):
        v = r.rolling(W).quantile(ALPHA).reindex(bench.index)
        a2.plot(v.index, v.values, color=c, lw=1.8, ls=ls,
                label=f"$W = {W}$ days")
    a2.plot(bench.index, bench["ret"], lw=0.6, color=RETURNS, zorder=0,
            label="Realised return")
    a2.set_ylabel("Threshold (%)")
    a2.set_title("It steps, and $W$ places the steps", fontsize=11.7)
    datefmt(a2)

    fig_legend(fig, [a1, a2], ncol=3, fontsize=9.4)
    save(fig, "s_hs")

    v250 = r.rolling(W_SHORT).quantile(ALPHA).reindex(bench.index)
    return {"hs_obs_below": int(np.ceil(k)),
            "hs_window_var": round(var_hs, 2),
            "hs_distinct_250": int(v250.round(2).nunique()),
            "hs_distinct_1000": int(r.rolling(W_LONG).quantile(ALPHA)
                                    .reindex(bench.index).round(2).nunique())}


def fig_garch(rets, bench):
    """GARCH(1,1)-t, drawn: a squared shock, and a threshold that moves every day.

    Left is the news impact curve of the FITTED model, not a sketch: sigma_t^2 as a
    function of yesterday's return is a parabola in that return, so a -10% day and a
    +10% day leave the same volatility behind. That is the property the frame states
    and the reason the model cannot express a leverage effect.

    Right is the contrast with historical simulation on the same days. GARCH moves
    every day because sigma_t does; HS holds still and then jumps.
    """
    from arch import arch_model

    r = rets["ret"].dropna()
    res = arch_model(r, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    w, a1_, b1 = (float(res.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
    nu = float(res.params["nu"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_TWO,
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    # --- left: the news impact curve ----------------------------------------------
    sbar = float(np.sqrt(res.conditional_volatility.mean() ** 2))
    shock = np.linspace(-10, 10, 601)
    nic = np.sqrt(w + a1_ * shock ** 2 + b1 * sbar ** 2)
    ax1.plot(shock, nic, color=al.MAIN_BLUE, lw=2.2, label="Next-day $\\sigma_t$")
    for sgn, c in ((-10, al.IDA_RED), (10, al.FOREST)):
        y = float(np.sqrt(w + a1_ * sgn ** 2 + b1 * sbar ** 2))
        ax1.plot([sgn], [y], "o", color=c, ms=9)
        ax1.annotate(f"{sgn:+d}%", xy=(sgn, y), xytext=(sgn * 0.62, y * 1.02),
                     fontsize=10.1, color=c, ha="center")
    ax1.axhline(float(np.sqrt(w + a1_ * 100 + b1 * sbar ** 2)), color=GREY, lw=1.2,
                ls=":", label="Same $\\sigma_t$ either way")
    ax1.set_xlabel("Yesterday's return $r_{t-1}$ (%)")
    ax1.set_ylabel("Today's $\\sigma_t$ (%)")
    ax1.set_ylim(top=float(nic.max()) * 1.16)
    ax1.set_title(f"Fitted $\\alpha_1 = {a1_:.3f}$, $\\beta_1 = {b1:.3f}$", fontsize=11.7)

    # --- right: continuous against discrete ---------------------------------------
    ax2.plot(bench.index, bench["ret"], lw=0.6, color=RETURNS, zorder=0,
             label="Realised return")
    ax2.plot(bench.index, -bench["GARCH-t"], color=al.MAIN_BLUE, lw=1.7,
             label="GARCH-$t$: moves daily")
    ax2.plot(bench.index, -bench["HS"], color=al.IDA_RED, lw=1.7, ls="--",
             label="HS: steps")
    ax2.set_ylabel("Threshold (%)")
    ax2.set_title("The scale moves daily", fontsize=10.1)
    datefmt(ax2)
    fig_legend(fig, [ax1, ax2], ncol=2, fontsize=9.4)
    save(fig, "s_garch")

    return {"garch_omega": round(w, 4), "garch_alpha1": round(a1_, 3),
            "garch_beta1": round(b1, 3), "garch_nu": round(nu, 2),
            "garch_persistence": round(a1_ + b1, 3)}


def fig_vares(rets):
    """What VaR and ES are, drawn, and why the deck keeps saying they are different.

    Left: orientation. The 1% return quantile is a THRESHOLD, one number on the axis;
    expected shortfall is the MEAN OF WHAT LIES BEYOND IT, a property of the shaded
    region rather than of its edge. Shown against the whole distribution so the reader
    sees how far out both objects sit.

    Right: the tail itself, magnified, which is the only way to see it on a linear
    axis -- at 1% the region carries a five-hundredth of the mass. Two laws are scaled
    to share the 1% quantile EXACTLY, so VaR is identical by construction and the only
    thing left to differ is the mass behind it. A ranking on VaR alone cannot separate
    them. This panel is constructed, not estimated, and the slide says so.
    """
    r = rets["ret"].dropna().values
    q = float(np.quantile(r, ALPHA))
    es = float(r[r <= q].mean())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_TWO)

    # --- left: where the two objects sit on the realised distribution -------------
    grid = np.linspace(r.min(), 4, 800)
    d = stats.gaussian_kde(r)(grid)
    a1.plot(grid, d, color=RETURNS, lw=2.0, label="Realised returns")
    # The region, not the sliver of area under it: on a linear density axis the 1% tail
    # is a five-hundredth of the mass and shading under the curve shows nothing. ES
    # averages over this BAND, so draw the band.
    tail = grid <= q
    a1.axvspan(r.min(), q, color=al.IDA_RED, alpha=0.13,
               label=f"The {ALPHA:.0%} tail")
    a1.fill_between(grid[tail], d[tail], color=al.IDA_RED, alpha=0.75)
    a1.axvline(q, color=al.IDA_RED, lw=2, label=f"$-$VaR $= {q:.2f}$")
    a1.axvline(es, color=al.MAIN_BLUE, lw=2, ls="--", label=f"$-$ES $= {es:.2f}$")
    a1.set_xlim(r.min(), 4)
    a1.set_ylim(0, float(d.max()) * 1.12)
    a1.set_xlabel("Daily return (%)")
    a1.set_ylabel("Density")
    a1.set_title(f"S&P 500, {len(r)} days", fontsize=11.7)

    # --- right: the two numbers side by side --------------------------------------
    # Not a second density. Matched at the 1% quantile the Gaussian is the WIDER law,
    # so near the threshold it sits above the t3 and a density picture reads backwards;
    # the heavier t3 tail lives far out, where on a linear axis both curves are zero.
    # The quantity the panel is about is the pair of numbers, so plot the pair.
    zn, nu = stats.norm.ppf(ALPHA), 3
    zt = stats.t.ppf(ALPHA, nu)
    sn, st_ = q / zn, q / zt
    es_n = -sn * stats.norm.pdf(zn) / ALPHA
    es_t = -st_ * (nu + zt ** 2) / (nu - 1) * stats.t.pdf(zt, nu) / ALPHA

    x, w = np.arange(2), 0.34
    # Colour means the same thing in both panels: red is VaR, blue is ES. Colouring
    # by law instead would put a red bar under a blue legend swatch.
    a2.bar(x - w / 2, [-q, -q], width=w, color=al.IDA_RED, label="VaR")
    a2.bar(x + w / 2, [-es_n, -es_t], width=w, color=al.MAIN_BLUE, label="ES")
    for xi, v in zip(x - w / 2, [-q, -q]):
        a2.text(xi, v + 0.12, f"{v:.2f}", ha="center", fontsize=9.4, color=al.IDA_RED)
    for xi, v in zip(x + w / 2, [-es_n, -es_t]):
        a2.text(xi, v + 0.12, f"{v:.2f}", ha="center", fontsize=9.4,
                color=al.MAIN_BLUE)
    a2.set_xticks(x)
    a2.set_xticklabels(["Gaussian", f"Student-$t_{{{nu}}}$"])
    a2.set_ylabel("Loss (%), positive")
    a2.set_ylim(0, max(-es_t, -es_n) * 1.28)
    a2.grid(axis="x", visible=False)
    a2.set_title(f"Both scaled to VaR $= {-q:.2f}$", fontsize=11.7)
    a2.legend(loc="upper left", fontsize=8.6, frameon=False)

    fig_legend(fig, a1, ncol=3, fontsize=9.4)
    save(fig, "s_vares")

    return {"vares_q01": round(q, 2), "vares_es01": round(es, 2),
            "vares_es_gauss": round(es_n, 2), "vares_es_t3": round(es_t, 2),
            "vares_es_ratio": round(es_t / es_n, 2)}


def fig_qr(rets):
    """The loss decides what you fit, on one scatter.

    Same returns, same predictor, same linear f: only the loss changes. Squared error
    puts the line at the conditional mean, which for daily returns is flat and useless.
    The pinball loss at 1% puts it at the conditional quantile, which slopes steeply in
    volatility -- the scale/shape decomposition, visible in one picture.
    """
    import statsmodels.api as sm

    r = rets["ret"].dropna()
    vol = r.rolling(20).std().shift(1)          # known at t-1
    d = pd.DataFrame({"y": r, "x": vol}).dropna()
    X = sm.add_constant(d["x"])
    ols = sm.OLS(d["y"], X).fit()
    q01 = sm.QuantReg(d["y"], X).fit(q=ALPHA)
    q05 = sm.QuantReg(d["y"], X).fit(q=ALPHA_SECONDARY)

    fig, ax = plt.subplots(figsize=FIG_ONE)
    ax.scatter(d["x"], d["y"], s=2, color=RETURNS, alpha=0.18, linewidths=0,
               label="Daily returns")
    g = np.linspace(d["x"].min(), d["x"].quantile(0.995), 100)
    G = sm.add_constant(pd.Series(g, name="x"))
    ax.plot(g, ols.predict(G), color=GREY, lw=2.4,
            label=f"Squared error: slope {ols.params['x']:+.2f}")
    ax.plot(g, q05.predict(G), color=MODEL_COLORS["GARCH-t"], lw=2.0, ls="--",
            label=f"Pinball {ALPHA_SECONDARY:.0%}: slope {q05.params['x']:+.2f}")
    ax.plot(g, q01.predict(G), color=al.IDA_RED, lw=2.6,
            label=f"Pinball {ALPHA:.0%}: slope {q01.params['x']:+.2f}")
    ax.set_xlim(d["x"].min(), d["x"].quantile(0.995))
    ax.set_ylim(-9, d["y"].quantile(0.999))
    ax.set_xlabel("20-day volatility at $t-1$ (%)")
    ax.set_ylabel("Return at $t$ (%)")
    ax.set_title("One predictor, one linear $f$, three losses", fontsize=13)

    fig_legend(fig, ax, ncol=2)
    save(fig, "s_qr")

    return {"qr_n": int(len(d)),
            "qr_slope_mean": round(float(ols.params["x"]), 3),
            "qr_slope_05": round(float(q05.params["x"]), 2),
            "qr_slope_01": round(float(q01.params["x"]), 2)}


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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_TWO)

    err = np.linspace(-3, 3, 601)
    for lev, c in ((ALPHA, al.IDA_RED), (ALPHA_SECONDARY, al.MAIN_BLUE)):
        a1.plot(err, np.where(err < 0, (lev - 1) * err, lev * err), lw=2, color=c,
                label=fr"$\alpha = {lev:.0%}$".replace("%", r"\%"))
    a1.axvline(0, color=GREY, lw=1, zorder=0)
    a1.annotate(r"slope $\alpha - 1$", xy=(-2.2, 2.15), fontsize=9.4, color=GREY)
    a1.annotate(r"slope $\alpha$", xy=(1.2, 0.30), fontsize=9.4, color=GREY)
    a1.set_xlabel(r"Forecast error $r_t - q_t$ (%)")
    a1.set_ylabel("Pinball loss")
    a1.set_title("The two arms have different slopes", fontsize=11.7)

    for lev, c in ((ALPHA, al.IDA_RED), (ALPHA_SECONDARY, al.MAIN_BLUE)):
        q0 = float(np.quantile(r, lev))
        grid = np.linspace(q0 - 1.6, q0 + 1.6, 321)
        e = np.array([np.mean(np.where(r < g, (lev - 1) * (r - g), lev * (r - g)))
                      for g in grid])
        a2.plot(grid, e / e.min(), lw=2, color=c,
                label=fr"$\alpha = {lev:.0%}$".replace("%", r"\%"))
        a2.plot([q0], [1.0], "o", color=c, ms=7)
    a2.set_xlabel("Candidate quantile (%)")
    a2.set_ylabel("Mean loss, relative\nto its minimum")
    a2.set_title("Flat around the minimum at 1%", fontsize=11.7)
    fig_legend(fig, a1, ncol=2)

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

    fig, ax = plt.subplots(figsize=FIG_SQ)
    cs = ax.contourf(V, E, L, levels=28, cmap="Blues_r", alpha=0.9)
    ax.contour(V, E, L, levels=14, colors="white", linewidths=0.5, alpha=0.6)
    ax.plot(v0, e0, "o", color=al.IDA_RED, ms=15, mfc="none", mew=2.5,
            label="Sample pair")
    ax.plot(V[i, j], E[i, j], "x", color="black", ms=10, mew=2,
            label="FZ$_0$ minimiser")
    ax.set_xlabel(r"Candidate 1% quantile $v$ (%)")
    ax.set_ylabel("Candidate ES $e$ (%)")
    ax.grid(visible=False)
    cb = fig.colorbar(cs, ax=ax, label="FZ$_0$ loss")
    cb.ax.tick_params(labelsize=9.4)
    fig_legend(fig, ax, ncol=2)
    save(fig, "s_fz")

    return {"fz_var_truth": round(v0, 3), "fz_es_truth": round(e0, 3),
            "fz_var_argmin": round(float(V[i, j]), 3),
            "fz_es_argmin": round(float(E[i, j]), 3)}


def fig_floor():
    """Two Chronos-T5 runs on identical inputs, drawn as the VaR path each one produced.

    The earlier version of this figure counted days into categories, which stated the
    result without showing it. Here the two estimates are simply plotted on top of each
    other: same weights, same prices, same dates, and the only difference is how many
    paths were sampled. Where they part they part by one token of the vocabulary, which
    is why the right panel has a spike at zero and a second spike at the token width
    rather than a smooth spread.
    """
    lo = PRECOMP / f"chronos_t5_{ASSET}.csv"
    hi = PRECOMP / f"chronos_t5_{ASSET}_N2000.csv"
    l = pd.read_csv(lo, parse_dates=["date"])
    h = pd.read_csv(hi, parse_dates=["date"])
    j = (l[np.isclose(l["level"], ALPHA)].set_index("date")["var"]
         .to_frame("n500")
         .join(h[np.isclose(h["level"], ALPHA)].set_index("date")["var"]
               .rename("n2000"), how="inner").dropna())
    d = (j["n2000"] - j["n500"]).abs()
    step = float(d[d > 0.4].median())
    differs = 100 * float((d >= 0.05).mean())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIG_TWO,
                                 gridspec_kw={"width_ratios": [1.7, 1]})

    # --- left: the two estimates, as VaR ------------------------------------------
    w = j.iloc[:30]
    a1.step(w.index, w["n500"], where="mid", lw=2.0, color=al.IDA_RED,
            marker="o", ms=4.5, label="500 paths")
    a1.step(w.index, w["n2000"], where="mid", lw=2.0, color=al.MAIN_BLUE, ls="--",
            marker="s", ms=4.0, label="2000 paths")
    gap = (w["n2000"] - w["n500"]).abs()
    a1.fill_between(w.index, w[["n500", "n2000"]].min(axis=1),
                    w[["n500", "n2000"]].max(axis=1), where=(gap >= 0.05).values,
                    step="mid", color=GREY, alpha=0.35, lw=0)
    a1.set_ylabel(r"$\widehat{\mathrm{VaR}}_{t}(1\%)$, %")
    a1.set_title("Same model, same prices, 30 days", fontsize=10.6)
    a1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # --- right: how big the disagreement is ---------------------------------------
    a2.hist(d.values, bins=np.arange(0, d.max() + 0.12, 0.06), color=RETURNS)
    a2.axvline(step, color=al.IDA_RED, lw=1.6, ls=":",
               label=f"one token = {step:.2f}")
    a2.set_xlabel("Gap between runs (%)")
    a2.set_ylabel("Days")
    a2.set_title(f"Differs on {differs:.0f}% of days", fontsize=10.6)

    fig_legend(fig, [a1, a2], ncol=3)
    save(fig, "s_floor")
    return {"t5_rerun_step_01": round(step, 2),
            "t5_rerun_differs_01": round(differs),
            "t5_rerun_same_01": round(100 - differs),
            "t5_rerun_n_01": int(len(d))}

def main():
    rets, bench, models, bolt = load()
    print(f"{ASSET}: {len(bench)} test days, models {sorted(models)}")

    facts = {}
    facts.update(fig_power(bench))
    # fig_clamp is not called: Chronos-Bolt was dropped from the deck. The generator
    # is kept so the capability audit can be rerun if it returns.
    facts.update(fig_thresholds(bench, models))
    facts.update(fig_allvar(bench, models))
    facts.update(fig_backtest(bench, models))
    facts.update(fig_disagreement(models))
    facts.update(fig_llm(bench, models))
    facts.update(fig_news(bench, models))
    facts.update(fig_open(bench, models))
    facts.update(fig_news_assets())
    facts.update(fig_vs_garch())
    facts.update(fig_data(rets, bench))
    facts.update(fig_coherent())
    facts.update(fig_nnarch())
    facts.update(fig_chronosarch())
    facts.update(fig_llmarch())
    facts.update(fig_hs(rets, bench))
    facts.update(fig_garch(rets, bench))
    facts.update(fig_vares(rets))
    facts.update(fig_qr(rets))
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
