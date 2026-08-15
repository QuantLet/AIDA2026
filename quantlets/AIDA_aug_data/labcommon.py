"""Shared constants and paths for the August course lab build scripts.

The lab is a cut-down, teachable slice of `aida-risk`. It reuses that repository's
data, sign convention and design constants rather than re-deriving them, so a number
a student computes in the notebook can be traced back to the certified pipeline.

SIGN CONVENTION (identical to aida-risk/src/common.py): VaR and ES are stored as
positive losses, and a breach on day t is the event ret_t < -VaR_t(alpha).

TEST SPAN. The lab evaluates the last TEST_N trading days of the sample. TEST_N is
500, not the 40-60 a two-hour lab first suggests, because a backtest is the point of
the exercise and a short span cannot support one: at alpha = 5% a 50-day span expects
2.5 breaches, and Kupiec's test then accepts almost any model, which would teach the
opposite of the intended lesson. 500 days expects 25 breaches at 5% and separates the
models. See DECISIONS.md 2026-07-31.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # august-course/
SCHOOL = ROOT.parent                                # 2026 AIDA Summer School/
RISK = SCHOOL / "aida-risk"                         # the certified pipeline
LAB = ROOT / "data" / "lab"
LLM = ROOT / "data" / "llm"
PRECOMP = ROOT / "precomputed"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
for _d in (LAB, LLM, PRECOMP, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --- lab design constants (echoed into every table note and README) ----------
TEST_N = 500            # trading days evaluated, ending at the sample end
ALPHA = 0.01            # headline level: the level Basel counts and the tail that hurts
ALPHA_SECONDARY = 0.05  # reported alongside, because the LLM returns both in one reply
ALPHAS = (0.01, 0.05, 0.10)
HS_WINDOW = 250         # rolling window for the lab's own historical simulation
CONTEXT = 512           # prices fed to a foundation model, matches aida-risk 02c
NUM_SAMPLES = 500       # Chronos-T5 sample paths; drives the 5% quantile noise floor

# Assets available to the lab, one per group. Keys match aida-risk.
ASSETS = {
    "SPX":  {"ticker": "^GSPC",   "label": "S&P 500"},
    "DAX":  {"ticker": "^GDAXI",  "label": "DAX 40"},
    "N225": {"ticker": "^N225",   "label": "Nikkei 225"},
    "GOLD": {"ticker": "GC=F",    "label": "Gold futures"},
    "BTC":  {"ticker": "BTC-USD", "label": "Bitcoin"},
}
DEFAULT_ASSET = "SPX"

# Colours: the IDA convention from aida-risk/src/common.py, extended for the two
# lab-only model families so a chart never reuses a hue across models.
MAIN_BLUE = "#003DA5"
IDA_RED = "#C8102E"
FOREST = "#228B22"
GREY = "#6E6E6E"
MODEL_COLORS = {
    "HS": GREY,
    "GARCH-t": MAIN_BLUE,
    "NN-t": "#16A085",
    "Chronos-T5": "#5B2C6F",
    "LLM-series": "#E67E22",
    "LLM-series+state": "#B9770E",
    "LLM-dated": "#7D6608",
    "LLM-dated+news": IDA_RED,
    "LLM-ensemble": "#943126",
    # Open weights, matching notebook/aidalab.py so a bar means the same colour in the
    # deck and in the laboratory. The size sweep darkens with parameter count.
    "Open-1.5B": "#1B7837",
    "Open-3B": "#A6761D",
    "Open-7B": "#8C6D1F",
    "Open-14B": "#5E4A12",
}
MODEL_LABELS = {
    "HS": "Historical simulation",
    "GARCH-t": "GARCH(1,1)-$t$",
    "NN-t": "Neural volatility + $t$ tail",
    "Chronos-T5": "Chronos-T5 (zero-shot, sampled)",
    "LLM-series": "LLM, serialised history only",
    "LLM-series+state": "LLM, history + market state",
    "LLM-dated": "LLM, dated (control)",
    "LLM-dated+news": "LLM, dated + real headlines",
    "LLM-ensemble": "LLM ensemble (mean of configurations)",
    "Open-1.5B": "Qwen2.5-1.5B (open weights)",
    "Open-3B": "Qwen2.5-3B (open weights)",
    "Open-7B": "Qwen2.5-7B (open weights)",
    "Open-14B": "Qwen2.5-14B (open weights)",
}


def risk_proc(asset):
    """Processed-data directory of the certified aida-risk run for an asset."""
    return RISK / "data" / "processed" if asset == DEFAULT_ASSET else \
        RISK / "data" / "processed" / asset


def legend_below(ax, ncol=3, y=-0.22, fontsize=11):
    """Place the legend outside the axes, centred below them.

    IDA convention, and not only cosmetic: the interesting part of a VaR chart is the
    lower-left tail, which is exactly where matplotlib prefers to drop a legend box.
    """
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     fontsize=fontsize, frameon=False)
