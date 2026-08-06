"""Generate the student notebook.

The notebook is not hand-edited. Cells are defined here as (kind, source) pairs and
written out as .ipynb, so a change to the lab is a change to one Python file and the
notebook can be regenerated without merge noise from execution counts and outputs.

    python notebook/build_notebook.py

Writes notebook/AIDA_Risk_Lab.ipynb.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MD, CODE = "markdown", "code"
CELLS = []


BADGE = {
    "required": "**`REQUIRED`**",
    "interpret": "**`INTERPRET`** --- write two or three sentences, then move on",
    "optional": "**`OPTIONAL`** --- only if your group is ahead",
    "slow": "**`SLOW / PRECOMPUTED FALLBACK AVAILABLE`** --- the shipped file is "
            "loaded automatically if the live run is unavailable",
}


def md(text, tag=None, mins=None):
    """A markdown cell. `tag` prints one of the four lab badges above it, and `mins`
    the wall-clock estimate for the section on a free Colab CPU runtime."""
    body = text.strip("\n")
    line = []
    if tag:
        line.append(BADGE[tag])
    if mins:
        line.append(f"*runs in about **{mins}** on a Colab CPU runtime*")
    if line:
        body = "  ·  ".join(line) + "\n\n" + body
    CELLS.append((MD, body))


def code(text):
    CELLS.append((CODE, text.strip("\n")))


# ===========================================================================
md(r"""
# Humans, foundation models and LLMs: who forecasts tail risk better?

**AIDA Summer School 2026 — coding laboratory (2 hours)**

One question, four kinds of answer.

$$\Pr\left(r_{t} < -\widehat{\mathrm{VaR}}_{t}(\alpha)\right) = \alpha ,
\qquad \alpha = 1\%$$

with everything on the right-hand side computed from information available at $t-1$.

Every model in this notebook sees the same information and is scored on the same
500 days. The models are:

| | how it produces tomorrow's number |
|---|---|
| **Historical simulation** | sort the last 250 days, read off the 3rd worst |
| **GARCH(1,1)-$t$** | a variance recursion, scaled by a Student-$t$ quantile |
| **NN vol + $t$** | a small network learns the scale, the tail shape stays parametric |
| **Chronos-Bolt** | a pretrained time-series model with a quantile head, zero-shot |
| **Chronos-T5** | the same family, but it samples paths, so any quantile can be read |
| **LLM** | the return series written out as text, the quantile asked for in words |

You will find out which of them is *calibrated*, which is a different question from
which of them is sophisticated.

**Sign convention.** VaR is reported as a positive loss. Day $t$ is a breach when
$r_t < -\mathrm{VaR}_t$. Charts are drawn on the return axis with the tail on the left.
""")
COLAB = ("https://colab.research.google.com/github/QuantLet/AIDA2026/blob/main/"
         "notebook/AIDA_Risk_Lab.ipynb")

md(f"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

*Click the badge to open this notebook in Google Colab. Nothing needs to be installed
locally, and no API key is required at any point.*
""")

md(r"""
### How long the laboratory takes

Sections 1 to 3 were **measured on Colab** on 2026-08-06. The rest are measured on one
CPU (Apple M5 Max) and multiplied by four, which is roughly how much slower a free
two-vCPU Colab runtime is on this kind of work. Colab caches nothing between sessions,
so budget the downloads every time.

| section | compute | of which download |
|---|---|---|
| 1. Setup and reproducibility checks | ~10 s | ~20 MB, the repository |
| 1.1 Power | seconds | — |
| 2. Historical simulation | seconds | — |
| 3. Chronos-Bolt capability audit | ~1.5 min | ~35 MB, `chronos-bolt-tiny` |
| 4. Chronos-T5 sampling resolution, 40 dates x 500 paths | ~3 min | ~80 MB, `chronos-t5-mini` |
| 5. LLM parsing and logical checks | seconds | shipped as data |
| 6. Dated versus dated+news | seconds | shipped as data |
| 6.1 Open weights live, 16 dates | ~8 min on CPU, ~1 min on a T4 | **3.1 GB**, `Qwen2.5-1.5B` |
| 7. Coverage, loss and disagreement | seconds | — |

**Compute is about 15 minutes of the two hours.** Everything else is reading, writing and
arguing about what the numbers mean, which is the point of the session. Two consequences
worth planning around:

- **Start section 6.1 early** if you want the live open-weights run, or set
  `RUN_OPEN_LIVE = False` and use the shipped file. The 3.1 GB download is the single
  longest wait in the laboratory.
- Everything marked `SLOW` has a precomputed fallback that loads in under a second, so a
  failed download never blocks the rest of the notebook.
""")

md(r"""
## 1. Setup and reproducibility checks

Run this cell first. Inside the course repository it uses the files next to it; on
Colab it clones the public repository, which takes a few seconds and needs nothing from
you. If GitHub is unreachable it falls back to asking for `aida_lab_data.zip`.

**You need no API key at any point.** The language-model forecasts and the news
headlines are shipped as data. The only thing fetched at run time is the Chronos model
itself, from Hugging Face, which is a public download.
""", "required", mins="10 seconds")

code(r"""
import os, sys, pathlib

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB and not pathlib.Path("aidalab.py").exists():
    # The course repository is public, so Colab fetches everything itself: the helper
    # module, the return series, the aligned headlines and every precomputed forecast.
    # About 20 MB, a few seconds, and no API key of any kind.
    !git clone --depth 1 -q https://github.com/QuantLet/AIDA2026.git
    if pathlib.Path("AIDA2026/notebook/aidalab.py").exists():
        os.chdir("AIDA2026/notebook")
        sys.path.insert(0, os.getcwd())
        print("cloned the course repository")
    else:
        # Fallback for a room with no access to GitHub: upload aida_lab_data.zip.
        from google.colab import files
        files.upload()
        import zipfile
        zipfile.ZipFile("aida_lab_data.zip").extractall(".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import aidalab as al
al.set_seeds(42)

ASSET = "SPX"      # your group's asset: SPX, DAX, N225, GOLD, BTC
ALPHA = 0.01       # the level everything is scored at: 1%, the Basel tail

returns = al.load_returns(ASSET)
bench   = al.load_benchmarks(ASSET, level=ALPHA)

print(f"{ASSET}: {len(returns)} daily observations, "
      f"{returns.index[0].date()} to {returns.index[-1].date()}")
print(f"test span: {bench.index[0].date()} to {bench.index[-1].date()}, {len(bench)} days")
bench.head()
""")

md(r"""
### 1.1 Power: how much can 500 days at 1% actually tell you?

A backtest is a hypothesis test, and a hypothesis test needs events. At
$\alpha = 1\%$ a 500-day span expects **five** breaches. Five. The cell below computes
the smallest true breach rate Kupiec's test can reject, as a function of sample length,
and the answer for our span is sobering: it detects a *doubling* of the breach rate and
nothing finer.

This is the binding constraint of the whole laboratory and it is not hidden anywhere in
the results. Read every verdict in section 7 through it: a model that "passes" at 1% on
500 days has cleared a low bar, and the classical models are also shown on the certified
5541-day run at the end of section 7, where the bar is much higher.
""", "required", mins="seconds")

code(r"""
from scipy import stats

def detectable(n, alpha=ALPHA, level=0.05):
    # Smallest true breach rate the Kupiec test rejects at `level`, given n days.
    crit = stats.chi2.ppf(1 - level, 1)
    for p in np.arange(alpha, 0.5, 0.001):
        x = p * n
        ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
        ll1 = (n - x) * np.log(1 - p) + x * np.log(p)
        if -2 * (ll0 - ll1) > crit:
            return p
    return np.nan

print(f"alpha = {ALPHA:.0%}\n")
for n in (250, 500, 1000, 2000, 5541):
    d = detectable(n)
    print(f"  {n:>5} days: expect {ALPHA * n:5.1f} breaches, "
          f"rejects only above {d:.2%}  ({d / ALPHA:.1f}x nominal)")
print(f"\n  our span is 500 days -> {detectable(500) / ALPHA:.1f}x nominal is the "
      f"finest deviation the test can see")
""")

md(r"""
## 2. The historical-simulation benchmark

No model, no parameters, no distribution. Sort the window and read off the quantile.

$$\widehat{\mathrm{VaR}}^{\mathrm{HS}}_{t}(\alpha) = -\,\widehat{Q}_{\alpha}
\!\left(r_{t-W},\dots,r_{t-1}\right), \qquad W = 250$$

**Write it yourself.** One line. Two things to be careful about.

- The **window ends at $t-1$**. Slicing up to and including $t$ puts the day you are
  forecasting inside its own estimation window, which is the most common way this
  exercise silently cheats.
- At $\alpha = 1\%$ a 250-day window has **two and a half** observations below the
  quantile. The estimator is reading two or three days. Everything historical simulation
  does at this level rests on them.
""", "required", mins="seconds")

code(r"""
# EXERCISE. Complete the function, then run the cell below it.
def my_hs_var(window_returns, alpha=ALPHA):
    # Empirical alpha-quantile of a return window, returned as a positive loss.
    return ...        # <-- your code here


# check against the reference implementation on one window
w = returns["ret"].iloc[-260:-10]
print("yours:    ", my_hs_var(w))
print("reference:", al.hs_var(w))
""")

code(r"""
# Roll it over the test span. `rolling_hs` drops day t from its own window.
hs250 = al.rolling_hs(returns["ret"], bench.index, window=250, alpha=ALPHA)

fc = {
    "HS":      bench["HS"],        # certified run, 1000-day window
    "GARCH-t": bench["GARCH-t"],
    "NN-t":    bench["NN-t"],
}
fc["HS-250"] = hs250

fig, ax = plt.subplots(figsize=(10, 4.2))
al.plot_tail(bench["ret"], {k: fc[k] for k in ("HS", "HS-250", "GARCH-t")},
             title=f"{ASSET}: realised returns and three VaR thresholds",
             ax=ax, show_breaches="GARCH-t")
plt.show()
""")

md(r"""
**Question 1.** The two historical-simulation lines use the same estimator and differ
only in window length (1000 days against 250). Look at where they sit relative to each
other and to the returns. Which one reacts, which one does not, and why does the longer
window sit so far out in 2023?
""")

md(r"""
## 3. Chronos-Bolt: a capability audit

Chronos is pretrained on a large corpus of time series and applied here with **no
fitting on this asset at all**. Two heads, and the difference between them is the
lesson of this section.

- **Chronos-Bolt** has a *quantile head*: it outputs a fixed grid of quantiles directly.
  Small and fast enough to run live.
- **Chronos-T5** *samples paths*: any quantile can be read off the sample distribution.

**It is fed prices, not returns.** Chronos is trained on series levels. Given the last
512 closing prices it forecasts the next one, and the implied return is recovered as
$r = 100\log(\hat P_t / P_{t-1})$, a monotone transform, so a price quantile maps to
the return quantile of the same level. Feeding it the return series directly was tried
in the certified pipeline: returns are near-zero-mean noise and the model collapses to
a predictive standard deviation of 0.14 against a realised 1.22.
""", "required", mins="1.5 minutes")

code(r"""
!pip install -q chronos-forecasting

# Hugging Face asks Colab for an `HF_TOKEN` secret and, when the notebook has no access
# to it, Colab raises a permission dialog. Nothing here needs a token: every model is a
# public download. This marks the lookup as already done, so the dialog never appears.
# If you ever see it, in this notebook or another, the correct answer is Cancel.
try:
    from huggingface_hub.utils import _auth
    _auth._IS_GOOGLE_COLAB_CHECKED = True
    _auth._GOOGLE_COLAB_SECRET = None
except Exception:
    pass
""")

code(r"""
import torch, warnings
from chronos import BaseChronosPipeline

px    = returns["close"].values.astype(float)
pos   = {d: i for i, d in enumerate(returns.index)}
idx   = np.array([pos[d] for d in bench.index])
CTX   = 512

bolt = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-tiny", device_map="cpu", dtype=torch.float32)

# Ask for the levels we actually care about.
LEVELS = [0.01, 0.05, 0.10, 0.50, 0.90]
ctx = torch.tensor(np.stack([px[t - CTX:t] for t in idx[:64]]), dtype=torch.float32)
with torch.no_grad():
    q, _ = bolt.predict_quantiles(ctx, prediction_length=1, quantile_levels=LEVELS)

print("price quantiles for the first forecast date:")
for lvl, v in zip(LEVELS, q[0, 0].numpy()):
    print(f"  q{lvl:<5} = {v:.4f}")
""")

md(r"""
**Look at the warning, and look at the numbers.**

Chronos-Bolt was trained on the quantile grid $0.1, 0.2, \dots, 0.9$. Asked for 0.01 it
does not extrapolate: it returns the value at the lowest level it was trained on. The
1%, 5% and 10% entries above are the same number.

At our reported level this is a **factor-of-ten** substitution. A "1% VaR" that is
silently the 10% quantile will be breached roughly ten times as often as it claims, and
the pipeline that produced it raised no error at any point.

Verify it rather than taking it on trust.
""")

code(r"""
qq = q[:, 0, :].numpy()
i01, i05, i10 = LEVELS.index(0.01), LEVELS.index(0.05), LEVELS.index(0.10)
same = np.isclose(qq[:, i01], qq[:, i10], rtol=0, atol=0) & \
       np.isclose(qq[:, i05], qq[:, i10], rtol=0, atol=0)
print(f"dates where q01 == q05 == q10 exactly: {same.sum()} of {len(same)}")
""")

md(r"""
**This is the single most important cell in the notebook.**

A pipeline that requested `quantile_levels=[0.01]`, took the number, and called it a 1%
VaR would have produced a full set of results, a leaderboard position and a plot — all
of them silently reporting the 10% quantile, a **factor of ten** away. Nothing would have failed. The backtest
would have shown a suspiciously high breach rate and the natural reading would have been
"the foundation model is badly calibrated in the tail", which is a claim about the model
rather than about the pipeline.

Two things follow, and they are the transferable part of this laboratory:

1. **Ask a model what it can answer, not just what you want.** The trained quantile grid
   is a property of the artefact, as much as its parameter count.
2. **Check the output, not the call.** The call succeeded. Only the values reveal it.

We therefore run Chronos-Bolt at its lowest unclamped level, 10%, and get the 1%
quantile from the sampling head instead.
""")

code(r"""
# Chronos-Bolt at 10%, the lowest level it was trained on, over the full test span.
rows = []
for i in range(0, len(idx), 64):
    chunk = idx[i:i + 64]
    ctx = torch.tensor(np.stack([px[t - CTX:t] for t in chunk]), dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            qh, _ = bolt.predict_quantiles(ctx, prediction_length=1, quantile_levels=[0.10])
    for j, t in enumerate(chunk):
        r_q = 100.0 * np.log(max(float(qh[j, 0, 0]), 1e-8) / px[t - 1])
        rows.append((bench.index[i + j], -r_q))

bolt10 = pd.Series(dict(rows), name="Chronos-Bolt").rename_axis("date")
print(f"Chronos-Bolt 10% VaR: mean {bolt10.mean():.2f}, {len(bolt10)} dates")
""")

md(r"""
### 4. Chronos-T5: the sampling-resolution experiment

The sampling head draws `num_samples` price paths and the quantile is read off the
sample distribution. The sample count sets the noise floor, and at our level it is the
binding constraint: with 500 draws, **five** fall below the 1% quantile. The library
default of 20 draws cannot express a 1% quantile at all — there is no draw below it.

So the Chronos-T5 number is an order statistic of five points. That is a modelling
choice with the same standing as the window length in historical simulation, and it is
reported about as often as never.

It costs about 3 seconds per forecast on a CPU, so the full 500-day run is precomputed
and shipped. Reproduce a slice of it live, then check your slice against the file.
""", "slow", mins="3 minutes")

code(r"""
t5_pre = al.load_precomputed(f"chronos_t5_{ASSET}.csv")
if t5_pre is not None:
    t5 = t5_pre[np.isclose(t5_pre["level"], ALPHA)].set_index("date")["var"].rename("Chronos-T5")
    print(f"precomputed Chronos-T5: {len(t5)} dates, mean VaR {t5.mean():.2f}")
else:
    t5 = None
    print("no precomputed file found; run the live cell below with a larger N_LIVE")
""")

code(r"""
N_LIVE, N_SAMPLES = 40, 500

# One date at a time. The sampling head decodes BATCH * N_SAMPLES sequences in
# parallel, so a batch of four at 500 draws is 2000 of them at once: that fits on a
# workstation and exhausts the RAM of a free Colab runtime, which then restarts and
# loses every variable above. Measured, not guessed.
BATCH = 1

t5m = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-mini", device_map="cpu", dtype=torch.float32)

torch.manual_seed(42)
live = {}
for k in range(0, N_LIVE, BATCH):
    chunk = idx[k:k + BATCH]
    ctx = torch.tensor(np.stack([px[t - CTX:t] for t in chunk]), dtype=torch.float32)
    with torch.no_grad():
        samp = t5m.predict(ctx, 1, num_samples=N_SAMPLES)[:, :, 0].numpy()
    for j, t in enumerate(chunk):
        rs = 100.0 * np.log(np.maximum(samp[j], 1e-8) / px[t - 1])
        live[bench.index[k + j]] = -float(np.quantile(rs, ALPHA))
    del samp

live = pd.Series(live).rename_axis("date")
if t5 is not None:
    both = pd.concat([live.rename("live"), t5.rename("precomputed")], axis=1).dropna()
    print(both.head())
    print(f"\ncorrelation over {len(both)} shared dates: "
          f"{both['live'].corr(both['precomputed']):.3f}")
""")

md(r"""
**Question 2.** Your live run and the precomputed file use the same model, the same
context and the same seed, and the numbers still differ slightly. Where does the
difference come from, and what does it imply about reporting a single VaR number from
a sampling model to three decimal places?
""")

md(r"""
## 5. The LLM: output parsing and logical checks

The method is the one in Pele et al. (2026), *In the Beginning was the Word: LLM-VaR
and LLM-ES* (Expert Systems with Applications). The return series is written out as
text, the model is asked for the 1% and 5% quantiles of tomorrow's return, and the
number is
read out of the reply. There is no VaR head and no fine-tuning; the forecast comes from
prompted generation.

Four information sets are shipped, for every asset:

| configuration | what the prompt contains |
|---|---|
| `series` | the last 60 returns, anonymised |
| `series+state` | the same, plus realised volatility, drawdown and the HS VaR |
| `dated` | the same returns, plus the asset name and the date |
| `dated+news` | the same, plus the real headlines available at $t-1$ (section 6) |

**Why two of them are anonymised.** The model's training data covers these dates. Tell
it "S&P 500, 5 August 2024" and a good forecast may be recall rather than inference,
and recall is not available for tomorrow. The `dated` configuration exists to *measure*
that effect — and to serve as the control for the news test in section 6.

**The headlines are real.** They come from EODHD, not from anyone's imagination: a
headline file written to match the returns would manufacture the relation it claims to
test. How they are aligned to the forecast dates is section 6, and it is the part
worth arguing with.

This section works with the first two. The last two are section 6's business.
""", "required", mins="seconds")

code(r"""
MODEL = "claude-haiku-4-5"
llm = {}
for cfg, key in [("series", "LLM-series"), ("series_state", "LLM-series+state")]:
    d = al.load_precomputed(f"llm_{cfg}_{ASSET}_{MODEL}.csv")
    if d is None:
        print(f"  {key}: not generated for {ASSET}")
        continue
    d = d.set_index("date")
    ok = d["raw_ok"] & d["sign_ok"]
    llm[key] = d.loc[ok, "var"]
    print(f"  {key}: {len(d)} replies, {int((~d['raw_ok']).sum())} unparseable, "
          f"{int((d['raw_ok'] & ~d['sign_ok']).sum())} with a non-negative quantile, "
          f"mean VaR {llm[key].mean():.2f}")
""")

md(r"""
Two failure counts are printed above and both are results.

An **unparseable** reply is an engineering failure: the model wrote prose where JSON was
asked for. A reply with a **non-negative quantile** is worse, because it parses. It
asserts that the worst 5% of days is a gain, and it would flow into every downstream
number without tripping anything. Neither is repaired here; both are counted.
""")

code(r"""
if len(llm) == 2:
    a, b = llm["LLM-series"], llm["LLM-series+state"]
    j = pd.concat([a.rename("series"), b.rename("series+state"),
                   bench["HS"].rename("HS"), bench["GARCH-t"].rename("GARCH")],
                  axis=1).dropna()
    print("correlations between the LLM forecasts and the statistical ones\n")
    print(j.corr().round(3).to_string())
""")

md(r"""
### 6. Dated versus dated+news: does real text move the number?

The two configurations above contain no words. This one does: real headlines from EODHD,
attached to each forecast date under one rule.

> A headline may enter the prompt for date $t$ only if it was published at or before the
> **as-of cutoff** on $t-1$ — the previous decision point, 20:00 UTC.

`src/06_news.py` enforces that rule, stores each headline's publication timestamp
alongside it, and audits both. This matters more than the price alignment: a price series
can only leak by being shifted, whereas a news series leaks the moment an article
published at 22:00 describing the day's close is handed to a model asked to forecast that
close. Nothing in the reply would look wrong.

**The comparison is against `dated`, not against `series`.** Headlines reveal the period
on their own — they name companies, events and numbers that place the text in time. So
pairing them with the anonymised configuration would confound two effects:

| | period revealed | headlines |
|---|---|---|
| `series` | no | no |
| `dated` | yes | no | ← the control |
| `dated+news` | yes | yes | ← the treatment |

The difference between the last two identifies the contribution of the text alone.
""", "required", mins="seconds")

code(r"""
news = al.load_precomputed(f"../data/news/headlines_{ASSET}.csv")
if news is None:
    import pathlib
    p = pathlib.Path("../data/news") / f"headlines_{ASSET}.csv"
    news = pd.read_csv(p, parse_dates=["date", "asof_utc", "published_utc"]) if p.exists() else None

for cfg, key in [("dated", "LLM-dated"), ("dated_news", "LLM-dated+news")]:
    d = al.load_precomputed(f"llm_{cfg}_{ASSET}_{MODEL}.csv")
    if d is None:
        print(f"  {key}: not generated for {ASSET}")
        continue
    d = d.set_index("date")
    llm[key] = d.loc[d["raw_ok"] & d["sign_ok"], "var"]
    print(f"  {key}: mean VaR {llm[key].mean():.2f}, {len(llm[key])} usable days")

if news is not None:
    cov = news.groupby("date").size().reindex(bench.index, fill_value=0)
    covered = bench.index[cov > 0]
    print(f"\nheadlines: {len(news)} over {len(covered)} of {len(bench)} dates "
          f"({100*len(covered)/len(bench):.0f}%), "
          f"{cov[cov>0].mean():.1f} per covered date")
""")

md(r"""
On a date with **no** headline in its window the two prompts are identical by
construction, so those days contribute exactly zero to the loss differential and only
dilute the test. Run it on the covered subset, and on the full sample for comparison.
""")

code(r"""
if {"LLM-dated", "LLM-dated+news"} <= set(llm) and news is not None:
    a, b = llm["LLM-dated+news"], llm["LLM-dated"]

    t1, p1, d1, n1, _ = al.dm_test(bench["ret"], a, b, ALPHA)
    print(f"full sample      n={n1:>3}  mean loss diff {d1:+.5f}  DM p {p1:.3f}")

    t2, p2, d2, n2, _ = al.dm_test(bench.loc[covered, "ret"],
                                   a.reindex(covered).dropna(),
                                   b.reindex(covered).dropna(), ALPHA)
    print(f"covered dates    n={n2:>3}  mean loss diff {d2:+.5f}  DM p {p2:.3f}")
    print(f"\n(a negative difference means the headlines helped)")

    print("\ncoverage-weighted check: does the effect grow with headline count?")
    m = pd.concat([a.rename('news'), b.rename('nonews'), cov.rename('k')],
                  axis=1, sort=False).dropna()
    for lo, hi in [(1, 2), (3, 4), (5, 99)]:
        s = m[(m['k'] >= lo) & (m['k'] <= hi)]
        if len(s) > 20:
            print(f"  {lo}-{min(hi,5)} headlines, {len(s):>3} days: "
                  f"mean |news - nonews| = {(s['news'] - s['nonews']).abs().mean():.3f}pp")
""")

md(r"""
Your asset is one of five. The shipped bundle carries the same four configurations and
the same aligned headlines for all of them, so the replication is one loop away — and a
single-asset null is worth very little compared with five.
""")

code(r"""
rows = []
for a in ["SPX", "DAX", "N225", "GOLD", "BTC"]:
    try:
        ba = al.load_benchmarks(a, ALPHA)
        mm = {}
        for cfg, k in [("dated", "d"), ("dated_news", "dn")]:
            dd = al.load_precomputed(f"llm_{cfg}_{a}_{MODEL}.csv").set_index("date")
            mm[k] = dd.loc[dd["raw_ok"] & dd["sign_ok"], "var"]
        hh = al.load_precomputed(f"../data/news/headlines_{a}.csv")
        cv = hh.groupby("date").size().reindex(ba.index, fill_value=0)
        cd = ba.index[cv > 0]
        _, p, diff, n, _ = al.dm_test(ba.loc[cd, "ret"], mm["dn"].reindex(cd).dropna(),
                                      mm["d"].reindex(cd).dropna(), ALPHA)
        rows.append({"asset": a, "covered": int((cv > 0).sum()), "n": n,
                     "loss_diff": round(diff, 5), "DM_p": round(p, 3)})
    except Exception as e:
        rows.append({"asset": a, "covered": None, "n": None,
                     "loss_diff": None, "DM_p": None})
print(pd.DataFrame(rows).to_string(index=False))
print("\nnegative loss_diff = headlines helped")
""")

md(r"""
**Question 3a.** Whatever your numbers say, write the sentence you would put in a report.
Then write the sentence you would have written if you had compared `dated+news` against
`series` instead, and say which of the two claims the data supports.
""")

md(r"""
**Question 3.** Read the correlation between `series+state` and `HS`. The state block in
that prompt *contains* the historical-simulation VaR. If the correlation is very high,
what has the LLM contributed beyond copying a number it was handed? Design the prompt
you would use to separate the two.
""")

md(r"""
### 6.1 The same experiment, with a model you run yourself

Everything above came from a commercial endpoint. If you have a key, section 5 is
reproducible; if you do not, it is a table someone else computed. So the same prompt is
also put through a model with **open weights**, downloaded from Hugging Face exactly the
way Chronos was, running on this machine.

- `Qwen2.5-1.5B-Instruct` — about 3 GB, roughly a second per forecast on a Colab GPU
  and a few seconds on CPU.
- `Qwen2.5-3B-Instruct` — twice the size, and it is in the package for a reason you will
  see in a moment.

The prompt, the JSON schema and the parser are **imported** from the commercial stage
rather than copied, so the two legs differ in exactly one thing: which model reads the
prompt.
""", "slow", mins="8 minutes on CPU, 1 on a GPU")

code(r"""
open_llm = {}
for slug, key in [("Qwen2.5-1.5B-Instruct", "Open-1.5B"),
                  ("Qwen2.5-3B-Instruct", "Open-3B")]:
    d = al.load_precomputed(f"local_series_{ASSET}_{slug}.csv")
    if d is None:
        print(f"  {key}: not generated for {ASSET}")
        continue
    d = d.set_index("date")
    ok = d["raw_ok"] & d["sign_ok"] & d.get("order_ok", True)
    open_llm[key] = d.loc[ok, "var"]
    print(f"  {key}: {int(d['raw_ok'].sum())}/{len(d)} parsed, "
          f"{int((~d['sign_ok'] & d['raw_ok']).sum())} wrong sign, "
          f"{int((~d.get('order_ok', True) & d['raw_ok']).sum())} inverted, "
          f"mean VaR {open_llm[key].mean():.2f}")
""")

md(r"""
Every validity check the pipeline has can pass while the forecast is worthless. Count
how many **distinct** numbers each model actually emits.
""")

code(r"""
cand = {**{k: llm[k] for k in ["LLM-series"] if k in llm}, **open_llm}
print(f"{'model':<24} {'usable':>7} {'distinct':>9} {'%':>6} {'sd':>7} {'corr GARCH':>11}")
for k, v in cand.items():
    v = v.dropna()
    d = v.round(2).nunique()
    c = pd.concat([v, bench["GARCH-t"]], axis=1, sort=False).dropna().corr().iloc[0, 1]
    print(f"{k:<24} {len(v):>7} {d:>9} {100*d/len(v):>5.0f}% {v.std():>7.3f} {c:>11.3f}")
""")

md(r"""
**Question 3b.** One of these models produces well-formed JSON, a correctly signed
quantile and a correctly ordered pair on essentially every day — and is still useless.
Which one, how did you tell, and which of the checks in the parser would have caught it?

Now run it yourself. The cell below re-derives a slice of the open model's forecasts
live; it is the only place in this notebook where **you** run a language model.
""")

code(r"""
# ~1 min on a Colab GPU, a few minutes on CPU. Raise N_OPEN if you have time.
N_OPEN = 16
RUN_OPEN_LIVE = True

if RUN_OPEN_LIVE:
    !pip install -q transformers accelerate
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MID = "Qwen/Qwen2.5-1.5B-Instruct"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MID, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(
        MID, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    SYS = ("You are a market risk analyst. From a series of recent daily returns you "
           "estimate the risk of the next trading day. You answer with a single JSON "
           "object and nothing else: no explanation before it, no code fence around it.")
    SCHEMA = (
        "Return exactly this JSON object:" + chr(10)
        + "{" + chr(10)
        + '  "q01": <number>,   // 1% quantile of the next return, percent, negative'
        + chr(10)
        + '  "q05": <number>    // 5% quantile, negative and above q01' + chr(10)
        + "}"
    )

    import json, re
    live_open = {}
    for d in bench.index[:N_OPEN]:
        hist = returns["ret"].loc[:d].iloc[:-1].tail(60).values
        user = ("Daily returns in percent, oldest first, 60 observations:"
                + chr(10) + ", ".join(f"{x:.2f}" for x in hist)
                + chr(10) + chr(10) + SCHEMA)
        chat = tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False)
        enc = tok(chat, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = mod.generate(**enc, max_new_tokens=120, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            try:
                live_open[d] = -float(json.loads(m.group(0))["q01"])
            except Exception:
                pass

    live_open = pd.Series(live_open).rename_axis("date")
    print(chr(10) + f"{len(live_open)}/{N_OPEN} usable forecasts, "
          f"{live_open.round(2).nunique()} distinct values")
    if "Open-1.5B" in open_llm:
        j = pd.concat([live_open.rename("live"),
                       open_llm["Open-1.5B"].rename("shipped")], axis=1).dropna()
        print(j.head(8).round(3).to_string())
""")

md(r"""
**Question 3c.** Your live run used greedy decoding, the same model and the same prompt
as the shipped file. Do the numbers match exactly? If they do not, name every place a
difference could have entered — and say which of them would also affect a commercial
endpoint you cannot inspect.
""")

md(r"""
## 7. Coverage, pinball loss and disagreement

Three quantities, and they answer different questions.

**Breach rate.** Does the threshold get crossed as often as it should?

$$\hat\alpha = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}\!\left\{r_t < -\widehat{\mathrm{VaR}}_t\right\}$$

**Kupiec's unconditional-coverage test.** Is $\hat\alpha$ far enough from $\alpha$ to
reject the model? $LR_{UC}\sim\chi^2_1$ under the null.

**Christoffersen's independence test.** Do breaches *cluster*? A model can have the right
number of breaches and still fail, if they all arrive in the same fortnight.

**Pinball loss.** The strictly consistent scoring rule for a quantile, and therefore the
right thing to rank on. With $q_t = -\widehat{\mathrm{VaR}}_t$,

$$L_\alpha = \frac{1}{T}\sum_{t=1}^{T}\left(\alpha - \mathbf{1}\{r_t < q_t\}\right)(r_t - q_t)$$

The breach count is not a scoring rule: a model can hit 5% exactly with thresholds that
are wildly wrong day by day.
""", "required", mins="seconds")

code(r"""
models = {
    "HS":               bench["HS"],
    "HS-250":           hs250,
    "GARCH-t":          bench["GARCH-t"],
    "NN-t":             bench["NN-t"],
    "Chronos-Bolt":     bolt10,      # NOTE: this is a 10% forecast, see below
    **({"Chronos-T5":   t5} if t5 is not None else {}),
    **llm,
}

bt = al.backtest_table(bench["ret"], models, alpha=ALPHA)
lb = al.leaderboard(bt, alpha=ALPHA)
lb.round(4)
""")

md(r"""
**Chronos-Bolt is scored against the wrong nominal level on purpose.** Its number is a
10% quantile; the table judges every column at 1%. It should look badly calibrated, and
it is not a defect of the model: it is what happens when a forecast is used at a level
the artefact cannot produce. Score it at its own level to see the difference.
""")

code(r"""
bt10 = al.backtest_table(bench["ret"], {"Chronos-Bolt": bolt10}, alpha=0.10)
print("Chronos-Bolt judged at its own 10% level:")
print(bt10.round(4).to_string(index=False))
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
al.plot_breaches(bt, alpha=ALPHA, ax=axes[0])
al.plot_leaderboard(lb, ax=axes[1])
plt.tight_layout(); plt.show()
""")

md(r"""
### 7.1 Ranking against calibration: is the gap real?

A leaderboard is an ordering. Whether the gap between two rows is larger than sampling
noise is a separate question, and the Diebold–Mariano test on the loss differential is
how it is answered. Daily loss differentials are autocorrelated, because volatility
clusters, so the standard error is HAC-corrected: without that, ordinary noise reads as
a significant win.

$$DM = \frac{\bar d}{\widehat{\mathrm{se}}_{HAC}(\bar d)},
\qquad d_t = L_\alpha^{(A)}(t) - L_\alpha^{(B)}(t)$$
""")

code(r"""
top = lb.iloc[0]["model"]
print(f"top of the leaderboard: {top}\n")
for m in lb["model"][1:]:
    t, p, d, n, lag = al.dm_test(bench["ret"], models[top], models[m], ALPHA)
    verdict = "distinguishable" if p < 0.05 else "NOT distinguishable"
    print(f"  vs {m:<18} mean loss diff {d:+.5f}  DM t {t:+5.2f}  "
          f"p {p:.3f}  -> {verdict}")
print("\npairwise p-values:")
print(al.dm_matrix(bench["ret"], models, ALPHA).round(3).to_string())
""")

md(r"""
**Question 4a.** How many of the pairwise gaps survive the test? If the top of your
leaderboard is not distinguishable from the second row, what is the correct sentence to
put in a report — and what is the incorrect one that a leaderboard invites?
""")

md(r"""
**Question 4.** Find a model that *wins* on pinball loss and is *rejected* on coverage,
or the reverse. Both happen. Which of the two would you report to a risk committee, and
what would you say about the other?
""")

md(r"""
### 7.2 How much do the models actually disagree?

Different names, different vendors, different training corpora. That does not make them
independent forecasts. The daily spread is the quantity that tells you whether averaging
them buys any diversification at all.

$$D_t = \max_i \widehat{\mathrm{VaR}}_{i,t} - \min_i \widehat{\mathrm{VaR}}_{i,t},
\qquad
R_t = \frac{\max_i \widehat{\mathrm{VaR}}_{i,t}}{\min_i \widehat{\mathrm{VaR}}_{i,t}}$$

This is the idea behind SYNCRISK: correlated risk *assessments* are themselves a source
of systemic risk, because everyone de-risks on the same day.
""", "optional", mins="seconds")

code(r"""
spread = al.disagreement({k: v for k, v in models.items() if k != "Chronos-Bolt"})
fig, ax = plt.subplots(figsize=(11, 4.2))
al.plot_disagreement(spread, ax=ax)
plt.show()

worst = spread["range"].idxmax()
print(f"widest disagreement: {worst.date()}  range {spread.loc[worst, 'range']:.2f}pp  "
      f"ratio {spread.loc[worst, 'ratio']:.2f}x")
print(f"realised return that day: {bench.loc[worst, 'ret']:.2f}%")
""")

code(r"""
wide = pd.DataFrame({k: v for k, v in models.items()}).dropna()
print("pairwise correlation of the daily VaR forecasts\n")
print(wide.corr().round(3).to_string())
""")

md(r"""
**Question 5.** Take the two most correlated models in that matrix. Are they correlated
because they measure the same thing well, or because one of them is downstream of the
other? How would you tell the two cases apart from the numbers alone?
""")

md(r"""
### Five questions to answer before you leave

Write two or three sentences for each. They are the questions the lecture was built
around, and the ones the report from your group should answer.

1. **Pipeline validity.** You asked Chronos-Bolt for the 1% quantile and the call
   succeeded. What evidence did you use to decide whether the returned number *was*
   the 1% quantile, and what would you have concluded without that check?

2. **Ranking against calibration.** Order the models by pinball loss, then by their
   Kupiec verdict. Are the two orderings the same? Which question does each answer, and
   which one would you take to a risk committee?

3. **Power.** Your span expects five breaches. Given the detectable-rate table in
   section 1.1, what is the smallest true breach rate this backtest could have
   rejected? What does a "pass" therefore entitle you to say?

4. **Contamination.** The `dated` configuration names the asset and the date. What could
   the model be doing on those dates other than forecasting, and which of your results
   would change if it were? Note what you can establish from outputs alone and what you
   cannot.

5. **Experimental control.** Why is `dated` the right comparison for `dated+news`, and
   what would you have measured had you compared `series` with `dated+news` instead?
""", "interpret")

md(r"""
## 8. What to take away, and the optional extension

1. **A zero-shot foundation model produces a number for any question you ask it.**
   Whether that number answers the question you asked is a separate matter, and the
   Chronos-Bolt clamp is the cheap version of a failure that is usually expensive.
2. **An LLM can state a VaR, and the statement can be well or badly formed.**
   Count the replies that do not parse and the replies that parse into a nonsense sign,
   before reading anything into the ones that survive.
3. **Ranking and validation are different operations.** The pinball loss orders models;
   the coverage and independence tests decide whether any of them may be used. A model
   can win the first and fail the second.
4. **Correlated models are not a portfolio of opinions.** Check the spread before
   treating an ensemble as diversification.

### Extensions

Everything below runs from the files you already have. **None of them needs an API key**
— which is itself the point of the last one.

- **Swap the asset.** All five carry the same models. Bitcoin is a different
  regime, the Nikkei's news coverage is half the S&P's, and the ranking is not stable
  across them. Change `ASSET` in the first cell and rerun.
- **Go to $\alpha = 1\%$.** `load_benchmarks(ASSET, level=0.01)` and the Chronos-T5
  file both carry it. Five expected breaches in 500 days: which tests still have the
  power to reject anything, and what does that do to the leaderboard?
- **The one you cannot do, and why that matters.** The 1% extension above stops at the
  LLM: asking the model for a 1% quantile is a *new set of 500 requests*, which you have
  no key for. Historical simulation, GARCH and Chronos-T5 gave you a new level for free,
  because you hold the model. That asymmetry — a level costs nothing from a model you
  run, and a full re-run from a model you rent — belongs in any comparison of the two.
""")


# ===========================================================================
def build():
    cells = []
    for n, (kind, src) in enumerate(CELLS):
        # Stable ids so a regenerated notebook diffs cleanly against the previous one.
        cell = {"cell_type": kind, "id": f"aida-{n:03d}", "metadata": {},
                "source": (src + "\n").splitlines(keepends=True)}
        if kind == CODE:
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = HERE / "AIDA_Risk_Lab.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    n_code = sum(1 for k, _ in CELLS if k == CODE)
    print(f"{out.name}: {len(CELLS)} cells ({n_code} code, {len(CELLS) - n_code} markdown)")


if __name__ == "__main__":
    build()
