"""Stage 03 — LLM risk forecasts (LLM-VaR), three information sets.

The method is the one in Pele, Bolovaneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma,
Lessmann and Haerdle (2025), "In the Beginning was the Word: LLM-VaR and LLM-ES",
Expert Systems with Applications: the return history is serialised as text, the model
is asked for a tail quantile, and the number is read out of the reply. The model has no
VaR head and is not fine-tuned; the forecast comes from prompted generation.

INFORMATION SETS. Four, and the differences between them are the exercise.

    series        the last CONTEXT_DAYS returns, nothing else
    series+state  the same returns plus derived state: realised volatility at two
                  horizons, the current drawdown, and the rolling historical-simulation
                  VaR. Tests whether the model adds anything to statistics a student
                  can compute in one line.
    dated         the same returns, plus the asset name and the forecast date
    dated+news    the same as `dated`, plus the real headlines available at t-1

The first two are anonymised on purpose. No ticker, no dates, no price levels, so the
model cannot look the answer up in what it memorised. Returns are rounded to two
decimals, which is what a reader of a price table would see.

WHY `dated+news` IS PAIRED WITH `dated` AND NOT WITH `series`. Real headlines
necessarily reveal the period: they name companies, events and numbers that place the
text in time. Comparing `series` with a news configuration would therefore confound two
effects -- "the text carries information" and "the model now knows which period this
is". `dated` is the control that holds the second fixed, so the difference between the
two identifies the first. `dated` on its own is also the contamination probe: the model
was trained on data covering these dates, so a good score there can be recall rather
than inference.

NEWS. Real, or absent. Headlines come from EODHD via src/06_news.py, which aligns them
to forecast dates under an explicit as-of rule and audits that rule. Nothing is
invented: a headline file written to fit the returns would produce a text-to-risk
relation that exists only because it was constructed. Dates with no headline in their
window get the returns-only prompt, so `dated+news` equals `dated` on those days by
construction and the informative comparison is on the covered subset.

Output, one row per date, into precomputed/llm_<config>_<ASSET>_<model>.csv:
    date, var, prob_negative, prob_tail, risk_score, confidence, raw_ok

Usage:
    python src/03_llm_lab.py --config series --asset SPX --model claude-haiku-4-5
    python src/03_llm_lab.py --config series+state --asset SPX --live --limit 20
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labcommon import ALPHA, ASSETS, HS_WINDOW, LAB, PRECOMP, ROOT  # noqa: E402

CONTEXT_DAYS = 60          # returns shown to the model
MAX_TOKENS = 400
DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM = (
    "You are a market risk analyst. From a series of recent daily returns you estimate "
    "the risk of the next trading day. You answer with a single JSON object and nothing "
    "else: no explanation before it, no code fence around it."
)

# Both quantiles are asked for in ONE call. The 1% level is what the course reports,
# but a second field costs a handful of output tokens where a second run would cost a
# second 500-request batch, so the 5% column comes along for nothing and the two levels
# are guaranteed to come from the same reply rather than from two sessions of the model.
SCHEMA_TEXT = """Return exactly this JSON object:
{
  "q01": <number>,            // the 1% quantile of tomorrow's return, in percent. Negative.
  "q05": <number>,            // the 5% quantile of tomorrow's return, in percent. Negative.
  "prob_negative": <number>,  // probability tomorrow's return is below zero, 0 to 1
  "prob_tail": <number>,      // probability tomorrow's return is below -2%, 0 to 1
  "risk_score": <integer>,    // 0 calm to 100 extreme
  "confidence": <number>,     // your confidence in q01, 0 to 1
  "drivers": [<string>, ...]  // at most three short phrases
}

q01 must be more negative than q05: a 1% quantile lies further into the left tail."""


def _fmt(xs):
    return ", ".join(f"{x:.2f}" for x in xs)


def build_prompt(config, hist, asset=None, date=None, headlines=None):
    """Assemble the user message for one forecast date. `hist` ends at t-1."""
    r = hist.values.astype(float)
    ctx = r[-CONTEXT_DAYS:]
    lines = []

    if config.startswith("dated"):
        lines.append(f"Asset: {ASSETS[asset]['label']}. Forecast date: {date:%Y-%m-%d}.")

    lines.append(f"Daily returns in percent, oldest first, {len(ctx)} observations:")
    lines.append(_fmt(ctx))

    if config == "series+state":
        vol20 = float(np.std(r[-20:], ddof=1))
        vol60 = float(np.std(r[-60:], ddof=1))
        cum = np.cumsum(r[-250:])
        drawdown = float(cum[-1] - np.max(cum))
        hs = float(-np.quantile(r[-HS_WINDOW:], ALPHA)) if len(r) >= HS_WINDOW else float("nan")
        lines += [
            "",
            "Derived state, computed from the same history:",
            f"  realised volatility, last 20 days: {vol20:.2f}% per day",
            f"  realised volatility, last 60 days: {vol60:.2f}% per day",
            f"  drawdown from the 250-day peak: {drawdown:.2f}%",
            f"  historical-simulation 5% VaR over {HS_WINDOW} days: {hs:.2f}%",
        ]

    if headlines:
        lines += ["", "Headlines published before the close of the last observation:"]
        lines += [f"  - {h}" for h in headlines]

    lines += ["", SCHEMA_TEXT]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing: the model is not trusted to return clean JSON
# ---------------------------------------------------------------------------
FIELDS = ("q01", "q05", "prob_negative", "prob_tail", "risk_score", "confidence")


def parse_reply(text):
    """Extract the JSON object and check every field. Returns (dict, ok).

    A reply that parses but carries a positive `q05` is a failure, not a datum: the
    5% quantile of a daily return distribution is below zero, and a model that
    returns +1.3 has answered a different question. Those rows are marked and kept,
    because how often a model does this is itself a result.
    """
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}, False
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}, False

    out, ok = {}, True
    for k in FIELDS:
        v = d.get(k)
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k], ok = float("nan"), False
    if ok and not (np.isfinite(out["q01"]) and np.isfinite(out["q05"])):
        ok = False
    # Three separate things can be wrong with a reply that parses, and they are counted
    # separately because they fail differently: a positive quantile asserts the tail is
    # a gain, and a 1% quantile above the 5% one asserts the distribution is inverted.
    out["sign_ok"] = bool(np.isfinite(out["q01"]) and out["q01"] < 0
                          and np.isfinite(out["q05"]) and out["q05"] < 0)
    out["order_ok"] = bool(np.isfinite(out["q01"]) and np.isfinite(out["q05"])
                           and out["q01"] <= out["q05"])
    return out, ok


def _rows(recs, parsed):
    rows = []
    for (date, _), (d, ok) in zip(recs, parsed):
        rows.append({
            "date": date,
            "var_01": -d.get("q01", float("nan")),   # positive-loss convention
            "var_05": -d.get("q05", float("nan")),
            "var": -d.get("q01", float("nan")),      # `var` is the reported level, 1%
            "prob_negative": d.get("prob_negative", float("nan")),
            "prob_tail": d.get("prob_tail", float("nan")),
            "risk_score": d.get("risk_score", float("nan")),
            "confidence": d.get("confidence", float("nan")),
            "raw_ok": bool(ok),
            "sign_ok": bool(d.get("sign_ok", False)),
            "order_ok": bool(d.get("order_ok", False)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_live(client, model, recs):
    """One request per date. Slower and dearer than a batch; useful for a smoke test."""
    parsed, t0 = [], time.time()
    for i, (_, prompt) in enumerate(recs):
        msg = client.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        parsed.append(parse_reply(text))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    return parsed


def run_batch(client, model, recs, poll=30):
    """Message Batches API: half price, and the whole run is one object to watch."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    reqs = [
        Request(
            custom_id=f"d{i:05d}",
            params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=MAX_TOKENS, system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        for i, (_, prompt) in enumerate(recs)
    ]
    batch = client.messages.batches.create(requests=reqs)
    print(f"  batch {batch.id}: {len(reqs)} requests submitted", flush=True)

    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        c = b.request_counts
        print(f"  {b.processing_status}  done {c.succeeded}/{len(reqs)}  "
              f"errored {c.errored}  {time.time() - t0:.0f}s", flush=True)
        time.sleep(poll)

    by_id = {}
    for res in client.messages.batches.results(batch.id):
        if res.result.type == "succeeded":
            text = "".join(x.text for x in res.result.message.content if x.type == "text")
            by_id[res.custom_id] = parse_reply(text)
        else:
            by_id[res.custom_id] = ({}, False)
    print(f"  batch finished in {time.time() - t0:.0f}s", flush=True)
    return [by_id.get(f"d{i:05d}", ({}, False)) for i in range(len(recs))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="series",
                   choices=["series", "series+state", "dated", "dated+news"])
    p.add_argument("--asset", default="SPX", choices=list(ASSETS))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0, help="first N dates only (smoke test)")
    p.add_argument("--live", action="store_true", help="one request per date instead of a batch")
    p.add_argument("--news-file", default=None, help="CSV of date,headline (real news only)")
    a = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; nothing was sent and nothing was faked")

    rets = pd.read_csv(LAB / f"returns_{a.asset}.csv", index_col=0, parse_dates=True)
    bench = pd.read_csv(LAB / f"bench_{a.asset}.csv", parse_dates=["date"])
    dates = pd.DatetimeIndex(sorted(bench["date"].unique()))
    if a.limit:
        dates = dates[:a.limit]

    news = {}
    if a.config == "dated+news" and not a.news_file:
        a.news_file = ROOT / "data" / "news" / f"headlines_{a.asset}.csv"
    if a.news_file:
        nf = pd.read_csv(a.news_file, parse_dates=["date"])
        news = nf.groupby("date")["headline"].apply(list).to_dict()
        print(f"headlines: {len(nf)} over {len(news)} dates from "
              f"{Path(a.news_file).name}")
    if a.config == "dated+news" and not news:
        raise SystemExit("dated+news needs aligned headlines; run src/06_news.py first")

    recs = []
    for d in dates:
        hist = rets.loc[:d, "ret"].iloc[:-1]        # ends at t-1; day d is excluded
        recs.append((d, build_prompt(a.config, hist, a.asset, d, news.get(d))))

    import anthropic
    client = anthropic.Anthropic()
    print(f"{a.config} / {a.asset} / {a.model}: {len(recs)} forecasts, "
          f"{'live' if a.live else 'batch'}")
    parsed = (run_live if a.live else run_batch)(client, a.model, recs)

    out = _rows(recs, parsed)
    out["n_headlines"] = [len(news.get(d, [])) for d, _ in recs]
    tag = a.config.replace("+", "_")

    # A --limit run is a smoke test and must never land on the canonical filename.
    # It did once, silently replacing a finished 500-day file with a 2-row one, and
    # nothing downstream would have complained: the loaders would simply have scored
    # two days. Partial runs get their own name.
    suffix = f"_smoke{len(recs)}" if a.limit else ""
    path = PRECOMP / f"llm_{tag}_{a.asset}_{a.model}{suffix}.csv"
    out.to_csv(path, index=False)

    n_bad = int((~out["raw_ok"]).sum())
    n_sign = int((~out["sign_ok"] & out["raw_ok"]).sum())
    n_ord = int((~out["order_ok"] & out["raw_ok"]).sum())
    print(f"\n  parsed {len(out) - n_bad}/{len(out)}; {n_sign} with a non-negative "
          f"quantile; {n_ord} with q01 above q05 (inverted tail)")
    ok = out["sign_ok"] & out["order_ok"]
    print(f"  mean VaR(1%) {out.loc[ok, 'var_01'].mean():.2f}   "
          f"mean VaR(5%) {out.loc[ok, 'var_05'].mean():.2f}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
