"""Stage 06 — real news headlines from EODHD, aligned to the forecast dates.

This is the file that makes a text-and-risk experiment possible without inventing
anything. Headlines are fetched from EODHD's financial-news endpoint, stored raw with
their publication timestamps, and then aligned to forecast dates under one rule.

THE ALIGNMENT RULE, which is the whole point of the stage.

    A headline may enter the prompt for forecast date t only if it was published at or
    before the as-of cutoff on t-1.

The cutoff is that ASSET'S OWN market close, in UTC, taken from `AS_OF_HOURS` below --
20:00 for the S&P, 06:00 for the Nikkei, and so on. A single US close applied to every
asset would hand a Tokyo forecast fourteen hours of news published after Tokyo shut.
Every headline carries its publication timestamp and its cutoff into the aligned file,
so the rule can be audited rather than trusted, and `--audit` re-checks both.

This matters more here than anywhere else in the package. A price series can only leak
by being shifted; a news series leaks whenever an article published at 22:00 describing
the day's close is handed to a model asked to forecast that close. Nothing in the reply
would look wrong.

WHY THE COMPARISON IS AGAINST `dated`, NOT `series`.

Real headlines necessarily reveal the period: they name companies, events and numbers
that place the text in time. So `series` (anonymised) against `series+news` would
confound two effects, "news carries information" and "the model now knows which period
this is". The identified comparison is

    dated          returns + asset name + forecast date, no headlines      (control)
    dated+news     the same, plus the headlines available at t-1           (treatment)

Both reveal the period; only one carries text.

Outputs:
    data/news/raw_<SYMBOL>.jsonl          every article returned, as returned
    data/news/headlines_<ASSET>.csv       date, asof_utc, published_utc, headline
    data/news/coverage_<ASSET>.csv        per-date headline counts

Usage:
    python src/06_news.py --asset SPX --probe          # 1 request, reports what comes back
    python src/06_news.py --asset SPX                  # fetch + align
    python src/06_news.py --asset SPX --audit          # re-check the alignment rule
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labcommon import ASSETS, LAB, ROOT  # noqa: E402

NEWS = ROOT / "data" / "news"
NEWS.mkdir(parents=True, exist_ok=True)

API = "https://eodhd.com/api/news"
# The as-of cutoff is the hour, in UTC, at which THAT asset's trading day closes --
# measured per exchange, not assumed. Using New York's close for the Nikkei would hand
# the model fourteen hours of news published after Tokyo shut, which is the exact leak
# the rule exists to prevent. Where a market observes daylight saving the EARLIER of the
# two UTC closes is used, so the window is never too generous.
#
#   SPX   ^GSPC   NYSE      16:00 ET      = 21:00 UTC (EST) / 20:00 (EDT)  -> 20.0
#   DAX   ^GDAXI  Xetra     17:30 CET     = 16:30 UTC (CET) / 15:30 (CEST) -> 15.5
#   N225  ^N225   TSE       15:00 JST     = 06:00 UTC, no DST in Japan     ->  6.0
#   GOLD  GC=F    COMEX     13:30 ET pit  = 18:30 UTC (EST) / 17:30 (EDT)  -> 17.5
#   BTC   BTC-USD 24/7      Yahoo's daily bar runs 00:00-24:00 UTC         -> 24.0
AS_OF_HOURS = {"SPX": 20.0, "DAX": 15.5, "N225": 6.0, "GOLD": 17.5, "BTC": 24.0}
LOOKBACK_HOURS = 24        # first forecast only, where there is no previous cutoff
MAX_PER_DATE = 5           # headlines per prompt
PAGE = 1000                # EODHD page size

# EODHD symbols. Every entry was probed over 2023-01-01..2024-12-30 and chosen on the article count
# returned, not on which ticker looked right. Counts on the first page (cap 1000):
#   SPX   GSPC.INDX 1000  |  SPY.US 222, QQQ.US 200, VOO.US 22, DIA.US 63
#   DAX   GDAXI.INDX 1000 |  DAX.INDX 0, EXS1.XETRA 0, DAX.XETRA 0, EWG.US 0
#   N225  N225.INDX 444   |  NI225.INDX 0, EWJ.US 11, 1321.TSE 404
#   GOLD  XAUUSD.FOREX 1000 | GLD.US 89, IAU.US 3, GC.COMM 404, GC=F.COMM 422
#   BTC   BTC-USD.CC 1000  |  BTC.CC 147, BTCUSD.CC 0, GBTC.US 174
# N225 is the thin one and its coverage is reported rather than smoothed over.
# GOLD news is quoted on spot XAU/USD while the price series is the COMEX future: the
# same market, two instruments, and the mismatch is stated rather than hidden.
SYMBOLS = {
    "SPX":  "GSPC.INDX",
    "DAX":  "GDAXI.INDX",
    "N225": "N225.INDX",
    "GOLD": "XAUUSD.FOREX",
    "BTC":  "BTC-USD.CC",
}


def _key():
    k = os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_API_TOKEN")
    if not k:
        raise SystemExit(
            "EODHD_API_KEY is not set.\n"
            "Add `export EODHD_API_KEY=...` to ~/.zshrc (that is where the other keys "
            "in this project live), then re-run. Nothing was fetched and nothing was "
            "invented.")
    return k


def _get(params):
    url = f"{API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as fh:
        return json.loads(fh.read().decode())


def fetch(symbol, start, end, probe=False):
    """Page through the news endpoint. Returns the raw article list."""
    base = {"s": symbol, "from": start, "to": end, "limit": PAGE,
            "api_token": _key(), "fmt": "json"}
    if probe:
        got = _get({**base, "limit": 5, "offset": 0})
        print(f"probe {symbol}: {len(got)} articles returned")
        for a in got[:3]:
            print(f"  {a.get('date')}  {str(a.get('title'))[:96]}")
        if got:
            print(f"  fields: {sorted(got[0])}")
        return got

    out, offset = [], 0
    while True:
        page = _get({**base, "offset": offset})
        out.extend(page)
        print(f"  offset {offset:>6}: +{len(page):>4} articles", flush=True)
        if len(page) < PAGE:
            break
        offset += PAGE
        time.sleep(0.3)
    return out


def _cutoff(day, asset):
    """UTC instant at which `asset` closed on `day` -- the as-of point of a forecast."""
    return day.tz_localize("UTC") + timedelta(hours=AS_OF_HOURS[asset])


def align(raw, dates, asset):
    """Attach headlines to forecast dates under the as-of rule.

    Returns (headlines, coverage). `headlines` carries the publication timestamp of
    every row, so the rule is checkable after the fact.
    """
    art = pd.DataFrame(raw)
    if art.empty:
        return pd.DataFrame(columns=["date", "published_utc", "headline"]), pd.DataFrame()
    art["published_utc"] = pd.to_datetime(art["date"], utc=True, errors="coerce")
    art["title"] = art["title"].astype(str).str.strip()
    art = art.dropna(subset=["published_utc"]).drop_duplicates(subset=["title"])
    art = art.sort_values("published_utc")

    # The window for forecast date d_i runs from the PREVIOUS decision point to this
    # one: (cutoff(d_{i-2}), cutoff(d_{i-1})]. Using a fixed 24-hour window instead
    # would drop every article published between the cutoff and midnight, and would
    # drop whole weekends, because those hours fall in no window at all. Consecutive
    # cutoffs tile the calendar with no gap and no overlap, so each article is offered
    # to exactly one forecast -- the first one that could legitimately have seen it.
    idx = pd.DatetimeIndex(dates)

    rows = []
    for i, d in enumerate(idx):
        hi = _cutoff(idx[i - 1] if i >= 1 else d - timedelta(days=1), asset)
        lo = _cutoff(idx[i - 2], asset) if i >= 2 else hi - timedelta(hours=LOOKBACK_HOURS)
        w = art[(art["published_utc"] > lo) & (art["published_utc"] <= hi)]
        for _, a in w.tail(MAX_PER_DATE).iterrows():     # most recent first-in-window
            rows.append({"date": d, "asof_utc": hi,
                         "published_utc": a["published_utc"], "headline": a["title"]})

    h = pd.DataFrame(rows)
    cov = (h.groupby("date").size().rename("n_headlines")
           .reindex(pd.DatetimeIndex(dates), fill_value=0).rename_axis("date")
           .reset_index()) if not h.empty else pd.DataFrame()
    return h, cov


def audit(asset, dates):
    """Re-check the as-of rule on the aligned file. Non-zero exit on any violation.

    Two independent checks, because the stored cutoff could itself be wrong:
      1. every headline was published at or before the cutoff stored on its row;
      2. that stored cutoff equals the previous forecast date's 20:00 UTC, recomputed
         here from the forecast-date index rather than read from the file.
    """
    h = pd.read_csv(NEWS / f"headlines_{asset}.csv",
                    parse_dates=["date", "asof_utc", "published_utc"])
    for c in ("asof_utc", "published_utc"):
        if h[c].dt.tz is None:
            h[c] = h[c].dt.tz_localize("UTC")

    idx = pd.DatetimeIndex(dates)
    prev = {d: (idx[i - 1] if i >= 1 else d - timedelta(days=1))
            for i, d in enumerate(idx)}
    want = h["date"].map(lambda d: _cutoff(prev[d], asset))

    late = h[h["published_utc"] > h["asof_utc"]]
    wrong = h[h["asof_utc"] != want]
    print(f"audit {asset}: {len(h)} headlines over {h['date'].nunique()} dates")
    print(f"  published after their own as-of cutoff: {len(late)}")
    print(f"  rows whose as-of cutoff is not the previous forecast date at "
          f"{AS_OF_HOURS[asset]:.1f}h UTC: {len(wrong)}")
    if len(late) or len(wrong):
        print(pd.concat([late, wrong]).head(10).to_string(index=False))
        raise SystemExit(1)
    lag = (h["asof_utc"] - h["published_utc"]).dt.total_seconds() / 3600.0
    print(f"  publication lead before the cutoff: median {lag.median():.1f}h, "
          f"min {lag.min():.1f}h, max {lag.max():.1f}h")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="SPX", choices=list(ASSETS))
    p.add_argument("--symbol", default=None, help="override the EODHD symbol")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--audit", action="store_true")
    a = p.parse_args()

    bench = pd.read_csv(LAB / f"bench_{a.asset}.csv", parse_dates=["date"])
    dates = pd.DatetimeIndex(sorted(bench["date"].unique()))

    if a.audit:
        audit(a.asset, dates)
        return

    sym = a.symbol or SYMBOLS[a.asset]
    start = (dates[0] - timedelta(days=5)).strftime("%Y-%m-%d")
    end = dates[-1].strftime("%Y-%m-%d")
    print(f"{a.asset} via {sym}: {start} .. {end}, {len(dates)} forecast dates")

    raw = fetch(sym, start, end, probe=a.probe)
    if a.probe:
        return

    rp = NEWS / f"raw_{sym.replace('.', '_')}.jsonl"
    with rp.open("w") as fh:
        for art in raw:
            fh.write(json.dumps(art) + "\n")
    print(f"  raw -> {rp.name}  ({len(raw)} articles)")

    h, cov = align(raw, dates, a.asset)
    h.to_csv(NEWS / f"headlines_{a.asset}.csv", index=False)
    cov.to_csv(NEWS / f"coverage_{a.asset}.csv", index=False)

    if cov.empty:
        print("  no headline fell inside any as-of window; nothing aligned")
        return
    n0 = int((cov["n_headlines"] == 0).sum())
    print(f"  aligned -> headlines_{a.asset}.csv: {len(h)} headlines over "
          f"{len(cov) - n0} of {len(cov)} dates")
    print(f"  dates with no headline at all: {n0} "
          f"({100 * n0 / len(cov):.1f}%) -- these get a returns-only prompt")
    print(f"  mean headlines per covered date: "
          f"{cov.loc[cov['n_headlines'] > 0, 'n_headlines'].mean():.1f}")
    audit(a.asset, dates)


if __name__ == "__main__":
    main()
