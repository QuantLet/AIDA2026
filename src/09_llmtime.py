"""Stage 09 — LLM-VaR the published way: sample completions, read the quantile off them.

This is the method of Pele, Bolovaneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma,
Lessmann and Haerdle (2025), which follows LLMTime (Gruver et al., 2023), verified
against the published code at github.com/QuantLet/LLM_Risk.

    serialise the returns  ->  sample N completions at temperature T  ->  VaR is the
    empirical alpha-quantile of the sampled next values

It is architecturally the same object as the Chronos-T5 sampling head, with a general
language model in place of a pretrained time-series model, and it has the same floor:
at alpha = 1% with N = 200 draws, two draws lie below the quantile.

WHAT THIS IS NOT. Stage 03 asks the model for the quantile in words and parses the
reply. That is *elicitation*, not sampling, and it is a different method however
similar the prompt looks. Both are run here so the difference can be measured instead
of argued about.

SERIALISATION. The published settings are
    SerializerSettings(base=10, prec=2, signed=True, time_sep=', ', bit_sep='',
                       minus_sign='-')
which is exactly ", ".join(f"{x:.2f}") -- the format stage 03 already uses. So the two
legs share the input representation and differ only in how the number is extracted.

Open weights only. Sampling needs N generations per forecast, which is where a
commercial endpoint becomes expensive and a local model does not.

Output: precomputed/llmtime_<ASSET>_<model-slug>_T<temp>.csv
    date, var_01, var_05, var, n_draws, raw_ok, sign_ok, order_ok

Usage:
    python src/09_llmtime.py --asset SPX --limit 60
    python src/09_llmtime.py --asset SPX --samples 200 --temp 0.7
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labcommon import ALPHA, ALPHA_SECONDARY, ASSETS, LAB, PRECOMP, SEED  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT_DAYS = 60
N_SAMPLES = 200          # draws per forecast; sets the quantile noise floor
TEMP = 0.7               # the published grid runs 0 to 1; 0.7 is LLMTime's default
MAX_NEW = 12             # one number is a handful of tokens
BATCH = 32               # draws are independent, so they batch perfectly

NUM = re.compile(r"-?\d+\.?\d*")


def device_of():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def serialise(x):
    """The published format: base 10, two decimals, signed, ', ' between steps."""
    return ", ".join(f"{v:.2f}" for v in x)


def draw(mod, tok, dev, prompt, n, temp, batch=BATCH):
    """Sample n continuations of the serialised series; return the parsed next values."""
    import torch
    out = []
    enc = tok([prompt], return_tensors="pt").to(dev)
    for i in range(0, n, batch):
        k = min(batch, n - i)
        with torch.no_grad():
            gen = mod.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                               temperature=temp, top_p=0.9, num_return_sequences=k,
                               pad_token_id=tok.pad_token_id)
        for row in gen:
            txt = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = NUM.search(txt)
            if m:
                try:
                    v = float(m.group(0))
                except ValueError:
                    continue
                # A completion is only usable if it looks like a daily return at all.
                # Values beyond +/-40% are decoding failures, not forecasts, and are
                # dropped rather than winsorised: the count is reported.
                if abs(v) <= 40:
                    out.append(v)
    return np.array(out, dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="SPX", choices=list(ASSETS))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--temp", type=float, default=TEMP)
    p.add_argument("--batch", type=int, default=BATCH,
                   help="draws generated per forward pass; pure throughput")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dtype", default="", choices=["", "float32", "float16", "bfloat16"],
                   help="override the weight precision; default is fp16 on CUDA, "
                        "fp32 elsewhere")
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    rets = pd.read_csv(LAB / f"returns_{a.asset}.csv", index_col=0, parse_dates=True)
    bench = pd.read_csv(LAB / f"bench_{a.asset}.csv", parse_dates=["date"])
    dates = pd.DatetimeIndex(sorted(bench["date"].unique()))
    if a.limit:
        dates = dates[:a.limit]

    dev = device_of()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    dtype = getattr(torch, a.dtype) if a.dtype else (
        torch.float16 if dev == "cuda" else torch.float32)
    mod = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()

    print(f"{a.asset} / {a.model} on {dev}: {len(dates)} forecasts, "
          f"{a.samples} draws each at T={a.temp}")
    print(f"  floor: {a.samples * ALPHA:.0f} draws below the {ALPHA:.0%} quantile")

    # Checkpoint every date. The Chronos stage has had this from the start; this one
    # did not, and a four-hour run with nothing on disk until the end is a run you
    # cannot move to another machine, cannot interrupt and cannot survive a crash with.
    # The draw count is part of the identity of a run, not a tuning knob: a 200-draw
    # and a 1000-draw forecast for the same date are different numbers. It goes in the
    # filename so a raised run cannot resume from a checkpoint written at a lower count.
    slug = a.model.split("/")[-1]
    suffix = f"_smoke{len(dates)}" if a.limit else ""
    prec = f"_{a.dtype}" if a.dtype else ""
    stem = f"llmtime_{a.asset}_{slug}_T{a.temp}_N{a.samples}{prec}{suffix}"
    path = PRECOMP / f"{stem}.csv"
    ckpt = PRECOMP / f"_{stem}.csv"

    done = {}
    if ckpt.exists():
        prev = pd.read_csv(ckpt, parse_dates=["date"])
        done = {d: r for d, r in zip(prev["date"], prev.to_dict("records"))}
        print(f"  resuming: {len(done)} forecasts already on disk")

    rows, t0, n_new = [], time.time(), 0
    for i, d in enumerate(dates):
        if d in done:
            rows.append(done[d])
            continue
        hist = rets.loc[:d, "ret"].iloc[:-1].tail(CONTEXT_DAYS).values
        prompt = serialise(hist) + ","
        s = draw(mod, tok, dev, prompt, a.samples, a.temp, a.batch)
        if len(s) >= 20:
            q01, q05 = float(np.quantile(s, ALPHA)), float(np.quantile(s, ALPHA_SECONDARY))
            rows.append({"date": d, "var_01": -q01, "var_05": -q05, "var": -q01,
                         "n_draws": len(s), "raw_ok": True,
                         "sign_ok": bool(q01 < 0 and q05 < 0),
                         "order_ok": bool(q01 <= q05)})
        else:
            rows.append({"date": d, "var_01": np.nan, "var_05": np.nan, "var": np.nan,
                         "n_draws": len(s), "raw_ok": False,
                         "sign_ok": False, "order_ok": False})
        n_new += 1
        pd.DataFrame([rows[-1]]).to_csv(ckpt, mode="a", header=not ckpt.exists(),
                                        index=False)
        if n_new % 10 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(dates)}  {el / 60:.1f}m  "
                  f"eta {el / n_new * (len(dates) - i - 1) / 60:.1f}m", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)

    ok = out["raw_ok"] & out["sign_ok"] & out["order_ok"]
    print(f"\n  usable {int(ok.sum())}/{len(out)}; median draws kept "
          f"{out['n_draws'].median():.0f} of {a.samples}")
    if ok.any():
        v = out.loc[ok, "var_01"]
        print(f"  VaR(1%): mean {v.mean():.2f}  sd {v.std():.2f}  "
              f"distinct {v.round(2).nunique()} of {int(ok.sum())}")
    print(f"  -> {path.name}")


if __name__ == "__main__":
    main()
