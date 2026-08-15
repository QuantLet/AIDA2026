"""Stage 09 — LLM-VaR by sampling: draw completions, read the quantile off them.

The method is Pele, Bolovaneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma, Lessmann and
Haerdle (2025), which follows LLMTime (Gruver et al., 2023).

NOT A REPLICATION, AND THE DIFFERENCE IS IN THE PROMPT. Checked against models/gpt.py
and models/llmtime.py at github.com/QuantLet/LLM_Risk on 2026-08-15, this stage differs
from the published protocol in two ways:

    1. The published prompt wraps the sequence in a system message ("You are a helpful
       assistant that performs time series predictions...") and a user instruction
       ("Please continue the following sequence without producing any additional
       text..."). This stage sends the serialised sequence and nothing else.
    2. The published pipeline rescales each context with get_scaler(alpha=0.95,
       beta=0.3) before serialising. This stage serialises the raw returns.

Both are defensible for a base model prompted for raw continuation, and neither is what
the paper ran. Any comparison drawn from this stage is a comparison of extraction
methods on the same weights, not a reproduction of the published numbers, and the deck
says so on the frame that shows the two prompts side by side.

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


# The published wrapper, transcribed from models/gpt.py at github.com/QuantLet/LLM_Risk
# on 2026-08-15. --wrapper reproduces it; the default keeps the bare-continuation prompt
# the precomputed files were produced with, so old and new runs cannot be confused.
PUB_SYSTEM = ("You are a helpful assistant that performs time series predictions. The "
              "user will provide a sequence and you will predict the remaining "
              "sequence. The sequence is represented by decimal strings separated by "
              "commas.")
PUB_USER = ("Please continue the following sequence without producing any additional "
            "text. Do not say anything like 'the next terms in the sequence are', just "
            "return the numbers. Sequence:\n")


def get_scaler(history, alpha=0.95, beta=0.3):
    """The published affine scaler, transcribed from get_scaler in models/llmtime.py at
    github.com/QuantLet/LLM_Risk, retrieved 2026-08-15, with basic=False as the paper
    runs it.

    Note what it does to a return series: min_ sits BELOW the sample minimum by beta of
    the range, so every scaled value is positive and the serialised sequence carries no
    minus signs at all. The model therefore never sees a negative token, which is a very
    different input from the raw returns this stage sent before.
    """
    history = history[~np.isnan(history)]
    min_ = np.min(history) - beta * (np.max(history) - np.min(history))
    q = float(np.quantile(history - min_, alpha))
    if q == 0:
        q = 1.0
    return (lambda x: (x - min_) / q), (lambda x: x * q + min_)


def wrap_published(tok, series_str):
    """The sequence inside the published system message and instruction."""
    msgs = [{"role": "system", "content": PUB_SYSTEM},
            {"role": "user", "content": PUB_USER + series_str}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def draw(mod, tok, dev, prompt, n, temp, batch=BATCH, inv=None):
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
                # Scaled draws come back on the scaled axis, so they are mapped
                # home before the sanity filter, which is stated in return units.
                if inv is not None:
                    v = float(inv(v))
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
    p.add_argument("--scaler", action="store_true",
                   help="apply the published affine scaler before serialising and map "
                        "the draws back afterwards; writes to its own output file")
    p.add_argument("--wrapper", action="store_true",
                   help="wrap the sequence in the published system message and "
                        "instruction; writes to its own output file")
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
    # the prompt is part of the identity of a run, so it is part of the file name
    wrap = ("_pubprompt" if a.wrapper else "") + ("_scaled" if a.scaler else "")
    stem = f"llmtime_{a.asset}_{slug}_T{a.temp}_N{a.samples}{prec}{wrap}{suffix}"
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
        inv = None
        ser = hist
        if a.scaler:
            fwd, inv = get_scaler(hist)
            ser = fwd(hist)
        prompt = serialise(ser) + ","
        if a.wrapper:
            prompt = wrap_published(tok, prompt)
        s = draw(mod, tok, dev, prompt, a.samples, a.temp, a.batch, inv=inv)
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
