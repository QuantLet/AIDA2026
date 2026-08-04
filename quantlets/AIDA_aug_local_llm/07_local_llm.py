"""Stage 07 — the same LLM-VaR experiment with an open-weights model students can run.

A commercial endpoint is not available to everyone in the room, and a laboratory whose
central exercise requires a credit card is not a laboratory. This stage runs the
identical prompt through a model with open weights, downloaded from Hugging Face the
same way Chronos is, on whatever device is present. Anyone with a Colab session can
reproduce every number in it.

The prompt, the schema and the parser are imported from stage 03 rather than copied, so
the open and the commercial legs differ in exactly one thing: which model reads the
prompt. That is the point of the comparison.

MODEL CHOICE, MEASURED. Probed 2026-07-31 on eight dates spanning the widest range of
realised volatility in the S&P test span, greedy decoding:

    Qwen2.5-1.5B-Instruct   7 of 8 distinct forecasts, sd 0.37, corr 0.65 with GARCH VaR
    Qwen2.5-3B-Instruct     3 of 8 distinct, sd 0.006, every answer about -0.01

The larger model is the worse one. It emits well-formed JSON with a correctly signed
quantile that is a constant near zero, which every validity check in the pipeline
passes. It is kept as a second model precisely because of that: `--model` takes either,
and the degenerate one is the most useful exhibit in the package.

Output matches stage 03 exactly, into precomputed/:
    local_<tag>_<ASSET>_<model-slug>.csv

Usage:
    python src/07_local_llm.py --asset SPX --config series
    python src/07_local_llm.py --asset SPX --model Qwen/Qwen2.5-3B-Instruct --limit 40
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labcommon import ASSETS, LAB, PRECOMP, ROOT, SEED  # noqa: E402

# Reuse stage 03's prompt, system message and parser. Importing rather than copying is
# what makes "same prompt, different model" a claim rather than an aspiration.
_spec = importlib.util.spec_from_file_location("llm03", HERE / "03_llm_lab.py")
llm03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llm03)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 220
BATCH = 8


def device_of():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate(model_id, prompts, dev, batch=BATCH, quiet=False, dtype_name="",
             ckpt=None):
    """Greedy decoding, batched. Returns the raw completion text for each prompt.

    With `ckpt`, every finished batch is appended to disk and a restart picks up where
    it stopped. A 14B model over 500 dates is hours of decoding; a run that keeps
    nothing until the end is a run that cannot be interrupted or moved."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    done = {}
    if ckpt is not None and ckpt.exists():
        prev = pd.read_csv(ckpt, keep_default_na=False)
        done = dict(zip(prev["i"].astype(int), prev["text"].astype(str)))
        print(f"  resuming: {len(done)} completions already on disk", flush=True)
    todo = [i for i in range(len(prompts)) if i not in done]
    if not todo:
        return [done[i] for i in range(len(prompts))]

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, dtype_name) if dtype_name else (
        torch.float16 if dev == "cuda" else torch.float32)
    mod = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()

    t0 = time.time()
    for i in range(0, len(todo), batch):
        idx = todo[i:i + batch]
        chunk = [prompts[j] for j in idx]
        chats = [tok.apply_chat_template(
            [{"role": "system", "content": llm03.SYSTEM},
             {"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False) for p in chunk]
        enc = tok(chats, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = mod.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        batch_txt = [tok.decode(out[j][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True) for j in range(len(chunk))]
        for j, t in zip(idx, batch_txt):
            done[j] = t
        if ckpt is not None:
            pd.DataFrame({"i": idx, "text": batch_txt}).to_csv(
                ckpt, mode="a", header=not ckpt.exists(), index=False)
        if not quiet and (i // batch) % 5 == 0:
            n = i + len(chunk)
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el / 60:.1f}m elapsed, "
                  f"eta {el / n * (len(todo) - n) / 60:.1f}m", flush=True)
    return [done[i] for i in range(len(prompts))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="SPX", choices=list(ASSETS))
    p.add_argument("--config", default="series",
                   choices=["series", "series+state", "dated", "dated+news"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--dtype", default="", choices=["", "float32", "float16", "bfloat16"],
                   help="weight precision; default is fp16 on CUDA, fp32 elsewhere. "
                        "A 14B model needs fp16 to fit comfortably, and the precision "
                        "is written into the output filename so a size curve is never "
                        "assembled from mixed precisions by accident.")
    a = p.parse_args()

    rets = pd.read_csv(LAB / f"returns_{a.asset}.csv", index_col=0, parse_dates=True)
    bench = pd.read_csv(LAB / f"bench_{a.asset}.csv", parse_dates=["date"])
    dates = pd.DatetimeIndex(sorted(bench["date"].unique()))
    if a.limit:
        dates = dates[:a.limit]

    news = {}
    if a.config == "dated+news":
        nf = pd.read_csv(ROOT / "data" / "news" / f"headlines_{a.asset}.csv",
                         parse_dates=["date"])
        news = nf.groupby("date")["headline"].apply(list).to_dict()

    recs = [(d, llm03.build_prompt(a.config, rets.loc[:d, "ret"].iloc[:-1],
                                   a.asset, d, news.get(d))) for d in dates]

    dev = device_of()
    slug = a.model.split("/")[-1]
    tag = a.config.replace("+", "_")
    prec = f"_{a.dtype}" if a.dtype else ""
    suffix = f"_smoke{len(recs)}" if a.limit else ""
    stem = f"local_{tag}_{a.asset}_{slug}{prec}{suffix}"

    print(f"{a.config} / {a.asset} / {a.model} on {dev}: {len(recs)} forecasts")
    texts = generate(a.model, [p for _, p in recs], dev, batch=a.batch,
                     dtype_name=a.dtype, ckpt=PRECOMP / f"_{stem}.csv")
    parsed = [llm03.parse_reply(t) for t in texts]

    out = llm03._rows(recs, parsed)
    out["n_headlines"] = [len(news.get(d, [])) for d, _ in recs]

    path = PRECOMP / f"{stem}.csv"
    out.to_csv(path, index=False)

    n_bad = int((~out["raw_ok"]).sum())
    n_sign = int((~out["sign_ok"] & out["raw_ok"]).sum())
    n_ord = int((~out["order_ok"] & out["raw_ok"]).sum())
    ok = out["sign_ok"] & out["order_ok"]
    # Distinct forecasts is the diagnostic a validity check cannot supply: a model can
    # pass parsing, sign and ordering on every row and still be emitting one number.
    dist = int(out.loc[ok, "var_01"].round(2).nunique())
    print(f"\n  parsed {len(out) - n_bad}/{len(out)}; {n_sign} non-negative; "
          f"{n_ord} inverted")
    print(f"  distinct VaR(1%) values: {dist} of {int(ok.sum())} usable "
          f"({100 * dist / max(int(ok.sum()), 1):.0f}%)")
    print(f"  mean VaR(1%) {out.loc[ok, 'var_01'].mean():.2f}   "
          f"sd {out.loc[ok, 'var_01'].std():.2f}")
    print(f"  -> {path.name}")


if __name__ == "__main__":
    main()
