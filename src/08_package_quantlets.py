"""Stage 08 — package each pipeline stage as a Quantlet.

Follows the convention already used by `aida-risk/quantlets/`: one directory per
Quantlet, carrying the script, its dependencies, a Metainfo.txt and a README, so the
folder runs where it sits and can be pushed to the QuantLet organisation unchanged.

Every empirical frame in the deck carries a `\\quantlet{}` mark pointing at one of
these names, so a reader who wants the number behind a slide has one place to go.

Usage:  python src/08_package_quantlets.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labcommon import ALPHA, ROOT, TEST_N  # noqa: E402

QL = ROOT / "quantlets"
SRC = ROOT / "src"
NB = ROOT / "notebook"
LOGO = ROOT.parent / "beamer-deck" / "ql_logo.png"

AUTHOR = "Daniel Traian Pele"
PUBLISHED = "AIDA Summer School 2026 - AI in Financial Risk Modelling (August course)"
SUBMITTED = "2026-07-31"
KEYWORDS = ("value-at-risk, backtesting, kupiec, christoffersen, diebold-mariano, "
            "foundation-model, large-language-model, zero-shot, risk-management, "
            "financial-econometrics")

# name -> (scripts, extra files, description, datafile, output)
PACKAGES = {
    "AIDA_aug_data": (
        ["01_lab_data.py"], [],
        "Carves the laboratory dataset out of the certified aida-risk run: five assets, "
        f"a common {TEST_N}-day test span, and the classical VaR forecasts that span "
        "traces to. Nothing is downloaded and nothing is simulated.",
        "aida-risk/data/raw/*.csv", "data/lab/returns_*.csv, bench_*.csv"),
    "AIDA_aug_chronos": (
        ["02_chronos_lab.py"], [],
        "Chronos zero-shot VaR through both heads. The quantile head is asked for levels "
        "below its trained grid and returns the clamped value, which the script records "
        "rather than repairs; the sampling head supplies the genuine tail quantile.",
        "data/lab/returns_*.csv", "precomputed/chronos_{bolt,t5}_*.csv"),
    "AIDA_aug_llm": (
        ["03_llm_lab.py"], [],
        "LLM-VaR through a commercial endpoint: the return series serialised as text, "
        "the tail quantile asked for in words, four information sets, and a parser that "
        "counts the replies that fail rather than dropping them.",
        "data/lab/returns_*.csv, data/news/headlines_*.csv",
        "precomputed/llm_*.csv"),
    "AIDA_aug_local_llm": (
        ["07_local_llm.py", "03_llm_lab.py"], [],
        "The same experiment with open weights, so it can be reproduced without a "
        "commercial key. Imports the prompt and parser from the commercial stage, so the "
        "two legs differ in exactly one thing: which model reads the prompt.",
        "data/lab/returns_*.csv", "precomputed/local_*.csv"),
    "AIDA_aug_news": (
        ["06_news.py"], [],
        "Real headlines from EODHD aligned to forecast dates under an audited as-of "
        "rule: each asset's own market close, consecutive cutoffs tiling the calendar, "
        "and every publication timestamp kept so the rule can be checked afterwards.",
        "EODHD news API", "data/news/headlines_*.csv, coverage_*.csv"),
    "AIDA_aug_backtest": (
        ["04_slide_figures.py", "05_verify_slide_claims.py"], ["aidalab.py"],
        f"Backtesting and scoring at alpha = {ALPHA:.0%}: Kupiec unconditional "
        "coverage, Christoffersen independence, the pinball loss, Diebold-Mariano with "
        "a HAC standard error, model disagreement, and a checker that re-derives every "
        "number printed on a slide.",
        "precomputed/*.csv", "figures/slides/*.pdf, precomputed/slide_facts_*.csv"),
    "AIDA_aug_lab": (
        [], ["aidalab.py", "build_notebook.py", "AIDA_Risk_Lab.ipynb"],
        "The student notebook and its helper module: loaders, historical simulation, "
        "the pinball loss, the coverage tests, Diebold-Mariano and the plots, depending "
        "only on numpy, pandas, scipy and matplotlib so it runs on a bare Colab session.",
        "aida_lab_data.zip", "figures produced in-notebook"),
}


def metainfo(name, desc, datafile, output):
    return (f"Name of QuantLet: {name}\n\n"
            f"Published in: {PUBLISHED}\n\n"
            f"Description: {desc}\n\n"
            f"Keywords: {KEYWORDS}\n\n"
            f"Author: {AUTHOR}\n\n"
            f"Submitted: {SUBMITTED}\n\n"
            f"Datafile: {datafile}\n\n"
            f"Output: {output}\n")


def main():
    if QL.exists():
        shutil.rmtree(QL)
    QL.mkdir(parents=True)

    for name, (scripts, extras, desc, datafile, output) in PACKAGES.items():
        d = QL / name
        d.mkdir()
        for f in scripts:
            shutil.copy2(SRC / f, d / f)
        for f in extras:
            src = NB / f if (NB / f).exists() else SRC / f
            shutil.copy2(src, d / f)
        # labcommon is the only shared dependency of the src stages
        if scripts:
            shutil.copy2(SRC / "labcommon.py", d / "labcommon.py")
        if LOGO.exists():
            shutil.copy2(LOGO, d / "ql_logo.png")
        (d / "Metainfo.txt").write_text(metainfo(name, desc, datafile, output))
        (d / "README.md").write_text(
            f"# {name}\n\n{desc}\n\n"
            f"- **Datafile:** {datafile}\n- **Output:** {output}\n"
            f"- **Part of:** {PUBLISHED}\n\n"
            f"Reproduce with `../../run_all.sh` from the package root; this folder is "
            f"the extracted, citable form of one stage.\n")
        n = len(list(d.glob("*")))
        print(f"  {name:<22} {n} files")

    (QL / "README.md").write_text(
        f"# Quantlets — {PUBLISHED}\n\n"
        "One directory per pipeline stage. Each carries its script, its dependencies, a\n"
        "Metainfo.txt and a README, and is the citable form of the stage the deck's\n"
        "`\\quantlet{}` marks point at.\n\n"
        + "".join(f"- **{k}** — {v[2].split('.')[0]}.\n" for k, v in PACKAGES.items()))
    print(f"\n  -> {QL}  ({len(PACKAGES)} quantlets)")


if __name__ == "__main__":
    main()
