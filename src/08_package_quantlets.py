"""Stage 08 — package each pipeline stage as a Quantlet.

Follows the convention already used by `aida-risk/quantlets/`: one directory per
Quantlet, carrying the script, its dependencies, a Metainfo.txt and a README, so the
folder runs where it sits and can be pushed to the QuantLet organisation unchanged.

Every empirical frame in the deck carries a `\\quantlet{}` mark pointing at one of
these names, so a reader who wants the number behind a slide has one place to go.

Each folder also ships what the stage produced, so the Quantlet can be read without
running anything:

  <name>.ipynb   a notebook that shows the figures and loads the data files below
  *.png          the slide figures whose frames carry this Quantlet's mark
  output/        the CSV, .tex and prompt files the stage writes

The figure list is derived from the deck rather than declared here: a frame's
`\\quantlet{}` mark and its `\\includegraphics` are read together, so a figure that
moves to another frame follows its Quantlet without anyone remembering to update a list.

Usage:  python src/08_package_quantlets.py
"""

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labcommon import ALPHA, ROOT, TEST_N  # noqa: E402

QL = ROOT / "quantlets"
SRC = ROOT / "src"
NB = ROOT / "notebook"
DECK = ROOT / "slides" / "ai_risk_course_2026-08.tex"
FIGDIRS = [ROOT / "figures" / "slides", ROOT / "figures"]
LOGO = ROOT.parent / "beamer-deck" / "ql_logo.png"

AUTHOR = "Daniel Traian Pele"
PUBLISHED = "AIDA Summer School 2026 - AI in Financial Risk Modelling (August course)"
SUBMITTED = "2026-07-31"
KEYWORDS = ("value-at-risk, backtesting, kupiec, christoffersen, diebold-mariano, "
            "foundation-model, large-language-model, zero-shot, risk-management, "
            "financial-econometrics")

# name -> (scripts, extra files, description, datafile, output, output globs)
# The globs are relative to the package root and select the files that are copied into
# <name>/output/. They are deliberately narrower than the Output: line, which describes
# the stage's full output: intermediate checkpoints (leading underscore) stay behind.
PACKAGES = {
    "AIDA_aug_data": (
        ["01_lab_data.py"], [],
        "Carves the laboratory dataset out of the certified aida-risk run: five assets, "
        f"a common {TEST_N}-day test span, and the classical VaR forecasts that span "
        "traces to. Everything is derived from files already on disk.",
        "aida-risk/data/raw/*.csv", "data/lab/returns_*.csv, bench_*.csv",
        ["data/lab/returns_*.csv", "data/lab/bench_*.csv"]),
    "AIDA_aug_chronos": (
        ["02_chronos_lab.py"], [],
        "Chronos-T5 zero-shot VaR: the last 512 closing prices are handed to the "
        "pretrained model, 500 paths are sampled for the next day, and the tail "
        "quantile is read off the draws as an order statistic.",
        "data/lab/returns_*.csv", "precomputed/chronos_t5_*.csv",
        ["precomputed/chronos_t5_*.csv"]),
    "AIDA_aug_llm": (
        ["03_llm_lab.py"], [],
        "LLM-VaR through a commercial endpoint: the return series serialised as text, "
        "the tail quantile asked for in words, four information sets, and a parser that "
        "counts the replies that fail rather than dropping them.",
        "data/lab/returns_*.csv, data/news/headlines_*.csv",
        "precomputed/llm_*.csv",
        ["precomputed/llm_*.csv", "tables/llm_system.txt", "tables/llm_user.txt",
         "tables/llm_schema.txt"]),
    "AIDA_aug_local_llm": (
        ["07_local_llm.py", "03_llm_lab.py", "09_llmtime.py"], [],
        "The same experiment with open weights, so it can be reproduced without a "
        "commercial key. Imports the prompt and parser from the commercial stage, so the "
        "two legs differ in exactly one thing: which model reads the prompt. Also runs "
        "the LLMTime sampling route, where the quantile is an order statistic of 1000 "
        "generated continuations.",
        "data/lab/returns_*.csv", "precomputed/local_*.csv, llmtime_*.csv",
        ["precomputed/local_*.csv", "precomputed/llmtime_*.csv",
         "tables/llmtime_prompt.txt", "tables/published_sampling_prompt.txt"]),
    "AIDA_aug_news": (
        ["06_news.py"], [],
        "Real headlines from EODHD aligned to forecast dates under an audited as-of "
        "rule: each asset's own market close, consecutive cutoffs tiling the calendar, "
        "and every publication timestamp kept so the rule can be checked afterwards.",
        "EODHD news API", "data/news/headlines_*.csv, coverage_*.csv",
        ["data/news/headlines_*.csv", "data/news/coverage_*.csv"]),
    "AIDA_aug_backtest": (
        ["04_slide_figures.py", "05_verify_slide_claims.py"], ["aidalab.py"],
        f"Backtesting and scoring at alpha = {ALPHA:.0%}: Kupiec unconditional "
        "coverage, Christoffersen independence, the pinball loss, Diebold-Mariano with "
        "a HAC standard error, model disagreement, and a checker that re-derives every "
        "number printed on a slide.",
        "precomputed/*.csv", "figures/slides/*.pdf, precomputed/slide_facts_*.csv",
        ["tables/*.tex", "precomputed/slide_facts_*.csv", "precomputed/lab_*.csv"]),
    "AIDA_aug_lab": (
        [], ["aidalab.py", "build_notebook.py", "AIDA_Risk_Lab.ipynb"],
        "The student notebook and its helper module: loaders, historical simulation, "
        "the pinball loss, the coverage tests, Diebold-Mariano and the plots, depending "
        "only on numpy, pandas, scipy and matplotlib so it runs on a bare Colab session.",
        "aida_lab_data.zip", "figures produced in-notebook", []),
}


def metainfo(name, desc, datafile, output):
    """Values are single-quoted, which is the form the Quantlet indexer reads and the
    form the other repositories in the organisation use."""
    return (f"Name of QuantLet: '{name}'\n\n"
            f"Published in: '{PUBLISHED}'\n\n"
            f"Description: '{desc}'\n\n"
            f"Keywords: '{KEYWORDS}'\n\n"
            f"Author: '{AUTHOR}'\n\n"
            f"Submitted: '{SUBMITTED}'\n\n"
            f"Datafile: '{datafile}'\n\n"
            f"Output: '{output}'\n")


def deck_figures():
    """name -> [figure basenames], read off the deck.

    A frame carries at most one `\\quantlet{}` mark, so every graphic in that frame
    belongs to that stage. Frames without a mark show a mechanism rather than a result
    and have no Quantlet to attach to.
    """
    src = DECK.read_text()
    out = {}
    for m in re.finditer(r"\\begin\{frame\}.*?\\end\{frame\}", src, re.S):
        body = m.group(0)
        mark = re.search(r"\\quantlet\{[^}]*\}\{[^}]*/(AIDA_aug_[a-z_]+)\}", body)
        if not mark:
            continue
        figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
        out.setdefault(mark.group(1), []).extend(figs)
    return {k: sorted(set(v)) for k, v in out.items()}


def copy_figures(name, dest, figs):
    """PNG at the package root, PDF beside it: the indexer renders the raster, the
    vector is what a paper would include."""
    shipped = []
    for g in figs:
        for ext in (".png", ".pdf"):
            hit = next((d / (g + ext) for d in FIGDIRS if (d / (g + ext)).exists()), None)
            if hit:
                shutil.copy2(hit, dest / hit.name)
                if ext == ".png":
                    shipped.append(hit.name)
    return shipped


def copy_outputs(dest, globs):
    """The data the stage writes, under output/, flattened."""
    if not globs:
        return []
    outdir = dest / "output"
    shipped = []
    for pattern in globs:
        for f in sorted(ROOT.glob(pattern)):
            if f.name.startswith("_"):
                continue                     # resumable checkpoints, not results
            outdir.mkdir(exist_ok=True)
            shutil.copy2(f, outdir / f.name)
            shipped.append(f.name)
    return shipped


def notebook(name, desc, scripts, figs, outs, datafile, output):
    """A notebook that reads the shipped Quantlet without recomputing it.

    It runs top to bottom on a bare Python with pandas and matplotlib, because the point
    is to see what the stage produced. Regenerating it needs the full course repository,
    so that command is printed rather than executed.
    """
    cells = []

    def md(t):
        cells.append({"cell_type": "markdown", "metadata": {},
                      "id": f"c{len(cells):02d}",
                      "source": t.strip("\n").splitlines(keepends=True)})

    def code(t):
        cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                      "execution_count": None, "id": f"c{len(cells):02d}",
                      "source": t.strip("\n").splitlines(keepends=True)})

    md(f"""# {name}

{desc}

**Published in:** {PUBLISHED}
**Author:** {AUTHOR}
**Datafile:** `{datafile}`
**Output:** `{output}`

This notebook shows what the stage produced. The figures and data files are shipped
beside it, so every cell below runs offline.
""")

    code("""
from pathlib import Path
import pandas as pd

HERE = Path.cwd()
print(sorted(p.name for p in HERE.iterdir() if p.is_file()))
""")

    if figs:
        md("## Figures\n\nThe slide figures whose frames carry this Quantlet's mark.")
        code("from IPython.display import Image, display\n\n"
             "FIGURES = " + json.dumps(figs) + "\n"
             "for f in FIGURES:\n"
             "    print(f)\n"
             "    display(Image(filename=str(HERE / f)))")

    if outs:
        csvs = [f for f in outs if f.endswith(".csv")]
        texts = [f for f in outs if not f.endswith(".csv")]
        md(f"## Output\n\n`output/` holds {len(outs)} file(s) written by this stage.")
        if csvs:
            code("OUTPUT = HERE / 'output'\n"
                 "CSVS = " + json.dumps(csvs) + "\n"
                 "summary = []\n"
                 "for f in CSVS:\n"
                 "    df = pd.read_csv(OUTPUT / f)\n"
                 "    summary.append({'file': f, 'rows': len(df),\n"
                 "                    'columns': ', '.join(df.columns[:6])})\n"
                 "pd.DataFrame(summary)")
            code("# the first of them, in full\n"
                 "pd.read_csv(OUTPUT / CSVS[0]).head(10)")
        if texts:
            code("TEXTS = " + json.dumps(texts) + "\n"
                 "for f in TEXTS:\n"
                 "    print('=' * 70)\n"
                 "    print(f)\n"
                 "    print('=' * 70)\n"
                 "    print((HERE / 'output' / f).read_text())")

    md("## Reproducing it\n\n"
       "The scripts are shipped beside this notebook, and they expect the full course\n"
       "repository around them, which is where the input data lives:\n\n"
       "```\n"
       "git clone https://github.com/QuantLet/AIDA2026.git\n"
       "cd AIDA2026\n"
       + ("".join(f"python src/{s}\n" for s in scripts) if scripts
          else "python notebook/build_notebook.py\n")
       + "```\n")

    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    if QL.exists():
        shutil.rmtree(QL)
    QL.mkdir(parents=True)

    figmap = deck_figures()
    unknown = set(figmap) - set(PACKAGES)
    if unknown:
        print(f"  WARNING: the deck marks {sorted(unknown)}, which is not packaged here")

    for name, (scripts, extras, desc, datafile, output, globs) in PACKAGES.items():
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

        figs = copy_figures(name, d, figmap.get(name, []))
        outs = copy_outputs(d, globs)

        # AIDA_aug_lab already is a notebook; a second one would only wrap it
        if name != "AIDA_aug_lab":
            (d / f"{name}.ipynb").write_text(json.dumps(
                notebook(name, desc, scripts, figs, outs, datafile, output),
                indent=1) + "\n")

        (d / "Metainfo.txt").write_text(metainfo(name, desc, datafile, output))
        (d / "README.md").write_text(
            f"# {name}\n\n{desc}\n\n"
            f"- **Datafile:** {datafile}\n- **Output:** {output}\n"
            f"- **Part of:** {PUBLISHED}\n\n"
            + (f"`{name}.ipynb` shows the result without running anything: "
               f"{len(figs)} figure(s) at the folder root and {len(outs)} data file(s) "
               f"under `output/`.\n\n" if name != "AIDA_aug_lab" else "")
            + f"Reproduce with `../../run_all.sh` from the package root; this folder is "
            f"the extracted, citable form of one stage.\n")
        print(f"  {name:<22} {len(list(d.rglob('*'))):>3} files  "
              f"({len(figs)} fig, {len(outs)} out)")

    # The repository itself carries a Metainfo too, the way the other repositories in
    # the organisation do: the indexer reads the root file for the repo landing page,
    # and hand-maintaining it would let it drift from the packages below.
    (ROOT / "Metainfo.txt").write_text(metainfo(
        "AIDA2026",
        "Lecture deck and two-hour coding laboratory asking whether AI produces valid "
        "1% Value at Risk forecasts. Five assets, 500 out-of-sample days and one frozen "
        "protocol compare historical simulation, GARCH(1,1)-t, a neural volatility "
        "model, Chronos-T5, four configurations of a commercial language model and two "
        "ways of reading a quantile out of an open-weights model. Every figure and "
        "table is generated by a script, and a checker re-derives every number printed "
        "on a slide from the data. "
        f"{len(PACKAGES)} Quantlets: " + ", ".join(PACKAGES) + ".",
        "data/lab/*.csv, data/news/headlines_*.csv, precomputed/*.csv",
        "slides/ai_risk_course_2026-08.pdf, figures/slides/*.pdf, tables/*.tex"))
    print(f"  -> {ROOT / 'Metainfo.txt'}")

    (QL / "README.md").write_text(
        f"# Quantlets — {PUBLISHED}\n\n"
        "One directory per pipeline stage. Each carries its script, its dependencies, a\n"
        "Metainfo.txt, a README, a notebook that displays what the stage produced, the\n"
        "slide figures it is cited on, and the data files it writes under `output/`.\n\n"
        + "".join(f"- **{k}** — {v[2].split('.')[0]}.\n" for k, v in PACKAGES.items()))
    print(f"\n  -> {QL}  ({len(PACKAGES)} quantlets)")


if __name__ == "__main__":
    main()
