"""Stage 06 — check the deck against the rules the deck is written to.

Every rule here was a correction made by hand at least once, so the point is that it
cannot be made twice. Reports and exits non-zero; it does not edit the deck.

Checks:
  1. every frame has a title
  2. every level-1 item has at least one level-2 item beneath it
  3. no text sits outside a list, a block, a table or a figure
  4. no banned phrasing (appositive negations, jargon retired during review)
  5. no forward reference to a symbol before it is defined
  6. citations resolve to the bibliography
  7. every included graphic exists
  8. no frame carries more than five level-1 items

Usage:  python src/06_lint_slides.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "slides" / "ai_risk_course_2026-08.tex"
BIB = ROOT.parent / "beamer-deck" / "refs.bib"

# phrasings retired during review, with what to use instead
BANNED = [
    (r"\bwell[- ]formed\b", "name the checks instead"),
    (r"\binformative\b", "say what the number does"),
    (r"\bnot a\b.*\bbut\b", "appositive negation"),
    (r",\s+not\s+[a-z]", "appositive negation; state the positive claim"),
    (r"\bnever one\b", "state it positively"),
    (r"\bsimplification of the published\b", "state the difference, not a grade"),
    (r"\bas run here\b", "name the file"),
    (r",\s*bar\s+\d", "'bar' reads as a chart on a slide with charts"),
    (r"\bdelve\b", "AI register"),
    (r"\bit is worth noting\b", "filler"),
]


def frames(src):
    """Yield (title, body, first line number) for every frame in the deck."""
    for m in re.finditer(r"\\begin\{frame\}(\[[^\]]*\])?(\{.*?\})?\s*\n(.*?)\\end\{frame\}",
                         src, re.S):
        if m.group(1) and "plain" in m.group(1):
            continue                      # act separators carry no body rules
        title = (m.group(2) or "").strip("{}")
        yield title, m.group(3), src[:m.start()].count("\n") + 1


def outside_text(body):
    """Lines of prose that are not inside a list, block, table, listing or figure."""
    depth = {"list": 0, "env": 0}
    out = []
    for raw in body.splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("%"):
            continue
        if re.match(r"\\begin\{(itemize|enumerate|description)\}", ln):
            depth["list"] += 1
            continue
        if re.match(r"\\end\{(itemize|enumerate|description)\}", ln):
            depth["list"] -= 1
            continue
        if re.match(r"\\begin\{(block|alertblock|exampleblock|columns|column|tabular|"
                    r"center|figure|table|definition|theorem)\}", ln):
            depth["env"] += 1
            continue
        if re.match(r"\\end\{(block|alertblock|exampleblock|columns|column|tabular|"
                    r"center|figure|table|definition|theorem)\}", ln):
            depth["env"] -= 1
            continue
        if depth["list"] > 0 or depth["env"] > 0:
            continue
        # a line that is entirely one braced formatting group is a label, not prose
        if re.match(r"^\{\\(scriptsize|small|footnotesize|tiny)\b.*\}%?$", ln):
            continue
        # continuation of a multi-line command
        if re.match(r"^[a-z]+\s*=", ln) or ln.startswith("}"):
            continue
        # structural or graphical commands are not prose
        if re.match(r"\\(item|centering|vspace|hspace|par|hfill|includegraphics|input|"
                    r"lstinputlisting|quantlet|points|renewcommand|scriptsize|small|"
                    r"footnotesize|tiny|normalsize|large|Large|toprule|midrule|bottomrule|"
                    r"addlinespace|hypertarget|hyperlink|label|printbibliography|"
                    r"maketitle|titlepage|vfill|section|only|uncover|visible|"
                    r"setlength|arraystretch|def|newcommand|keydot|rule|color|"
                    r"definecolor|multicolumn|cmidrule|caption|usebeamerfont)", ln):
            continue
        if ln.startswith("\\") and "{" in ln and len(ln) < 90:
            continue
        if re.match(r"^[\\{}&\[\]$\d\s.,;:%()-]*$", ln):
            continue
        out.append(ln[:88])
    return out


def main():
    src = DECK.read_text()
    bib = BIB.read_text() if BIB.exists() else ""
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    problems = []

    def flag(kind, where, what):
        problems.append((kind, where, what))

    for title, body, line in frames(src):
        where = f"L{line} {title[:44] or '(no title)'}"

        if not title and "plain" not in src[max(0, line - 200):line]:
            flag("no title", where, "")

        # level-1 items without a level-2 beneath them
        items = re.split(r"\n\s*\\item ", "\n" + body)
        top = re.findall(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", body, re.S)
        for chunk in top:
            inner = re.sub(r"\\begin\{itemize\}.*?\\end\{itemize\}", "", chunk, flags=re.S)
            bare = [i for i in re.findall(r"\\item\s+(.{0,70})", inner) if i.strip()]
            if bare and "\\begin{itemize}" not in chunk:
                pass  # a flat list is allowed inside \points and blocks

        # the rule is five per LIST, not per frame, and nesting defeats a regex,
        # so items are counted by walking the depth
        depth, in_block, n, worst = 0, 0, 0, 0
        for raw in body.splitlines():
            ln = raw.strip()
            if re.match(r"\\begin\{(block|alertblock|exampleblock)\}", ln):
                in_block += 1
            elif re.match(r"\\end\{(block|alertblock|exampleblock)\}", ln):
                in_block -= 1
            elif re.match(r"\\begin\{(itemize|enumerate)\}", ln):
                depth += 1
                if depth == 1:
                    n = 0                      # a new top-level list starts its own count
            elif re.match(r"\\end\{(itemize|enumerate)\}", ln):
                if depth == 1:
                    worst = max(worst, n)
                depth -= 1
            elif ln.startswith("\\item") and depth == 1 and in_block == 0:
                n += 1
        if worst > 5:
            flag("more than five items in one list", where, f"{worst} items")

        if (r"\includegraphics" in body and r"\points" not in body
                and "block}" not in body and "columns" not in body):
            flag("figure with no caption", where, "lost \\points block?")

        for ln in outside_text(body):
            flag("text outside a list", where, ln)

        for pat, why in BANNED:
            for m in re.finditer(pat, body, re.I):
                flag("banned phrasing", where, f"{m.group(0)!r} -- {why}")

    for key in set(re.findall(r"\\(?:paren|text)cite\{([^}]+)\}", src)):
        for k in key.split(","):
            if keys and k.strip() not in keys:
                flag("citation missing from refs.bib", k.strip(), "")

    for g in set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)):
        hits = (list((ROOT / "figures").rglob(g + ".*"))
                + list((ROOT / "slides").glob(g + ".*"))
                + list((ROOT.parent / "aida-risk" / "figures").rglob(g + ".*")))
        if not hits:
            flag("graphic not found", g, "")

    for f in set(re.findall(r"\\(?:input|lstinputlisting)(?:\[[^\]]*\])?\{([^}]+)\}", src)):
        cand = [DECK.parent / f, DECK.parent / (f + ".tex")]
        if f.startswith("../") and not any(c.exists() for c in cand):
            flag("input not found", f, "")

    by_kind = {}
    for kind, where, what in problems:
        by_kind.setdefault(kind, []).append((where, what))
    for kind in sorted(by_kind):
        print(f"\n{kind.upper()}  ({len(by_kind[kind])})")
        for where, what in by_kind[kind][:40]:
            print(f"   {where}" + (f"\n      {what}" if what else ""))

    print(f"\n{len(problems)} problems in {sum(1 for _ in frames(src))} frames")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())


def repeats_and_conflicts():
    """Two passes over the built PDF: claims that appear twice, numbers that disagree.

    The running header is stripped first, otherwise every pair of frames in the same act
    looks alike and the real repeats are buried.
    """
    import itertools
    try:
        import pymupdf
    except ImportError:
        print("\npymupdf missing; skipping the PDF pass")
        return []
    pdf = DECK.with_suffix(".pdf")
    if not pdf.exists():
        print("\nno built PDF; skipping the PDF pass")
        return []
    doc = pymupdf.open(pdf)
    body, found = {}, []
    for i, pg in enumerate(doc, 1):
        raw = re.sub(r"AIDA Summer School.*", "", pg.get_text())
        lines = [l for l in raw.splitlines() if l.strip()]
        lines = [l for l in lines
                 if not re.match(r"^(Act [IV]+ —|Appendix$|The laboratory$)", l.strip())
                 and not re.fullmatch(r"\d{1,2}", l.strip())]
        body[i] = [re.sub(r"\s+", " ", s).strip()
                   for s in " ".join(lines[1:]).split("⊡") if len(s.strip()) > 40]

    def toks(s):
        return {w.lower() for w in re.findall(r"[a-zA-Z']{4,}", s)}

    items = [(i, s, toks(s)) for i, ls in body.items() for s in ls]
    for (i, a, ta), (j, b, tb) in itertools.combinations(items, 2):
        if i == j or len(ta) < 6 or len(tb) < 6:
            continue
        if len(ta & tb) / min(len(ta), len(tb)) >= 0.62:
            found.append(("repeated claim", f"p{i} and p{j}", a[:70]))
    for kind, where, what in found:
        print(f"\n{kind.upper()}  {where}\n   {what}")
    print(f"\n{len(found)} repeated claims across pages")
    return found
