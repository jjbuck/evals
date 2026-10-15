"""Assemble the page: the template, its figures, each example's results text, and a live explorer panel per example
with that study's record embedded.

    npm --prefix explorer run build:embed
    python docs/optimization/page/assemble.py --study poem=studies/poem-length-20261015-051555 [--study ...] [-o OUT]

A study given no record shows as still running. The output is one self-contained HTML file.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PAGE = Path(__file__).resolve().parent
ROOT = PAGE.parents[2]
EMBED = ROOT / "explorer" / "dist-embed" / "explorer-embed.js"
SLOTS = ("poem", "document-reader", "tau2")
PENDING = '<p class="pending">This study is still running; its results appear here when it finishes.</p>'
STYLE = (
    ".pending { color: var(--muted, #898781); font-style: italic; }"
    " .viewer > .pending { margin: 0; padding: 2rem; font-style: normal; }"
)
BRANCH = (
    'The code is on the <a href="https://github.com/jjbuck/evals/tree/feat/optimization">'
    "<code>feat/optimization</code> branch</a>."
)


def figure(name: str) -> str:
    svg = (PAGE / f"fig-{name}.svg").read_text()
    return re.sub(r"^<\?xml[^>]*>\s*", "", svg)


def studies_script(records: dict[str, Path]) -> str:
    """The embedded studies, as ``var STUDIES = {...};`` (by record directory name)."""
    if not records:
        return "var STUDIES = {};"
    out = subprocess.run(
        ["node", str(ROOT / "explorer" / "scripts" / "embed-data.mjs"), "--var", "STUDIES", *map(str, records.values())],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def assemble(records: dict[str, Path]) -> str:
    html = (PAGE / "template.html").read_text()
    html = html.replace("</head>", f"<style>{STYLE}</style>\n</head>", 1)
    for name in ("loop", "blocks", "bayesbench"):
        html = html.replace(f"<!-- FIGURE:{name} -->", figure(name))
    for slot in SLOTS:
        text = PAGE / "results" / f"{slot}.html"
        body = text.read_text() if slot in records and text.exists() else PENDING
        html = html.replace(f"<!-- RESULTS:{slot} -->", body)
        if slot in records:  # the caption bar names the record
            html = html.replace(f"<code>{slot}</code>", f"<code>{records[slot].name}</code>", 1)
        else:
            html = html.replace(
                f'data-study="{slot}" data-page="progress"></div>',
                f'data-study="{slot}" data-page="progress">{PENDING}</div>',
            )
    html = re.sub(r'<p><span class="todo">TODO: replace the local path[^<]*</span></p>', f"<p>{BRANCH}</p>", html)
    ids = {slot: records[slot].name for slot in records}
    mount = (
        f"var SLOTS = {json.dumps(ids)};\n"
        'document.querySelectorAll(".viewer[data-study]").forEach(function (el) {\n'
        "  var id = SLOTS[el.dataset.study];\n"
        "  if (!id || !STUDIES[id]) return;\n"
        '  el.setAttribute("data-mounted", "");\n'
        "  StrandsExplorer.mount(el, STUDIES[id], { initialPage: el.dataset.page || \"progress\" });\n"
        "});"
    )
    scripts = "\n".join(
        [
            f"<script>{EMBED.read_text()}</script>",
            f"<script>{studies_script(records)}</script>",
            f"<script>\n{mount}\n</script>",
        ]
    )
    return html.replace("<!-- SCRIPTS -->", scripts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", action="append", default=[], help="slot=record directory (slots: %(choices)s)")
    parser.add_argument("-o", "--out", default=str(PAGE / "dist" / "strands-evals-optimization.html"))
    args = parser.parse_args()
    records = {}
    for item in args.study:
        slot, path = item.split("=", 1)
        if slot not in SLOTS:
            raise SystemExit(f"unknown slot {slot!r}; the slots are {SLOTS}")
        records[slot] = (ROOT / path).resolve()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(assemble(records))
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")  # noqa: T201
