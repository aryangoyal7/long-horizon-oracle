"""Regenerate the HTML table bodies in index.html from the paper draft.

Numbers are parsed out of the draft's LaTeX tabulars, never retyped, and
written between <!-- BEGIN:<id> --> / <!-- END:<id> --> markers in index.html.
Headers and captions live in index.html.

  table-ol-shares-window  <- \\label{tab:ol-shares-window}  (matched-window shares)
  table-pred-ol           <- \\label{tab:pred-ol-app}        (open-loop predictor;
                             asserted identical to the main-text tab:pred-ol rows)

usage: python build_tables.py <paper_src_dir_or_zip> [index.html]
"""
import html
import os
import re
import sys
import tempfile
import zipfile

src = sys.argv[1]
index = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "index.html")
if os.path.isfile(src):
    tmp = tempfile.mkdtemp()
    zipfile.ZipFile(src).extractall(tmp)
    src = tmp
paper = open(os.path.join(src, "iclr2027_conference.tex")).read()
# drop commented lines so commented-out copies of a table are never parsed
paper = "\n".join(l for l in paper.splitlines() if not l.lstrip().startswith("%"))


def tabular_before(label):
    at = paper.index(f"\\label{{{label}}}")
    b = paper.rindex("\\begin{tabular}", 0, at)
    e = paper.index("\\end{tabular}", b)
    body = paper[b:e]
    return body[body.index("\\midrule"):]


def cell(s):
    s = s.strip()
    if s in ("--", "-"):
        return "&ndash;"
    s = s.replace("\\_", "_").replace("{,}", ",")
    s = s.replace("$^{\\dagger}$", "<sup>&dagger;</sup>").replace("$^\\dagger$", "<sup>&dagger;</sup>")
    s = re.sub(r"\$-\$", "&minus;", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"<strong>\1</strong>", s)
    assert "\\" not in s and "$" not in s, f"unconverted LaTeX in cell: {s!r}"
    return s


def rows(body):
    """[('group', name) | ('row', [cells])] in table order."""
    out = []
    for raw in body.split("\\\\"):
        line = raw.replace("\\midrule", "").replace("\\bottomrule", "").strip()
        if not line:
            continue
        m = re.match(r"\\multicolumn\{\d+\}\{l\}\{\\emph\{([^}]*)\}(.*)\}$", line)
        if m:
            out.append(("group", (m.group(1) + m.group(2)).replace("---", "&mdash;").replace("\u2014", "&mdash;")))
            continue
        out.append(("row", [cell(c) for c in line.split("&")]))
    return out


def tbody(parsed, ncols, first_cols_left=2):
    h = []
    for kind, v in parsed:
        if kind == "group":
            h.append(f'<tr class="group-row"><th colspan="{ncols}">{v}</th></tr>')
            continue
        assert len(v) == ncols, (len(v), v)
        tds = []
        for i, c in enumerate(v):
            cls = ' class="lab"' if i < first_cols_left else ""
            tds.append(f"<td{cls}>{c}</td>")
        h.append("<tr>" + "".join(tds) + "</tr>")
    return "\n".join(h)


def splice(text, key, content):
    b, e = f"<!-- BEGIN:{key} -->", f"<!-- END:{key} -->"
    i, j = text.index(b) + len(b), text.index(e)
    return text[:i] + "\n" + content + "\n" + text[j:]


shares = rows(tabular_before("tab:ol-shares-window"))
pred = rows(tabular_before("tab:pred-ol-app"))
pred_main = rows(tabular_before("tab:pred-ol"))
assert pred == pred_main, "tab:pred-ol and tab:pred-ol-app disagree; resolve in the draft"
for kind, v in pred:
    if kind == "row":                     # draft labels the action variant A1
        v[1] = {"P": "P", "A1": "A1"}[v[1]]

page = open(index).read()
page = splice(page, "table-ol-shares-window", tbody(shares, 8))
page = splice(page, "table-pred-ol", tbody(pred, 7))
assert "LIBERO" not in page and "lib-" not in page
open(index, "w").write(page)
print(f"tables: {sum(k == 'row' for k, _ in shares)} share rows, "
      f"{sum(k == 'row' for k, _ in pred)} predictor rows -> {os.path.basename(index)}")
