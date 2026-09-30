"""Derive the site's TikZ sources from the paper draft.

The site shows open-loop results only, so three of the four figures lose their
closed-loop part. Everything else is copied verbatim, and each wrapper uses the
paper's own colour definitions (read from the draft preamble, not retyped).

  fig_method  <- method.tex (or figs/fig_method.tex): panel (b), the
                 closed-loop probe and its estimator, is removed and panel (a)
                 is centred over (c) and (d).
  fig_hist    <- figs/fig_hist.tex: the closed-loop row is removed and the
                 legend moved up by one row pitch (1.80).
  fig_stack   <- figs/fig_stack.tex: the closed-loop row is removed and the
                 legend moved up by one row pitch (2.55).
  fig_lamK8   <- figs/fig_lamK8.tex: unchanged.

Every edit asserts the text it expects, so a changed draft fails loudly here
instead of silently producing a figure with a closed-loop panel in it.

usage: python derive_figures.py <paper_src_dir> <out_dir>
"""
import os
import re
import sys

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def read(*names):
    for n in names:
        p = os.path.join(src, n)
        if os.path.exists(p):
            return open(p).read(), p
    sys.exit(f"none of {names} found in {src}")


def one(pattern, text, what):
    hits = [m.start() for m in re.finditer(pattern, text)]
    assert len(hits) == 1, f"{what}: expected 1 match for {pattern!r}, got {len(hits)}"
    return hits[0]


# ---- preamble: the paper's own colour definitions ---------------------------
paper, _ = read("iclr2027_conference.tex")
head = paper.split("\\begin{document}")[0]
colors = [l.strip() for l in head.splitlines()
          if l.strip().startswith("\\definecolor")]
assert {"unst", "stab", "dead"} <= {re.search(r"\{(\w+)\}", c).group(1)
                                    for c in colors}, colors

WRAP = r"""\documentclass[tikz,border=2pt]{standalone}
%% Same font, math and colour setup as the paper (iclr2027_conference.tex).
\usepackage{times}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usetikzlibrary{arrows.meta}
%(colors)s
\begin{document}
\input{%(body)s}
\end{document}
"""


def emit(name, body):
    with open(os.path.join(out, f"{name}_body.tex"), "w") as f:
        f.write(body)
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write(WRAP % {"colors": "\n".join(colors), "body": f"{name}_body"})
    print(f"derived {name}")


# ---- fig_method: drop panel (b) ------------------------------------------------
m, mp = read("figs/fig_method.tex", "method.tex")
b0 = one(r"%%[- ]*\(b\) closed-loop probe", m, "method (b) start")
c0 = one(r"%%[- ]*\(c\) per-branch fit", m, "method (c) start")
b0 = m.rfind("\n", 0, b0) + 1
c0 = m.rfind("\n", 0, c0) + 1
m = m[:b0] + m[c0:]
assert "closed-loop" not in m and "lambda_{\\mathrm{cl}}" not in m, "residual CL text"
a_shift = one(r"\\begin\{scope\}\[shift=\{\(0,0\)\}\]", m, "method (a) scope")
m = m.replace("\\begin{scope}[shift={(0,0)}]",
              "\\begin{scope}[shift={(3.4,-0.6)}]  % site: centred, (b) removed", 1)
emit("fig_method", m)

# ---- fig_hist: drop the closed-loop row --------------------------------------
h, _ = read("figs/fig_hist.tex")
lines = h.splitlines()
start = [i for i, l in enumerate(lines) if "shift={(0.00,-5.40)}" in l]
assert len(start) == 1, "fig_hist: closed-loop row start"
start = start[0]
last_lbl = [i for i, l in enumerate(lines) if "{RoboCasa closed loop}" in l]
assert len(last_lbl) == 1, "fig_hist: last closed-loop panel"
end = next(i for i in range(last_lbl[0], len(lines)) if lines[i].strip() == "\\end{scope}")
assert all("closed loop" in "\n".join(lines[start:end + 1]) for _ in [0])
lines = lines[:start] + lines[end + 1:]
row = [i for i, l in enumerate(lines) if l.rstrip().endswith("{closed loop};")]
assert len(row) == 1, "fig_hist: closed-loop row label"
del lines[row[0]]
h = "\n".join(lines) + "\n"
assert "closed loop" not in h, "fig_hist: residual closed-loop panel"
assert h.count("at (-0.4,-6.55)") == 1, "fig_hist: legend anchor"
h = h.replace("at (-0.4,-6.55)", "at (-0.4,-4.75)")
emit("fig_hist", h)

# ---- fig_stack: drop the closed-loop row -------------------------------------
s, _ = read("figs/fig_stack.tex")
lines = s.splitlines()
pol = [i for i, l in enumerate(lines) if l.rstrip().endswith("{policy chunk};")]
cl = [i for i, l in enumerate(lines) if l.rstrip().endswith("{closed loop};")]
assert len(pol) == 1 and len(cl) == 1 and cl[0] > pol[0], "fig_stack: row labels"
block = lines[pol[0] + 1:cl[0] + 1]
ys = [float(y) for l in block for y in re.findall(r",(-?\d+\.\d+)\)", l)]
assert ys and max(ys) <= -6.44 and min(ys) >= -7.72, "fig_stack: CL row extent"
lines = lines[:pol[0] + 1] + lines[cl[0] + 1:]
s = "\n".join(lines) + "\n"
assert "closed loop" not in s
assert s.count("at (0,-9.75)") == 1, "fig_stack: legend anchor"
s = s.replace("at (0,-9.75)", "at (0,-7.20)")
emit("fig_stack", s)

# ---- fig_lamK8: verbatim ------------------------------------------------------
k, _ = read("figs/fig_lamK8.tex")
emit("fig_lamK8", k)
