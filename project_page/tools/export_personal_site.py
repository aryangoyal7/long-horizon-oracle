"""Export the project page into the personal site (aryangoyal7.github.io).

Follows that site's convention for project pages: the publication entry *is*
the page, a standalone HTML file with front matter and `layout: none` in
_publications/, wrapped in {% raw %}, with its assets under
assets/figures/<slug>/. Bulma and Font Awesome load from a CDN instead of
vendored copies, and the page's own CSS and JS are inlined, as on the site's
other project pages. The page is marked as an incomplete draft (banner on the
page, "Drafts" category and status line in the publications list).

usage: python export_personal_site.py <site_repo> [--drive-url URL]
Re-run after rebuilding project_page/ (tools/build.sh) to refresh the copy.
"""
import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
SLUG = "stability-action-chunking"
DATE = "2026-09-30"
ASSET_URL = f"/assets/figures/{SLUG}/"
DRIVE_PLACEHOLDER = "DRIVE_LINK_TO_PAPER_PDF"

FRONT = """---
title: "Measuring the Stability Assumption Behind Action Chunking"
collection: publications
category: drafts
layout: none
permalink: /publication/2026-{slug}
excerpt: 'Per-state measurements of whether a small action error grows or dies out while an action chunk plays out without replanning, across twelve manipulation tasks from robomimic, MimicGen and RoboCasa.'
date: {date}
venue: 'Incomplete draft'
status: 'Incomplete draft'
paperurl: '{drive}'
citation: 'Goyal, A. (2026). &quot;Measuring the Stability Assumption Behind Action Chunking&quot;. Incomplete draft.'
---
{{% raw %}}"""

BANNER = """          <div class="draft-banner">Incomplete draft. Text, figures and numbers may still change, and the closed-loop results are not shown yet.</div>
"""

BANNER_CSS = """
/* personal-site export: draft notice under the title */
.draft-banner {
  display: inline-block;
  margin: 0.25rem auto 1rem;
  padding: 0.35rem 0.9rem;
  border: 1px solid #e6c98f;
  border-radius: 999px;
  background: #fdf6e7;
  color: #7a5a12;
  font-family: 'Roboto Mono', monospace;
  font-size: 0.8rem;
  font-weight: 600;
}
"""


def sub1(pattern, repl, text, what, flags=0):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    assert n == 1, f"export: expected one match for {what}"
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site_repo")
    ap.add_argument("--drive-url", default=DRIVE_PLACEHOLDER)
    args = ap.parse_args()

    page = open(os.path.join(SITE, "index.html")).read()
    css = open(os.path.join(SITE, "static", "css", "index.css")).read() + BANNER_CSS
    js_data = open(os.path.join(SITE, "static", "js", "label_videos.js")).read()
    js = open(os.path.join(SITE, "static", "js", "index.js")).read()
    js = js.replace('return "./static/videos/" + file;', f'return "{ASSET_URL}videos/" + file;')
    assert ASSET_URL + "videos/" in js, "export: video path not rewritten"

    # vendored libraries -> CDN (same versions as the Nerfies copies)
    page = sub1(r'<link rel="stylesheet" href="\./static/css/bulma\.min\.css">',
                '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.1/css/bulma.min.css">',
                page, "bulma")
    page = sub1(r'<script defer src="\./static/js/fontawesome\.all\.min\.js"></script>',
                '<script defer src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/js/all.min.js"></script>',
                page, "fontawesome")
    page = sub1(r'\s*<script src="https://ajax\.googleapis\.com/ajax/libs/jquery/[^"]+"></script>', "", page, "jquery")
    page = sub1(r'\s*<link rel="icon" href="\./static/images/favicon\.svg">', "", page, "favicon")
    # own CSS and JS inline
    page = sub1(r'<link rel="stylesheet" href="\./static/css/index\.css">',
                lambda m: "<style>\n" + css + "</style>", page, "index.css")
    page = sub1(r'<script src="\./static/js/label_videos\.js"></script>\s*<script src="\./static/js/index\.js"></script>',
                lambda m: "<script>\n" + js_data + "</script>\n  <script>\n" + js + "</script>",
                page, "scripts")
    page = page.replace("./static/images/", ASSET_URL)
    # links, authorship, draft marking
    page = sub1(r'href="\./static/paper\.pdf"', f'href="{args.drive_url}"', page, "paper link")
    page = sub1(r'<!-- AUTHORS:.*?-->\s*', "", page, "author comment", flags=re.S)
    page = sub1(r'<span class="author-block">Anonymous authors</span>',
                '<span class="author-block"><a href="/">Aryan Goyal</a></span>', page, "authors")
    page = sub1(r'author  = \{Anonymous\},', 'author  = {Goyal, Aryan},', page, "bibtex author")
    page = sub1(r'(<h1 class="title is-1 publication-title">[^<]*</h1>\n)',
                lambda m: m.group(1) + "\n" + BANNER, page, "title")
    assert "./static/" not in page, "export: unrewritten local path"
    assert "{% endraw %}" not in page and "LIBERO" not in page

    out_pub = os.path.join(args.site_repo, "_publications", f"{DATE}-{SLUG}.html")
    with open(out_pub, "w") as f:
        f.write(FRONT.format(slug=SLUG, date=DATE, drive=args.drive_url) + page.rstrip() + "\n{% endraw %}\n")

    adir = os.path.join(args.site_repo, "assets", "figures", SLUG)
    vdir = os.path.join(adir, "videos")
    os.makedirs(vdir, exist_ok=True)
    for f in sorted(os.listdir(os.path.join(SITE, "static", "images"))):
        if f.startswith("fig_") and f.endswith(".svg"):
            shutil.copyfile(os.path.join(SITE, "static", "images", f), os.path.join(adir, f))
    n = 0
    for f in sorted(os.listdir(os.path.join(SITE, "static", "videos"))):
        if f.endswith(".mp4") and not f.endswith(".part.mp4"):
            shutil.copyfile(os.path.join(SITE, "static", "videos", f), os.path.join(vdir, f))
            n += 1
    print(f"wrote {os.path.relpath(out_pub, args.site_repo)}, {n} videos and the figures to "
          f"{os.path.relpath(adir, args.site_repo)}/"
          + ("" if args.drive_url != DRIVE_PLACEHOLDER else
             f"\nDrive link not set: replace {DRIVE_PLACEHOLDER} in the publication file (2 places)"))


if __name__ == "__main__":
    main()
