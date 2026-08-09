"""
Sketch-Prompted VLA -- ground-truth inspector for one validation scene.

Read-only. Answers the question "which bowl, which plate?" for a scene whose
instruction is ambiguous by construction, by reading the authoritative sources
off disk instead of eyeballing `sketch.png`.

The authority, in order:

  1. `scene.bddl` `(:goal ...)`   -- the predicate the simulator actually scores.
  2. `meta.json` / `tokens.json`  -- `target`, `destination`, `destination_region`,
                                     re-parsed against (1) by
                                     `scripts/audit_validation_sets.py`.
  3. `meta.json["all_pixels"]`    -- the pixel centre of EVERY candidate instance,
                                     which is what turns an instance name into a
                                     blob I can point at in the image.

`(:obj_of_interest ...)` in the BDDL is NOT authoritative. It is inherited from
the stock LIBERO task and was not rewritten when the duplicate instances were
injected, so on 15 of the 114 scenes it names a different instance than the goal
(e.g. goal/scene_0026 lists plate_1 while the goal is `On akita_black_bowl_1
plate_2`). Ignore it.

Usage:
    python scripts/show_scene_truth.py goal 26          # one scene, printed + overlay PNG
    python scripts/show_scene_truth.py goal scene_0026 --no-image
    python scripts/show_scene_truth.py --all-ambiguous  # which scenes have decoys
    python scripts/show_scene_truth.py --subset         # answer key for the 36 study scenes
    python scripts/show_scene_truth.py --all            # answer key for all 114

`--subset` / `--all` write a single self-contained answer_key HTML (images
embedded, no loose files to keep together) next to the overlays. Open it in a
browser and scroll: each row is the labelled frame beside the instruction and
the two instance names. Nothing to read out of a JSON by hand.
"""

import argparse
import base64
import glob
import html
import io
import json
import math
import os
import re
import sys

from PIL import Image, ImageDraw

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(_REPO, "outputs")
SUITES = {"spatial": "validation_set_spatial",
          "object": "validation_set_object",
          "goal": "validation_set_goal"}
STUDY_DIR = os.path.join(ROOT, "human_study")
OUT_DIR = os.path.join(STUDY_DIR, "truth_overlays")

SCALE = 6                       # upscale factor for the overlay PNG
TARGET_RGB = (0, 200, 0)        # same green the auto circle uses
DEST_RGB = (200, 50, 50)        # same red the auto arrow uses
OTHER_RGB = (90, 110, 255)      # decoys / siblings / other destinations


def scene_path(suite, scene):
    if suite not in SUITES:
        sys.exit("unknown suite %r (spatial | object | goal)" % suite)
    name = scene if str(scene).startswith("scene_") else "scene_%04d" % int(scene)
    path = os.path.join(ROOT, SUITES[suite], name)
    if not os.path.isdir(path):
        sys.exit("no such scene on disk: %s" % path)
    return name, path


def bddl_goal_line(path):
    text = open(os.path.join(path, "scene.bddl"), encoding="utf-8").read()
    m = re.search(r"\(:goal(.*?)\n\s*\)", text, re.S)
    return " ".join(m.group(1).split()) if m else "(:goal not parsed)"


def report(suite, scene, make_image=True):
    name, path = scene_path(suite, scene)
    meta = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8"))
    px = meta.get("all_pixels", {})
    target, dest = meta["target"], meta["destination"]

    print("=" * 68)
    print("%s / %s     tier=%s     seed=%s" % (suite, name, meta["tier"], meta["seed"]))
    print("=" * 68)
    print("instruction         : %r" % meta["instruction"])
    print("BDDL goal (scored)  : %s" % bddl_goal_line(path))
    print()
    print("GRASP THIS          : %-28s" % target, end="")
    print("pixel %s" % (px.get(target, "(not a movable instance)"),))
    print("PUT IT ON/IN        : %-28s" % dest, end="")
    print("pixel %s" % (px.get(dest, "-> region, see place_px %s" % (meta["place_px"],)),))
    if meta.get("destination_region") != dest:
        print("  scored region     : %s   (BDDL 2nd arg; the arrow still points at %s)"
              % (meta["destination_region"], dest))
    print()

    print("every candidate in frame  [col, row] on the 128x128 image")
    for k in sorted(px, key=lambda k: (k != target, k != dest, k)):
        tag = "  <-- TARGET (circle)" if k == target else \
              "  <-- DESTINATION (arrow head)" if k == dest else ""
        print("    %-28s %s%s" % (k, px[k], tag))
    print("    (auto sketch: pick_px %s, place_px %s, radius %s)"
          % (meta["pick_px"], meta["place_px"], meta["radius"]))
    print()

    neg = meta.get("oracle_negatives") or {}
    if neg:
        print("oracle checks -- these were run in the simulator when the scene was built:")
        print("    %-42s -> scores %s" % ("%s -> %s" % (target, meta["destination_region"]),
                                          meta.get("oracle_success")))
        for k, v in neg.items():
            print("    %-42s -> scores %s" % (k, v))
        print("    so picking any other instance is a genuine failure, not a tie.")
        print()

    if make_image:
        out = overlay(path, meta, suite, name)
        print("labelled overlay written: %s" % out)
    return meta


def build_overlay(path, meta):
    """frame0 upscaled, every candidate ringed and named, target green / dest red."""
    img = Image.open(os.path.join(path, "frame0.png")).convert("RGB")
    img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
    d = ImageDraw.Draw(img)

    target, dest = meta["target"], meta["destination"]
    px = dict(meta.get("all_pixels", {}))
    if dest not in px:
        px[dest] = list(meta["place_px"])          # region-typed destination

    for k, (cx, cy) in px.items():
        col = TARGET_RGB if k == target else DEST_RGB if k == dest else OTHER_RGB
        r = max(6.0, float(meta.get("px_extent", {}).get(k, 11))) * SCALE * 0.5
        x, y = cx * SCALE, cy * SCALE
        d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3)
        d.line([x - 5, y, x + 5, y], fill=col, width=2)
        d.line([x, y - 5, x, y + 5], fill=col, width=2)
        label = k + ("  [TARGET]" if k == target else "  [DEST]" if k == dest else "")
        tx, ty = x + r + 4, y - 7
        tw = 6 * len(label) + 6
        if tx + tw > img.width:
            tx = x - r - 4 - tw
        d.rectangle([tx - 3, ty - 2, tx + tw, ty + 12], fill=(255, 255, 255))
        d.text((tx, ty), label, fill=col)
    return img


def overlay(path, meta, suite, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "%s_%s_truth.png" % (suite, name))
    build_overlay(path, meta).save(out)
    return out


# ------------------------------------------------------------------ answer key --
_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:28px 34px;
     background:#fafafa;color:#1a1a1a}
h1{font-size:20px;margin:0 0 4px} .sub{color:#666;margin:0 0 22px;max-width:70ch}
.warn{background:#fff6e5;border-left:3px solid #e8a33d;padding:10px 14px;margin:0 0 24px;
      max-width:70ch;font-size:13px}
.row{display:flex;gap:22px;align-items:flex-start;background:#fff;border:1px solid #e3e3e3;
     border-radius:8px;padding:16px;margin-bottom:14px}
.row img{width:384px;height:384px;image-rendering:pixelated;border:1px solid #ddd;border-radius:4px}
.info{flex:1;min-width:0}
.hdr{font-size:12px;color:#777;letter-spacing:.04em;text-transform:uppercase;margin-bottom:6px}
.instr{font-size:17px;font-weight:600;margin:0 0 14px}
table{border-collapse:collapse;font-size:13px} td{padding:3px 12px 3px 0;vertical-align:top}
td.k{color:#777;white-space:nowrap} code{font:12px ui-monospace,Consolas,monospace;
     background:#f2f2f2;padding:1px 5px;border-radius:3px}
.g code{background:#e4f6e4;color:#0a6b0a} .r code{background:#fdeaea;color:#a41d1d}
.tier{display:inline-block;font-size:11px;background:#eef1f7;color:#41506e;border-radius:10px;
      padding:2px 9px;margin-left:8px;vertical-align:2px}
"""


def answer_key(rows, out_name, title, note):
    """rows: list of (order_label, suite, dir). Writes one self-contained HTML."""
    os.makedirs(OUT_DIR, exist_ok=True)
    parts = ["<!doctype html><meta charset='utf-8'><title>%s</title><style>%s</style>"
             % (html.escape(title), _CSS),
             "<h1>%s</h1><p class='sub'>%s</p>" % (html.escape(title), html.escape(note)),
             "<div class='warn'><b>Draw first, check second.</b> 90 of the 114 scenes are "
             "ambiguous on purpose. Reading the answer before sketching turns my stroke into "
             "a copy of the auto-sketch and voids the comparison.</div>"]

    for label, suite, scene in rows:
        name, path = scene_path(suite, scene)
        meta = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8"))
        buf = io.BytesIO()
        build_overlay(path, meta).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        others = [k for k in sorted(meta.get("all_pixels", {}))
                  if k not in (meta["target"], meta["destination"])]
        reg = ""
        if meta.get("destination_region") != meta["destination"]:
            reg = ("<tr><td class='k'>scored region</td><td><code>%s</code></td></tr>"
                   % html.escape(meta["destination_region"]))
        parts.append(
            "<div class='row'><img src='data:image/png;base64,%s'><div class='info'>"
            "<div class='hdr'>%s &nbsp;·&nbsp; %s / %s<span class='tier'>%s</span></div>"
            "<p class='instr'>&ldquo;%s&rdquo;</p><table>"
            "<tr class='g'><td class='k'>grasp (circle)</td><td><code>%s</code></td></tr>"
            "<tr class='r'><td class='k'>place on/in (arrow)</td><td><code>%s</code></td></tr>%s"
            "<tr><td class='k'>decoys in frame</td><td>%s</td></tr>"
            "<tr><td class='k'>BDDL goal</td><td><code>%s</code></td></tr>"
            "</table></div></div>"
            % (b64, html.escape(label), suite, name, html.escape(meta["tier"]),
               html.escape(meta["instruction"]),
               html.escape(meta["target"]), html.escape(meta["destination"]), reg,
               html.escape(", ".join(others) or "none"),
               html.escape(bddl_goal_line(path))))

    out = os.path.join(OUT_DIR, out_name)
    open(out, "w", encoding="utf-8").write("\n".join(parts))
    return out


def subset_rows():
    p = os.path.join(STUDY_DIR, "scene_subset.json")
    if not os.path.isfile(p):
        sys.exit("no scene_subset.json -- run scripts/build_human_study_bundle.py first")
    d = json.load(open(p, encoding="utf-8"))
    rows = []
    if d.get("practice"):
        rows.append(("practice", d["practice"]["suite"], d["practice"]["dir"]))
    for s in sorted(d["scenes"], key=lambda s: s["order"]):
        rows.append(("scene %d of %d" % (s["order"] + 1, len(d["scenes"])),
                     s["suite"], s["dir"]))
    return rows


def all_rows():
    rows = []
    for suite in ("spatial", "object", "goal"):
        for path in sorted(glob.glob(os.path.join(ROOT, SUITES[suite], "scene_*"))):
            rows.append(("", suite, os.path.basename(path)))
    return rows


def all_ambiguous():
    """Every scene where more than one instance shares the target's or the
    destination's category -- i.e. every scene where the caption alone cannot
    settle which object is meant."""
    rows = []
    for suite in SUITES:
        for path in sorted(glob.glob(os.path.join(ROOT, SUITES[suite], "scene_*"))):
            meta = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8"))
            sib = len(meta.get("siblings") or [])
            oth = len(meta.get("other_dests") or meta.get("other_baskets") or [])
            if sib or oth:
                rows.append((suite, os.path.basename(path), meta["tier"],
                             meta["target"], meta["destination"], sib, oth,
                             meta["instruction"]))
    print("%-8s %-12s %-12s %-24s %-24s %s" %
          ("suite", "dir", "tier", "target", "destination", "n_sib/n_otherdest"))
    for r in rows:
        print("%-8s %-12s %-12s %-24s %-24s %d/%d   %r" %
              (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
    print("\n%d of 114 scenes are ambiguous from the caption alone." % len(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", nargs="?", help="spatial | object | goal")
    ap.add_argument("scene", nargs="?", help="26 or scene_0026")
    ap.add_argument("--no-image", action="store_true", help="print only")
    ap.add_argument("--all-ambiguous", action="store_true",
                    help="list every scene with a same-category decoy")
    ap.add_argument("--subset", action="store_true",
                    help="answer-key HTML for the scenes in scene_subset.json")
    ap.add_argument("--all", action="store_true",
                    help="answer-key HTML for all 114 scenes")
    a = ap.parse_args()
    if a.all_ambiguous:
        all_ambiguous()
        return
    if a.subset or a.all:
        rows = subset_rows() if a.subset else all_rows()
        out = answer_key(rows,
                         "answer_key_subset.html" if a.subset else "answer_key_all.html",
                         "Sketch-Prompted VLA — %s answer key"
                         % ("human-study subset" if a.subset else "full validation set"),
                         "Ground truth re-read from each scene's BDDL goal and meta.json. "
                         "Green ring = the instance to circle, red ring = the instance the "
                         "arrow head belongs on, blue rings = same-category decoys.")
        print("%d scenes -> %s" % (len(rows), out))
        return
    if not (a.suite and a.scene):
        ap.error("give a suite and a scene, or --all-ambiguous")
    report(a.suite, a.scene, make_image=not a.no_image)


if __name__ == "__main__":
    main()
