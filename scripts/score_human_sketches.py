"""
Sketch-Prompted VLA — human-vs-auto sketch comparison harness (stdlib + numpy +
Pillow + matplotlib only; runs anywhere, no robosuite / mujoco / libero).

Loads every annotator response in `outputs/human_study/responses/`, rejoins the
ground truth from disk on `(suite, dir)`, and answers the question the three
validation-set datasheets currently list as an open limitation: how large is the
synthetic-to-human gap, and is the auto-annotator's imprecision augmentation wide
enough to cover real human strokes?

Four metric families, each reported pooled AND broken down by suite and by tier:

  (a) human correctness   referential / directional / joint accuracy, skip rate
  (b) geometric deviation human stroke vs the auto symbolic_tokens, per scene
  (c) calibration verdict the four auto jitter ranges against the human empirical
                          distributions, with a recommended range per parameter
  (d) agreement + effort  pairwise inter-annotator agreement, timings, stroke counts

Outputs into `outputs/human_study/comparison/`:
    metrics.json            every number this script computes
    per_scene.csv           one row per (annotator, suite, dir)
    fig_accuracy.png        (a) by suite and tier
    fig_calibration.png     (c) human distribution vs the auto uniform range
    fig_geometry.png        (b) deviation distributions
    fig_effort.png          (d) time and stroke counts
    contact_sheet_<id>.png  auto sketch beside human sketch, per annotated scene

and the human sketches themselves into scene folders that MIRROR the validation
suites, so a loader is pointed at a different root and works unchanged:

    outputs/human_study/rendered/<annotator>/validation_set_<suite>/<dir>/
        sketch.png     128x128 RGB, auto colour scheme, format-identical to the auto one
        tokens.json    canonical shape, symbolic_tokens filled from the HUMAN geometry
    outputs/human_study/rendered/consensus/validation_set_<suite>/<dir>/
        the medoid annotator's sketch for that scene, plus the group median geometry

    cd C:\\Users\\Admin\\sketch_prompted_vla
    python scripts/score_human_sketches.py
    python scripts/score_human_sketches.py --smoke     # score the vertical slice
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- constants --
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rename-proof
ROOT = os.path.join(_REPO, "outputs")
STUDY_DIR = os.path.join(ROOT, "human_study")

SUITES = {"spatial": "validation_set_spatial",
          "object":  "validation_set_object",
          "goal":    "validation_set_goal"}
TIERS = ("control", "referential", "directional", "both")

IMG_PX = 128
CIRCLE_RGB = (0, 200, 0)      # must match draw_circle in the builders
ARROW_RGB = (200, 50, 50)     # must match draw_arrow in the builders
SS = 4                        # supersample factor for antialiased rendering
STROKE_W = 6                  # at SS=4 this is 1.5 px, inside the auto 1-2 px range

# The four auto jitter ranges under test, from draw_circle / draw_arrow in
# build_validation_set_{spatial,object,goal}.py. All three builders use
# byte-identical ranges, verified 2026-08-02, so one hypothesis covers all suites.
#   rng.integers(-3, 4)   -> integers in [-3, 3]      circle centre, per axis
#   rng.uniform(-2, 3)    -> continuous in [-2, 3)    circle radius, per axis
#   rng.integers(-2, 3)   -> integers in [-2, 2]      arrow endpoints, per axis
#   rng.integers(-7, 8)   -> integers in [-7, 7]      arrow midpoint bend, per axis
AUTO_JITTER = {
    "circle_centre_px":  {"lo": -3.0, "hi": 3.0, "kind": "int",
                          "label": "circle centre offset from pick_px (px, per axis)"},
    "circle_radius_px":  {"lo": -2.0, "hi": 3.0, "kind": "float",
                          "label": "circle radius minus auto radius (px, per axis)"},
    "arrow_endpoint_px": {"lo": -2.0, "hi": 2.0, "kind": "int",
                          "label": "arrow endpoint offset from pick_px / place_px (px, per axis)"},
    "arrow_bend_px":     {"lo": 0.0, "hi": None, "kind": "derived",
                          "label": "arrow deviation from the tail-head chord (px)"},
}
BEND_HALFWIDTH = 7            # the [-7, 7] midpoint offset the bend metric tests
BEND_SIM_N = 4000             # Monte-Carlo draws per scene for the auto bend law
BEND_SIM_SEED = 20260802

# A parameter is judged too tight if this much of the human mass falls outside the
# auto range, and too loose if the human 5-95 interval is this much narrower.
TOO_TIGHT_FRAC = 0.25
TOO_LOOSE_WIDTH_RATIO = 0.5

# Outlier cut for the robust companion recommendation, in normal-consistent
# median-absolute-deviations. 3 is conventional and is what the arrow-tail
# convention outliers were measured against; see the note in calibration().
ROBUST_K = 3.0


# ------------------------------------------------------------------ geometry --
def ellipse_contains(fit, pt):
    """Point-in-ellipse for a rotated ellipse {cx, cy, rx, ry, theta_deg}."""
    th = math.radians(fit.get("theta_deg") or 0.0)
    dx, dy = pt[0] - fit["cx"], pt[1] - fit["cy"]
    u = dx * math.cos(th) + dy * math.sin(th)
    v = -dx * math.sin(th) + dy * math.cos(th)
    rx, ry = max(fit["rx"], 1e-6), max(fit["ry"], 1e-6)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def ellipse_mask(fit, x0, y0, w, h, ss=4):
    """Boolean occupancy of an ellipse on a supersampled grid, for IoU."""
    th = math.radians(fit.get("theta_deg") or 0.0)
    xs = (np.arange(w * ss) + 0.5) / ss + x0
    ys = (np.arange(h * ss) + 0.5) / ss + y0
    gx, gy = np.meshgrid(xs, ys)
    dx, dy = gx - fit["cx"], gy - fit["cy"]
    u = dx * math.cos(th) + dy * math.sin(th)
    v = -dx * math.sin(th) + dy * math.cos(th)
    return (u / max(fit["rx"], 1e-6)) ** 2 + (v / max(fit["ry"], 1e-6)) ** 2 <= 1.0


def ellipse_iou(a, b):
    lo_x = min(a["cx"] - a["rx"] - a["ry"], b["cx"] - b["rx"] - b["ry"]) - 2
    lo_y = min(a["cy"] - a["rx"] - a["ry"], b["cy"] - b["rx"] - b["ry"]) - 2
    hi_x = max(a["cx"] + a["rx"] + a["ry"], b["cx"] + b["rx"] + b["ry"]) + 2
    hi_y = max(a["cy"] + a["rx"] + a["ry"], b["cy"] + b["rx"] + b["ry"]) + 2
    w, h = int(math.ceil(hi_x - lo_x)), int(math.ceil(hi_y - lo_y))
    if w <= 0 or h <= 0 or w > 4096 or h > 4096:
        return float("nan")
    ma, mb = ellipse_mask(a, lo_x, lo_y, w, h), ellipse_mask(b, lo_x, lo_y, w, h)
    union = np.count_nonzero(ma | mb)
    return float(np.count_nonzero(ma & mb) / union) if union else float("nan")


def chord_deviation(points, tail, head):
    """Maximum perpendicular distance of a polyline from the tail-head chord.
    Directly comparable to the auto drawer's midpoint bend, whose path is two
    straight segments and whose maximum deviation is therefore at the midpoint."""
    if points is None or len(points) < 3:
        return 0.0
    p = np.asarray(points, dtype=float)
    t, h = np.asarray(tail, dtype=float), np.asarray(head, dtype=float)
    d = h - t
    L = float(np.hypot(*d))
    if L < 1e-9:
        return float(np.max(np.hypot(*(p - t).T)))
    q = p - t
    return float(np.max(np.abs(d[0] * q[:, 1] - d[1] * q[:, 0]) / L))


def auto_bend_samples(tok, rng):
    """The auto drawer's chord deviation, simulated from its own [-7, 7] midpoint
    law on this scene's actual chord. symbolic_tokens records only the endpoints,
    so the bend has to be reconstructed rather than read."""
    x0, y0 = float(tok["x0"]), float(tok["y0"])
    x1, y1 = float(tok["x1"]), float(tok["y1"])
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    mx = (x0 + x1) // 2 + rng.integers(-BEND_HALFWIDTH, BEND_HALFWIDTH + 1, BEND_SIM_N)
    my = (y0 + y1) // 2 + rng.integers(-BEND_HALFWIDTH, BEND_HALFWIDTH + 1, BEND_SIM_N)
    if L < 1e-9:
        return np.hypot(mx - x0, my - y0)
    return np.abs(dx * (my - y0) - dy * (mx - x0)) / L


def ang_deg(x0, y0, x1, y1):
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def nanmed(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else None


def summarise(v):
    v = np.asarray([x for x in v
                    if x is not None and not (isinstance(x, float) and math.isnan(x))],
                   dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=0)),
            "p05": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
            "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max())}


# ---------------------------------------------------------------- truth from disk --
def load_truth(suite, scene_dir):
    path = os.path.join(ROOT, SUITES[suite], scene_dir, "meta.json")
    if not os.path.isfile(path):
        sys.exit("no truth on disk for (%s, %s): %s" % (suite, scene_dir, path))
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    cand = {k: [float(v[0]), float(v[1])] for k, v in meta["all_pixels"].items()}
    injected = meta["destination"] not in cand
    if injected:
        # 11 Goal scenes name a fixture or region as the destination
        # (flat_stove_1, wine_rack_1, wooden_cabinet_1, main_table). all_pixels
        # enumerates movable instances only, so the destination is absent and the
        # nearest-candidate directional test would be undefined. It is added at
        # place_px, which is the projected region point the auto arrow already
        # aims at. Flagged per scene so the substitution is auditable.
        cand[meta["destination"]] = [float(meta["place_px"][0]), float(meta["place_px"][1])]
    return {"meta": meta, "suite": suite, "dir": scene_dir, "tier": meta["tier"],
            "instruction": meta["instruction"], "target": meta["target"],
            "destination": meta["destination"], "candidates": cand,
            "dest_injected": injected,
            "pick_px": [float(x) for x in meta["pick_px"]],
            "place_px": [float(x) for x in meta["place_px"]],
            "radius": float(meta["radius"]),
            "auto_circle": dict(meta["symbolic_tokens"]["circle"], theta_deg=0.0),
            "auto_arrow": {k: float(v) for k, v in meta["symbolic_tokens"]["arrow"].items()},
            "path": os.path.join(ROOT, SUITES[suite], scene_dir)}


# --------------------------------------------------------------- per-scene scoring --
def score_one(ann, truth):
    """Metric families (a) and (b) for one annotator on one scene."""
    r = {"annotator": ann["_annotator"], "suite": truth["suite"], "dir": truth["dir"],
         "tier": truth["tier"], "instruction": truth["instruction"],
         "skipped": bool(ann.get("skipped")), "skip_reason": ann.get("skip_reason"),
         "dest_from_place_px": truth["dest_injected"]}
    eff = ann.get("effort") or {}
    for k in ("time_to_first_stroke_ms", "time_total_ms", "n_undo", "n_redraw",
              "n_points_circle", "n_points_arrow"):
        r[k] = eff.get(k)
    if r["skipped"]:
        return r

    circ = (ann.get("circle") or {}).get("fit")
    arrow = ann.get("arrow")
    r["has_circle"], r["has_arrow"] = circ is not None, arrow is not None

    # -- (a) referential ------------------------------------------------------
    if circ:
        inside = sorted(k for k, p in truth["candidates"].items()
                        if ellipse_contains(circ, p))
        r["contained"] = "|".join(inside)
        r["contains_target"] = truth["target"] in inside
        r["referential_ok"] = (inside == [truth["target"]])
        if r["referential_ok"]:
            r["referential_fail"] = ""
        elif len(inside) == 0:
            r["referential_fail"] = "contains none"
        elif len(inside) > 1:
            r["referential_fail"] = "contains several"
        else:
            r["referential_fail"] = "contains the wrong object"

    # -- (a) directional ------------------------------------------------------
    if arrow:
        head = (float(arrow["x1"]), float(arrow["y1"]))
        near = min(truth["candidates"].items(),
                   key=lambda kv: math.hypot(kv[1][0] - head[0], kv[1][1] - head[1]))
        r["head_nearest"] = near[0]
        r["directional_ok"] = (near[0] == truth["destination"])
        r["head_to_dest_px"] = math.hypot(head[0] - truth["place_px"][0],
                                          head[1] - truth["place_px"][1])

    if circ and arrow:
        r["joint_ok"] = bool(r.get("referential_ok")) and bool(r.get("directional_ok"))

    # -- (b) circle deviation vs auto ----------------------------------------
    a = truth["auto_circle"]
    if circ:
        r["circle_centre_offset_px"] = math.hypot(circ["cx"] - a["cx"], circ["cy"] - a["cy"])
        r["circle_centre_offset_norm"] = r["circle_centre_offset_px"] / max(a["rx"], 1e-6)
        r["circle_radius_ratio"] = ((circ["rx"] + circ["ry"]) / 2.0) / \
                                   max((a["rx"] + a["ry"]) / 2.0, 1e-6)
        r["circle_iou"] = ellipse_iou(circ, a)
        r["circle_theta_deg"] = circ.get("theta_deg")
        # (c) raw jitter residuals, measured against the SAME anchors the auto
        # drawer jitters away from: pick_px and the intended `radius`.
        r["j_circle_dx"] = circ["cx"] - truth["pick_px"][0]
        r["j_circle_dy"] = circ["cy"] - truth["pick_px"][1]
        r["j_radius_dx"] = circ["rx"] - truth["radius"]
        r["j_radius_dy"] = circ["ry"] - truth["radius"]

    # -- (b) arrow deviation vs auto -----------------------------------------
    b = truth["auto_arrow"]
    if arrow:
        hx0, hy0 = float(arrow["x0"]), float(arrow["y0"])
        hx1, hy1 = float(arrow["x1"]), float(arrow["y1"])
        r["arrow_tail_offset_px"] = math.hypot(hx0 - b["x0"], hy0 - b["y0"])
        r["arrow_head_offset_px"] = math.hypot(hx1 - b["x1"], hy1 - b["y1"])
        d = wrap180(ang_deg(hx0, hy0, hx1, hy1) - ang_deg(b["x0"], b["y0"], b["x1"], b["y1"]))
        r["arrow_angle_diff_deg"] = d
        r["arrow_angle_absdiff_deg"] = abs(d)
        auto_len = math.hypot(b["x1"] - b["x0"], b["y1"] - b["y0"])
        r["arrow_length_ratio"] = math.hypot(hx1 - hx0, hy1 - hy0) / max(auto_len, 1e-6)
        r["arrow_curvature_px"] = chord_deviation(arrow.get("points"), (hx0, hy0), (hx1, hy1))
        r["j_arrow_x0"] = hx0 - truth["pick_px"][0]
        r["j_arrow_y0"] = hy0 - truth["pick_px"][1]
        r["j_arrow_x1"] = hx1 - truth["place_px"][0]
        r["j_arrow_y1"] = hy1 - truth["place_px"][1]
    return r


# ---------------------------------------------------------------- (a) rollup --
def rate(rows, key):
    vals = [bool(r[key]) for r in rows if key in r and r[key] is not None]
    return {"n": len(vals), "rate": (sum(vals) / len(vals)) if vals else None}


def accuracy_block(rows):
    live = [r for r in rows if not r["skipped"]]
    out = {"n_scenes": len(rows), "n_skipped": sum(r["skipped"] for r in rows),
           "skip_rate": (sum(r["skipped"] for r in rows) / len(rows)) if rows else None,
           "skip_reasons": dict(Counter(r["skip_reason"] for r in rows if r["skipped"])),
           "referential": rate(live, "referential_ok"),
           "referential_loose_contains_target": rate(live, "contains_target"),
           "directional": rate(live, "directional_ok"),
           "joint": rate(live, "joint_ok"),
           "referential_failures": dict(Counter(
               r["referential_fail"] for r in live
               if r.get("referential_fail"))),
           }
    return out


def by_group(rows, keyfn, fn):
    out = {}
    for r in rows:
        out.setdefault(keyfn(r), []).append(r)
    return {k: fn(v) for k, v in sorted(out.items())}


# ------------------------------------------------------- (c) calibration verdict --
def calibration(rows, truths):
    """The four auto jitter ranges against the human empirical distributions.

    Only CORRECT strokes feed the calibration. A circle drawn around the wrong
    object is not an imprecise circle, it is a different intent, and letting it
    into the distribution would recommend widening the augmentation to cover
    annotator error — the opposite of what the augmentation is for. Circle
    parameters therefore use referentially correct scenes and arrow parameters
    directionally correct ones. The excluded counts are reported alongside."""
    live = [r for r in rows if not r["skipped"]]
    circ_ok = [r for r in live if r.get("referential_ok")]
    arrow_ok = [r for r in live if r.get("directional_ok")]
    human = {
        "circle_centre_px":  [r[k] for r in circ_ok for k in ("j_circle_dx", "j_circle_dy")
                              if r.get(k) is not None],
        "circle_radius_px":  [r[k] for r in circ_ok for k in ("j_radius_dx", "j_radius_dy")
                              if r.get(k) is not None],
        "arrow_endpoint_px": [r[k] for r in arrow_ok
                              for k in ("j_arrow_x0", "j_arrow_y0", "j_arrow_x1", "j_arrow_y1")
                              if r.get(k) is not None],
        "arrow_bend_px":     [r["arrow_curvature_px"] for r in arrow_ok
                              if r.get("arrow_curvature_px") is not None],
    }
    basis = {"circle_centre_px": len(circ_ok), "circle_radius_px": len(circ_ok),
             "arrow_endpoint_px": len(arrow_ok), "arrow_bend_px": len(arrow_ok)}
    excluded = {"circle": len(live) - len(circ_ok), "arrow": len(live) - len(arrow_ok)}

    # The auto bend law depends on each scene's chord, so it is simulated on the
    # chords of exactly the scenes that were annotated.
    rng = np.random.default_rng(BEND_SIM_SEED)
    keys = {(r["suite"], r["dir"]) for r in arrow_ok} or {(r["suite"], r["dir"]) for r in live}
    auto_bend = np.concatenate([auto_bend_samples(truths[k]["auto_arrow"], rng)
                                for k in sorted(keys)]) if keys else np.zeros(0)

    out = {}
    for name, spec in AUTO_JITTER.items():
        v = np.asarray(human[name], dtype=float)
        blk = {"label": spec["label"], "auto_lo": spec["lo"], "auto_hi": spec["hi"],
               "human": summarise(v.tolist()), "_values": human[name],
               "n_scenes_in_basis": basis[name],
               "n_scenes_excluded_as_incorrect":
                   excluded["circle" if name.startswith("circle") else "arrow"]}
        if v.size == 0:
            out[name] = blk
            continue

        if name == "arrow_bend_px":
            blk["auto"] = summarise(auto_bend.tolist())
            hi = float(np.percentile(auto_bend, 95)) if auto_bend.size else 0.0
            blk["auto_hi"] = float(auto_bend.max()) if auto_bend.size else 0.0
            blk["auto_p95"] = hi
            blk["frac_outside"] = float(np.mean(v > hi))
            blk["frac_above_auto_max"] = float(np.mean(v > blk["auto_hi"]))
            # Recommend the midpoint half-width k whose simulated 95th percentile
            # best matches the human 95th percentile, on the same chords.
            target = float(np.percentile(v, 95))
            best, best_err = BEND_HALFWIDTH, None
            for k in range(1, 26):
                r2 = np.random.default_rng(BEND_SIM_SEED + k)
                sim = []
                for key in sorted(keys):
                    tok = truths[key]["auto_arrow"]
                    dx = tok["x1"] - tok["x0"]; dy = tok["y1"] - tok["y0"]
                    L = math.hypot(dx, dy) or 1e-9
                    mx = (tok["x0"] + tok["x1"]) // 2 + r2.integers(-k, k + 1, 400)
                    my = (tok["y0"] + tok["y1"]) // 2 + r2.integers(-k, k + 1, 400)
                    sim.append(np.abs(dx * (my - tok["y0"]) - dy * (mx - tok["x0"])) / L)
                err = abs(float(np.percentile(np.concatenate(sim), 95)) - target)
                if best_err is None or err < best_err:
                    best, best_err = k, err
            blk["recommended"] = {"midpoint_halfwidth": best,
                                  "current": BEND_HALFWIDTH,
                                  "as_code": "rng.integers(-%d, %d)" % (best, best + 1)}
            width_ratio = (target / hi) if hi > 0 else float("inf")
        else:
            lo, hi = spec["lo"], spec["hi"]
            blk["frac_outside"] = float(np.mean((v < lo) | (v > hi)))
            blk["frac_below"] = float(np.mean(v < lo))
            blk["frac_above"] = float(np.mean(v > hi))
            p05, p95 = float(np.percentile(v, 5)), float(np.percentile(v, 95))
            if spec["kind"] == "int":
                rlo, rhi = int(math.floor(p05)), int(math.ceil(p95))
                blk["recommended"] = {"lo": rlo, "hi": rhi,
                                      "as_code": "rng.integers(%d, %d)" % (rlo, rhi + 1)}
            else:
                rlo, rhi = round(p05, 1), round(p95, 1)
                blk["recommended"] = {"lo": rlo, "hi": rhi,
                                      "as_code": "rng.uniform(%s, %s)" % (rlo, rhi)}
            width_ratio = (p95 - p05) / (hi - lo) if hi > lo else float("inf")

        # -- robust companion recommendation ---------------------------------
        # The raw 5th-95th percentiles are not resistant: a minority of strokes
        # drawn with a different CONVENTION rather than a shakier hand drags them
        # out badly. Measured on the first response file, 21% of arrow-endpoint
        # residuals sat beyond 3 robust SDs — nearly all of them arrow TAILS,
        # where the human starts the stroke at the edge of the object they just
        # circled while the auto drawer anchors at the centroid. Those 21% pushed
        # the raw recommendation to rng.integers(-14, 19), ±15% of a 128px frame,
        # which would model a convention difference as hand imprecision and blow
        # the augmentation wide open.
        #
        # So a second recommendation is reported alongside, computed after
        # dropping values more than ROBUST_K median-absolute-deviations from the
        # median (MAD scaled by 1.4826 to be a normal-consistent SD estimate).
        # Neither replaces the other: `recommended` is the raw distribution,
        # `recommended_robust` is the distribution with convention outliers
        # removed. Which one to paste into the builders is a judgement about
        # whether those strokes are noise to reproduce or behaviour to exclude —
        # record the choice rather than letting the scorer make it silently.
        med = float(np.median(v))
        rsd = 1.4826 * float(np.median(np.abs(v - med)))
        keep = v[np.abs(v - med) <= ROBUST_K * rsd] if rsd > 0 else v
        blk["robust"] = {"median": med, "mad_sd": rsd,
                         "n_kept": int(keep.size), "n_dropped": int(v.size - keep.size),
                         "frac_dropped": float((v.size - keep.size) / v.size) if v.size else 0.0}
        if keep.size and name != "arrow_bend_px":
            q05, q95 = (float(x) for x in np.percentile(keep, [5, 95]))
            if spec["kind"] == "int":
                blk["recommended_robust"] = {
                    "lo": int(math.floor(q05)), "hi": int(math.ceil(q95)),
                    "as_code": "rng.integers(%d, %d)" % (math.floor(q05), math.ceil(q95) + 1)}
            else:
                blk["recommended_robust"] = {
                    "lo": round(q05, 1), "hi": round(q95, 1),
                    "as_code": "rng.uniform(%s, %s)" % (round(q05, 1), round(q95, 1))}

        blk["human_width_over_auto_width"] = width_ratio
        if blk["frac_outside"] > TOO_TIGHT_FRAC:
            blk["verdict"] = "too tight"
        elif width_ratio < TOO_LOOSE_WIDTH_RATIO:
            blk["verdict"] = "too loose"
        else:
            blk["verdict"] = "about right"
        out[name] = blk
    return out


# --------------------------------------------------- (d) agreement and effort --
def agreement(per_annotator, truths):
    ids = sorted(per_annotator)
    if len(ids) < 2:
        return {"n_annotators": len(ids),
                "note": "Inter-annotator agreement needs at least 2 response files; "
                        "%d present. Every other metric family is unaffected."
                        % len(ids)}
    pairs = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            ra = {(r["suite"], r["dir"]): r for r in per_annotator[a] if not r["skipped"]}
            rb = {(r["suite"], r["dir"]): r for r in per_annotator[b] if not r["skipped"]}
            shared = sorted(set(ra) & set(rb))
            ious, angs, same_obj, same_dest = [], [], [], []
            for k in shared:
                fa = (ra[k].get("_fit"), rb[k].get("_fit"))
                if all(fa):
                    ious.append(ellipse_iou(fa[0], fa[1]))
                if ra[k].get("_ang") is not None and rb[k].get("_ang") is not None:
                    angs.append(abs(wrap180(ra[k]["_ang"] - rb[k]["_ang"])))
                # Which INSTANCE each annotator selected: nearest candidate to the
                # circle centre. More robust than containment, which can be empty.
                if ra[k].get("_sel") and rb[k].get("_sel"):
                    same_obj.append(ra[k]["_sel"] == rb[k]["_sel"])
                if ra[k].get("head_nearest") and rb[k].get("head_nearest"):
                    same_dest.append(ra[k]["head_nearest"] == rb[k]["head_nearest"])
            pairs["%s|%s" % (a, b)] = {
                "n_shared": len(shared),
                "circle_iou": summarise(ious),
                "arrow_angle_absdiff_deg": summarise(angs),
                "same_object_rate": (sum(same_obj) / len(same_obj)) if same_obj else None,
                "same_destination_rate": (sum(same_dest) / len(same_dest)) if same_dest else None,
            }
    return {"n_annotators": len(ids), "pairs": pairs}


def effort_block(rows):
    live = [r for r in rows if not r["skipped"]]
    return {
        "median_time_total_ms": nanmed([r.get("time_total_ms") for r in live]),
        "median_time_to_first_stroke_ms": nanmed([r.get("time_to_first_stroke_ms") for r in live]),
        "median_n_points_circle": nanmed([r.get("n_points_circle") for r in live]),
        "median_n_points_arrow": nanmed([r.get("n_points_arrow") for r in live]),
        "mean_n_undo": float(np.mean([r.get("n_undo") or 0 for r in live])) if live else None,
        "mean_n_redraw": float(np.mean([r.get("n_redraw") or 0 for r in live])) if live else None,
        "total_time_min": (sum((r.get("time_total_ms") or 0) for r in rows) / 60000.0)
                          if rows else None,
    }


# ------------------------------------------------------------------ rendering --
def render_human_sketch(truth, ann, out_path):
    """A 128x128 RGB PNG in the auto colour scheme, format-identical to sketch.png.

    Drawn at SS x and box-downsampled. A 4x box filter over a nearest-upscaled
    frame reproduces the original pixels exactly wherever no stroke was laid down,
    so the underlying frame is untouched while the strokes pick up the same kind of
    antialiasing cv2.LINE_AA gives the auto sketches."""
    frame = Image.open(os.path.join(truth["path"], "frame0.png")).convert("RGB")
    big = frame.resize((IMG_PX * SS, IMG_PX * SS), Image.NEAREST)
    dr = ImageDraw.Draw(big)

    circ = ann.get("circle")
    if circ and circ.get("points"):
        pts = [(p[0] * SS, p[1] * SS) for p in circ["points"]]
        dr.line(pts + [pts[0]], fill=CIRCLE_RGB, width=STROKE_W, joint="curve")

    arw = ann.get("arrow")
    if arw and arw.get("points"):
        pts = [(p[0] * SS, p[1] * SS) for p in arw["points"]]
        dr.line(pts, fill=ARROW_RGB, width=STROKE_W, joint="curve")
        hx, hy = pts[-1]
        bx, by = pts[0]
        for q in reversed(pts[:-1]):
            if math.hypot(hx - q[0], hy - q[1]) > 3 * SS:
                bx, by = q
                break
            bx, by = q
        a = math.atan2(hy - by, hx - bx)
        L, W = 6.5 * SS, 0.42
        dr.polygon([(hx, hy),
                    (hx - L * math.cos(a - W), hy - L * math.sin(a - W)),
                    (hx - L * math.cos(a + W), hy - L * math.sin(a + W))], fill=ARROW_RGB)

    small = big.resize((IMG_PX, IMG_PX), Image.BOX)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    small.save(out_path)
    return small


def human_tokens(truth, fit, arrow, extra):
    """A canonical tokens.json whose symbolic_tokens are the HUMAN geometry.

    Field-for-field the same contract as a scene's own tokens.json (SCHEMA.md), so
    a loader that reads the validation suites reads this without modification. The
    circle carries an extra `theta_deg` — the auto ellipse is axis-aligned and has
    no rotation to record — which is additive, like the legacy keys elsewhere."""
    meta = truth["meta"]
    tok = {k: meta[k] for k in ("instruction", "target", "destination",
                                "destination_region", "goal_predicate", "suite", "tier")}
    tok["schema_version"] = "1.0"
    tok["symbolic_tokens"] = {
        "circle": {"cx": round(fit["cx"], 3), "cy": round(fit["cy"], 3),
                   "rx": round(fit["rx"], 3), "ry": round(fit["ry"], 3),
                   "theta_deg": round(fit.get("theta_deg") or 0.0, 3)},
        "arrow": {"x0": round(float(arrow["x0"]), 3), "y0": round(float(arrow["y0"]), 3),
                  "x1": round(float(arrow["x1"]), 3), "y1": round(float(arrow["y1"]), 3)},
    }
    tok["radius"] = round((fit["rx"] + fit["ry"]) / 2.0, 3)
    tok["pick_px"] = meta["pick_px"]
    tok["place_px"] = meta["place_px"]
    tok["seed"] = meta["seed"]
    tok["sketch_source"] = "human"
    tok.update(extra)
    return tok


def export_scene(truth, ann, fit, arrow, out_dir, extra):
    """One scene folder mirroring outputs/validation_set_<suite>/<dir>/.

    Emits `sketch.png` and `tokens.json` only. `frame0.png`, `scene.bddl`,
    `target_vismask.png` and `meta.json` are NOT copied — they are identical to the
    originals and duplicating 36 BDDLs per annotator buys nothing. A consumer joins
    back on (suite, dir) for those, exactly as the scorer does."""
    os.makedirs(out_dir, exist_ok=True)
    img = render_human_sketch(truth, ann, os.path.join(out_dir, "sketch.png"))
    with open(os.path.join(out_dir, "tokens.json"), "w", encoding="utf-8") as fh:
        json.dump(human_tokens(truth, fit, arrow, extra), fh, indent=1)
    return img


def build_consensus(by_scene, truths, render_root, subset_seed):
    """One representative human sketch per scene, under <render_root>/consensus/.

    The consensus is the MEDOID, not a synthesised average: the real annotator whose
    geometry sits closest to the group median. An averaged ellipse and a straight
    averaged arrow would be a third stroke distribution, neither auto nor human,
    which is the one thing this export must not ship. Taking a real stroke keeps
    sketch.png and tokens.json consistent with each other and with what a person
    actually drew; the median is recorded alongside as `median_geometry` for
    anyone who wants it.

    Only annotators who agree on WHICH INSTANCE was circled vote. A tie means the
    scene is genuinely contested and no consensus is written for it."""
    made, contested = 0, []
    for key in sorted(by_scene):
        truth = truths[key]
        good = [(aid, ann, row) for aid, ann, row in by_scene[key]
                if not row["skipped"] and row.get("_fit") and ann.get("arrow")
                and row.get("_sel")]
        if not good:
            contested.append({"suite": key[0], "dir": key[1], "reason": "no usable annotation"})
            continue
        votes = Counter(row["_sel"] for _, _, row in good)
        top, n_top = votes.most_common(1)[0]
        if sum(1 for c in votes.values() if c == n_top) > 1:
            contested.append({"suite": key[0], "dir": key[1],
                              "reason": "tied vote on which object",
                              "votes": dict(votes)})
            continue
        grp = [g for g in good if g[2]["_sel"] == top]
        med = {k: float(np.median([g[2]["_fit"][k] for g in grp]))
               for k in ("cx", "cy", "rx", "ry")}
        med.update({k: float(np.median([float(g[1]["arrow"][k]) for g in grp]))
                    for k in ("x0", "y0", "x1", "y1")})

        def dist(g):
            f, a = g[2]["_fit"], g[1]["arrow"]
            return (math.hypot(f["cx"] - med["cx"], f["cy"] - med["cy"])
                    + abs((f["rx"] + f["ry"]) / 2 - (med["rx"] + med["ry"]) / 2)
                    + math.hypot(float(a["x1"]) - med["x1"], float(a["y1"]) - med["y1"]))

        aid, ann, row = min(grp, key=dist)
        out_dir = os.path.join(render_root, "consensus",
                               "validation_set_" + key[0], key[1])
        export_scene(truth, ann, row["_fit"], ann["arrow"], out_dir, {
            "sketch_source": "human_consensus",
            "consensus_method": "medoid",
            "consensus_medoid_annotator": aid,
            "consensus_from": sorted(g[0] for g in grp),
            "consensus_n_agreeing": len(grp),
            "consensus_n_annotators": len(good),
            "consensus_selected_object": top,
            "median_geometry": {k: round(v, 3) for k, v in med.items()},
            "subset_seed": subset_seed,
        })
        made += 1
    return made, contested


def contact_sheet(entries, out_path, title):
    """Auto sketch beside human sketch, one row per scene, for visual inspection."""
    n = len(entries)
    if n == 0:
        return
    cols = 4                                  # 4 scenes per row => 8 image columns
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 1.7, rows * 2.05),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (truth, human_img, res) in enumerate(entries):
        r, c = divmod(i, cols)
        auto = Image.open(os.path.join(truth["path"], "sketch.png")).convert("RGB")
        for k, (im, lab) in enumerate(((auto, "auto"), (human_img, "human"))):
            ax = axes[r][c * 2 + k]
            ax.imshow(np.asarray(im), interpolation="nearest")
            ax.axis("off")
            ax.set_title(lab, fontsize=6, pad=1.5)
        flag = "OK" if res.get("joint_ok") else "x"
        axes[r][c * 2].text(0, -18, "%s/%s  %s" % (truth["suite"][:4], truth["dir"][-4:], flag),
                            fontsize=5.5, color="#333")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# -------------------------------------------------------------------- figures --
def fig_accuracy(metrics, out_path):
    groups = [("pooled", metrics["accuracy"]["pooled"])]
    groups += [("suite: " + k, v) for k, v in metrics["accuracy"]["by_suite"].items()]
    groups += [("tier: " + k, v) for k, v in metrics["accuracy"]["by_tier"].items()]
    labels = [g[0] for g in groups]
    series = [("referential", "#2f7d3a"), ("directional", "#a32020"), ("joint", "#2b3a67")]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(1.05 * len(groups) + 3.2, 4.0))
    w = 0.26
    for i, (name, col) in enumerate(series):
        vals = [(g[1][name]["rate"] or 0) * 100 for g in groups]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=col)
        for b, v, g in zip(bars, vals, groups):
            if g[1][name]["n"]:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.2, "%.0f" % v,
                        ha="center", fontsize=7)
    sk = [(g[1]["skip_rate"] or 0) * 100 for g in groups]
    ax.plot(x, sk, "o--", color="#888", ms=4, lw=1, label="skip rate")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("%"); ax.set_ylim(0, 108)
    ax.set_title("(a) Human correctness — can a human supply the disambiguating signal?",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="lower right", framealpha=.9)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)


def fig_calibration(cal, out_path):
    names = list(AUTO_JITTER)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    for ax, name in zip(axes.ravel(), names):
        blk = cal[name]
        h = blk.get("_values", [])
        if not len(h):
            ax.text(.5, .5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name, fontsize=9); ax.axis("off"); continue
        ax.hist(h, bins=28, color="#4a6fa5", alpha=.85, density=True,
                label="human (n=%d)" % len(h))
        lo, hi = blk["auto_lo"], blk["auto_hi"]
        ax.axvspan(lo, hi, color="#e0a800", alpha=.22,
                   label="auto range [%.3g, %.3g]" % (lo, hi))
        ax.axvline(lo, color="#b07d00", lw=1); ax.axvline(hi, color="#b07d00", lw=1)
        rec = blk.get("recommended", {})
        if "lo" in rec:
            ax.axvline(rec["lo"], color="#2f7d3a", lw=1.4, ls="--")
            ax.axvline(rec["hi"], color="#2f7d3a", lw=1.4, ls="--",
                       label="recommended [%g, %g]" % (rec["lo"], rec["hi"]))
        ax.set_title("%s\n%s — %.0f%% of human strokes outside"
                     % (name, blk["verdict"].upper(), 100 * blk["frac_outside"]),
                     fontsize=9)
        ax.set_xlabel(blk["label"], fontsize=7.5)
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle("(c) Is the auto imprecision augmentation wide enough for real human strokes?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=170); plt.close(fig)


def fig_geometry(rows, out_path):
    """Both readings on the same axes: every stroke, and only the strokes that
    point at the auto drawer's instance. Where the two histograms separate, the
    separation is scene ambiguity, not hand wobble."""
    live = [r for r in rows if not r["skipped"]]
    panels = [("circle_centre_offset_px", "circle centre offset (px)"),
              ("circle_radius_ratio", "radius ratio human/auto"),
              ("circle_iou", "circle ellipse IoU"),
              ("arrow_tail_offset_px", "arrow tail offset (px)"),
              ("arrow_head_offset_px", "arrow head offset (px)"),
              ("arrow_angle_absdiff_deg", "|arrow angle difference| (deg)"),
              ("arrow_length_ratio", "arrow length ratio human/auto"),
              ("arrow_curvature_px", "arrow deviation from chord (px)")]

    def vals(src, key):
        return [r[key] for r in src
                if r.get(key) is not None and not math.isnan(r[key])]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, (key, lab) in zip(axes.ravel(), panels):
        gate = "referential_ok" if key.startswith("circle_") else "directional_ok"
        v_all = vals(live, key)
        v_ok = vals([r for r in live if r.get(gate)], key)
        if not v_all:
            ax.text(.5, .5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(lab, fontsize=8.5); continue
        bins = np.histogram_bin_edges(v_all, bins=20)
        ax.hist(v_all, bins=bins, color="#b9c4d6", alpha=.95,
                label="all n=%d" % len(v_all))
        if v_ok:
            ax.hist(v_ok, bins=bins, color="#4a6fa5", alpha=.9,
                    label="same referent n=%d" % len(v_ok))
            ax.axvline(float(np.median(v_ok)), color="#a32020", lw=1.4,
                       label="median %.2f" % float(np.median(v_ok)))
        ax.axvline(float(np.median(v_all)), color="#a32020", lw=1.1, ls=":",
                   label="median all %.2f" % float(np.median(v_all)))
        if key in ("circle_radius_ratio", "arrow_length_ratio"):
            ax.axvline(1.0, color="#2f7d3a", lw=1, ls="--", label="parity")
        ax.set_title(lab, fontsize=8.5); ax.legend(fontsize=6.5); ax.grid(alpha=.2)
    fig.suptitle("(b) Human stroke vs the auto symbolic_tokens — dark = strokes that "
                 "point at the same instance the auto drawer did", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=170); plt.close(fig)


def fig_effort(rows, out_path):
    live = [r for r in rows if not r["skipped"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    t = [(r.get("time_total_ms") or 0) / 1000.0 for r in live if r.get("time_total_ms")]
    if t:
        axes[0].hist(t, bins=20, color="#4a6fa5", alpha=.85)
        axes[0].axvline(float(np.median(t)), color="#a32020", lw=1.4,
                        label="median %.1f s" % float(np.median(t)))
        axes[0].legend(fontsize=7)
    axes[0].set_title("time per scene (s)", fontsize=9); axes[0].grid(alpha=.2)

    order = [x for x in TIERS if any(r["tier"] == x for r in live)]
    data = [[(r.get("time_total_ms") or 0) / 1000.0 for r in live if r["tier"] == x]
            for x in order]
    if any(data):
        axes[1].boxplot(data, tick_labels=order)
    axes[1].set_title("time per scene by tier (s)", fontsize=9)
    axes[1].tick_params(labelsize=8); axes[1].grid(alpha=.2)

    counts = {"undo": [r.get("n_undo") or 0 for r in live],
              "redraw": [r.get("n_redraw") or 0 for r in live]}
    axes[2].bar(list(counts), [float(np.mean(v)) if v else 0 for v in counts.values()],
                color=["#4a6fa5", "#e0a800"], width=.5)
    axes[2].set_title("mean corrections per scene", fontsize=9); axes[2].grid(alpha=.2)
    fig.suptitle("(d) Effort — the low-effort claim, measured", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=170); plt.close(fig)


# ---------------------------------------------------------------------- main --
CSV_COLS = ["annotator", "suite", "dir", "tier", "instruction", "skipped", "skip_reason",
            "dest_from_place_px", "has_circle", "has_arrow", "contained",
            "contains_target", "referential_ok", "referential_fail",
            "head_nearest", "directional_ok", "head_to_dest_px", "joint_ok",
            "circle_centre_offset_px", "circle_centre_offset_norm", "circle_radius_ratio",
            "circle_iou", "circle_theta_deg",
            "arrow_tail_offset_px", "arrow_head_offset_px", "arrow_angle_diff_deg",
            "arrow_angle_absdiff_deg", "arrow_length_ratio", "arrow_curvature_px",
            "j_circle_dx", "j_circle_dy", "j_radius_dx", "j_radius_dy",
            "j_arrow_x0", "j_arrow_y0", "j_arrow_x1", "j_arrow_y1",
            "time_to_first_stroke_ms", "time_total_ms", "n_undo", "n_redraw",
            "n_points_circle", "n_points_arrow"]


def main():
    ap = argparse.ArgumentParser(description="Score human sketches against the auto ones.")
    ap.add_argument("--smoke", action="store_true",
                    help="score outputs/human_study/smoke/ instead of the full study")
    ap.add_argument("--study-dir", default=None)
    args = ap.parse_args()

    study = args.study_dir or (os.path.join(STUDY_DIR, "smoke") if args.smoke else STUDY_DIR)
    resp_dir = os.path.join(study, "responses")
    out_dir = os.path.join(study, "comparison")
    render_root = os.path.join(study, "rendered")
    os.makedirs(out_dir, exist_ok=True)

    print("Sketch-Prompted VLA - human vs auto sketch comparison")
    print("  study dir    : %s" % study)

    subset_path = os.path.join(study, "scene_subset.json")
    subset = None
    if os.path.isfile(subset_path):
        with open(subset_path, encoding="utf-8") as fh:
            subset = json.load(fh)
        print("  roster       : %d scenes, seed %d, generated %s"
              % (subset["n_scored"], subset["subset_seed"], subset["generated_utc"]))

    files = sorted(f for f in os.listdir(resp_dir) if f.endswith(".json")) \
        if os.path.isdir(resp_dir) else []
    if not files:
        sys.exit("no response files in %s — nothing to score" % resp_dir)

    truths, per_annotator, all_rows = {}, {}, []
    render_index = defaultdict(list)
    by_scene = defaultdict(list)

    for fn in files:
        with open(os.path.join(resp_dir, fn), encoding="utf-8") as fh:
            doc = json.load(fh)
        aid = doc.get("annotator_id") or os.path.splitext(fn)[0]
        if subset and doc.get("subset_seed") != subset["subset_seed"]:
            print("  ! %s was collected on seed %s, roster is %s — scoring anyway"
                  % (fn, doc.get("subset_seed"), subset["subset_seed"]))
        rows, n_practice = [], 0
        for ann in doc.get("annotations", []):
            if ann.get("practice"):
                n_practice += 1
                continue
            ann["_annotator"] = aid
            key = (ann["suite"], ann["dir"])
            if key not in truths:
                truths[key] = load_truth(*key)
            t = truths[key]
            r = score_one(ann, t)
            # Carried for (d) only; not written to the CSV.
            r["_fit"] = (ann.get("circle") or {}).get("fit")
            r["_ang"] = (ang_deg(ann["arrow"]["x0"], ann["arrow"]["y0"],
                                 ann["arrow"]["x1"], ann["arrow"]["y1"])
                         if ann.get("arrow") else None)
            if r.get("_fit"):
                c = (r["_fit"]["cx"], r["_fit"]["cy"])
                r["_sel"] = min(t["candidates"].items(),
                                key=lambda kv: math.hypot(kv[1][0] - c[0], kv[1][1] - c[1]))[0]
            rows.append(r)
            by_scene[key].append((aid, ann, r))
            if not r["skipped"] and r.get("_fit") and ann.get("arrow"):
                img = export_scene(
                    t, ann, r["_fit"], ann["arrow"],
                    os.path.join(render_root, aid, "validation_set_" + t["suite"], t["dir"]),
                    {"annotator_id": aid, "subset_seed": doc.get("subset_seed")})
                render_index[aid].append((t, img, r))
        per_annotator[aid] = rows
        all_rows.extend(rows)
        print("  %-12s : %3d scored, %d practice, %d skipped, %d rendered"
              % (aid, len(rows), n_practice, sum(r["skipped"] for r in rows),
                 len(render_index[aid])))

    if subset:
        want = {(s["suite"], s["dir"]) for s in subset["scenes"]}
        for aid, rows in per_annotator.items():
            got = {(r["suite"], r["dir"]) for r in rows}
            if got - want:
                print("  ! %s annotated %d scene(s) not on the roster" % (aid, len(got - want)))
            if want - got:
                print("  ! %s is missing %d roster scene(s)" % (aid, len(want - got)))
        for s in subset["scenes"]:
            t = truths.get((s["suite"], s["dir"]))
            if t and t["target"] != s["truth"]["target"]:
                sys.exit("roster/disk truth mismatch at %s/%s" % (s["suite"], s["dir"]))

    # ---------------------------------------------------------------- metrics --
    n_consensus, contested = build_consensus(
        by_scene, truths, render_root, subset["subset_seed"] if subset else None)
    print("  consensus    : %d scene(s) written, %d contested" % (n_consensus, len(contested)))
    for c in contested:
        print("                 %s/%s — %s" % (c["suite"], c["dir"], c["reason"]))

    cal = calibration(all_rows, truths)
    geo_keys = ["circle_centre_offset_px", "circle_centre_offset_norm",
                "circle_radius_ratio", "circle_iou", "arrow_tail_offset_px",
                "arrow_head_offset_px", "arrow_angle_diff_deg",
                "arrow_angle_absdiff_deg", "arrow_length_ratio", "arrow_curvature_px"]

    def geo_block(rows):
        """(b) reported TWICE, because the two readings answer different questions.

        `all` pools every non-skipped stroke. On a scene whose target has a
        same-category clone, a human who circled the other instance contributes
        the distance BETWEEN TWO OBJECTS, not their stroke imprecision — 26 px on
        goal/scene_0026, against a real hand wobble of 2-3 px. Pooled, that reads
        as human imprecision and it is not.

        `same_referent` keeps only the strokes that point at the same instance
        the auto drawer did, so the residual is stroke noise alone. This is the
        one to quote for imprecision, and it is the same basis calibration (c)
        uses.

        The filter is applied PER FAMILY, not jointly: circle metrics gate on
        `referential_ok`, arrow metrics on `directional_ok`. Circle and arrow
        correctness are independent — an annotator can circle the intended bowl
        and still send the arrow to the other plate — so a joint gate would throw
        away a perfectly good arrow because of its circle, and vice versa.

        The gap between the two blocks is itself the quantity of interest: it
        measures how far apart the ambiguous scenes' candidate instances are, not
        how badly anyone draws."""
        rr = [r for r in rows if not r["skipped"]]
        ok = {"circle": [r for r in rr if r.get("referential_ok")],
              "arrow": [r for r in rr if r.get("directional_ok")]}

        def block(matched):
            out = {}
            for k in geo_keys:
                fam = "circle" if k.startswith("circle_") else "arrow"
                src = ok[fam] if matched else rr
                out[k] = summarise([r.get(k) for r in src])
            return out

        return {"same_referent": block(True), "all": block(False),
                "n_live": len(rr),
                "n_same_referent": {"circle": len(ok["circle"]), "arrow": len(ok["arrow"])},
                "n_different_referent": {"circle": len(rr) - len(ok["circle"]),
                                         "arrow": len(rr) - len(ok["arrow"])}}

    metrics = {
        "schema_version": "1.0",
        "study_dir": os.path.relpath(study, _REPO).replace("\\", "/"),
        "subset_seed": subset["subset_seed"] if subset else None,
        "n_annotators": len(per_annotator),
        "annotators": sorted(per_annotator),
        "n_rows": len(all_rows),
        "n_scenes_touched": len(truths),
        "n_dest_from_place_px": sum(1 for t in truths.values() if t["dest_injected"]),
        "auto_jitter_under_test": {k: {"lo": v["lo"], "hi": v["hi"], "kind": v["kind"]}
                                   for k, v in AUTO_JITTER.items()},
        "accuracy": {
            "pooled": accuracy_block(all_rows),
            "by_suite": by_group(all_rows, lambda r: r["suite"], accuracy_block),
            "by_tier": by_group(all_rows, lambda r: r["tier"], accuracy_block),
            "by_annotator": {a: accuracy_block(r) for a, r in sorted(per_annotator.items())},
        },
        "geometry": {
            "pooled": geo_block(all_rows),
            "by_suite": by_group(all_rows, lambda r: r["suite"], geo_block),
            "by_tier": by_group(all_rows, lambda r: r["tier"], geo_block),
        },
        "calibration": {k: {kk: vv for kk, vv in v.items() if kk != "_values"}
                        for k, v in cal.items()},
        "agreement": agreement(per_annotator, truths),
        "consensus": {"method": "medoid", "n_scenes": n_consensus,
                      "contested": contested,
                      "path": os.path.relpath(os.path.join(render_root, "consensus"),
                                              _REPO).replace("\\", "/")},
        "effort": {
            "pooled": effort_block(all_rows),
            "by_tier": by_group(all_rows, lambda r: r["tier"], effort_block),
            "by_annotator": {a: effort_block(r) for a, r in sorted(per_annotator.items())},
        },
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)

    with open(os.path.join(out_dir, "per_scene.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(all_rows, key=lambda r: (r["annotator"], r["suite"], r["dir"])):
            w.writerow({k: r.get(k) for k in CSV_COLS})

    fig_accuracy(metrics, os.path.join(out_dir, "fig_accuracy.png"))
    fig_calibration(cal, os.path.join(out_dir, "fig_calibration.png"))
    fig_geometry(all_rows, os.path.join(out_dir, "fig_geometry.png"))
    fig_effort(all_rows, os.path.join(out_dir, "fig_effort.png"))
    for aid, entries in render_index.items():
        contact_sheet(sorted(entries, key=lambda e: (e[0]["suite"], e[0]["dir"])),
                      os.path.join(out_dir, "contact_sheet_%s.png" % aid),
                      "auto vs human — annotator %s" % aid)

    # ----------------------------------------------------------------- stdout --
    p = metrics["accuracy"]["pooled"]
    print("\n(a) human correctness, pooled over %d scored scenes" % p["n_scenes"])
    for k in ("referential", "directional", "joint"):
        b = p[k]
        print("      %-13s %s  (n=%d)"
              % (k, "%5.1f%%" % (100 * b["rate"]) if b["rate"] is not None else "   n/a", b["n"]))
    print("      skipped       %5.1f%%  %s" % (100 * (p["skip_rate"] or 0),
                                               p["skip_reasons"] or ""))
    if p["referential_failures"]:
        print("      ref failures  %s" % p["referential_failures"])
    print("    by tier:")
    for tier, b in metrics["accuracy"]["by_tier"].items():
        print("      %-12s ref %s  dir %s  joint %s  (n=%d)"
              % (tier,
                 "%5.1f%%" % (100 * b["referential"]["rate"]) if b["referential"]["rate"] is not None else "  n/a",
                 "%5.1f%%" % (100 * b["directional"]["rate"]) if b["directional"]["rate"] is not None else "  n/a",
                 "%5.1f%%" % (100 * b["joint"]["rate"]) if b["joint"]["rate"] is not None else "  n/a",
                 b["n_scenes"]))

    g = metrics["geometry"]["pooled"]
    print("\n(b) human vs auto geometry, medians")
    print("    same_referent = strokes pointing at the auto drawer's instance; this is")
    print("    the imprecision figure. `all` also contains strokes that resolved the")
    print("    ambiguity differently, whose offset is the object-to-object distance.")
    print("      %-28s %9s %9s" % ("", "same_ref", "all"))
    for k in geo_keys:
        s, t = g["same_referent"][k], g["all"][k]
        if not (s.get("n") or t.get("n")):
            continue
        print("      %-28s %9s %9s"
              % (k,
                 "%.2f" % s["median"] if s.get("n") else "n/a",
                 "%.2f" % t["median"] if t.get("n") else "n/a"))
    print("      n: circle %d/%d same referent, arrow %d/%d"
          % (g["n_same_referent"]["circle"], g["n_live"],
             g["n_same_referent"]["arrow"], g["n_live"]))

    print("\n(c) augmentation calibration verdict")
    for name, blk in metrics["calibration"].items():
        if not blk["human"].get("n"):
            continue
        rec = blk.get("recommended", {})
        print("      %-18s %-11s  %5.1f%% of human strokes outside the auto range"
              % (name, blk["verdict"].upper(), 100 * blk["frac_outside"]))
        print("      %-18s current %s -> recommended %s"
              % ("", _fmt_range(name, blk), rec.get("as_code", "n/a")))
        rb = blk.get("recommended_robust")
        if rb:
            print("      %-18s robust      %-24s (dropped %d/%d = %.0f%% beyond %.1f MAD-SD)"
                  % ("", rb["as_code"], blk["robust"]["n_dropped"],
                     blk["robust"]["n_kept"] + blk["robust"]["n_dropped"],
                     100 * blk["robust"]["frac_dropped"], ROBUST_K))
        print("      %-18s basis: %d correct scene(s), %d excluded as incorrect"
              % ("", blk["n_scenes_in_basis"], blk["n_scenes_excluded_as_incorrect"]))

    ag = metrics["agreement"]
    print("\n(d) agreement and effort")
    if "note" in ag:
        print("      %s" % ag["note"])
    else:
        for pair, b in ag["pairs"].items():
            print("      %-14s n=%2d  circle IoU %.2f  |angle| %.1f deg  "
                  "same object %.0f%%  same dest %.0f%%"
                  % (pair, b["n_shared"], b["circle_iou"].get("median", float("nan")),
                     b["arrow_angle_absdiff_deg"].get("median", float("nan")),
                     100 * (b["same_object_rate"] or 0), 100 * (b["same_destination_rate"] or 0)))
    e = metrics["effort"]["pooled"]
    print("      median %.1f s per scene, %.1f s to first stroke, "
          "%.2f undo and %.2f redraw per scene"
          % ((e["median_time_total_ms"] or 0) / 1000.0,
             (e["median_time_to_first_stroke_ms"] or 0) / 1000.0,
             e["mean_n_undo"] or 0, e["mean_n_redraw"] or 0))

    print("\n  wrote        : %s" % os.path.relpath(out_dir, _REPO).replace("\\", "/"))
    print("                 metrics.json, per_scene.csv, 4 figures, %d contact sheet(s)"
          % len(render_index))
    print("  exported     : %d per-annotator scene folder(s) + %d consensus, under %s"
          % (sum(len(v) for v in render_index.values()), n_consensus,
             os.path.relpath(render_root, _REPO).replace("\\", "/")))
    print("                 each holds sketch.png + tokens.json; join to the original "
          "scene on (suite, dir)")


def _fmt_range(name, blk):
    if name == "arrow_bend_px":
        return "rng.integers(-%d, %d)" % (BEND_HALFWIDTH, BEND_HALFWIDTH + 1)
    spec = AUTO_JITTER[name]
    if spec["kind"] == "int":
        return "rng.integers(%g, %g)" % (spec["lo"], spec["hi"] + 1)
    return "rng.uniform(%g, %g)" % (spec["lo"], spec["hi"])


if __name__ == "__main__":
    main()
