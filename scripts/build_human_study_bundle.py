"""
Sketch-Prompted VLA — human sketch-collection bundle builder (pure stdlib; runs on
any machine, no robosuite / mujoco / libero).

Samples a stratified, seeded subset of the 114 validation scenes and emits a
SINGLE self-contained HTML file that a human annotator opens off `file://` and
draws on. Two artefacts:

    outputs/human_study/scene_subset.json   the roster + the ground truth the scorer needs
    outputs/human_study/sketch_tool.html    the annotator-facing artefact

The two are deliberately asymmetric. `scene_subset.json` carries the full truth
(`target`, `destination`, `pick_px`, `place_px`, `symbolic_tokens`,
`all_pixels`); `sketch_tool.html` carries ONLY the base64-embedded `frame0.png`,
the vague instruction, the suite and the scene id. Nothing that identifies the
intended object or destination — not even the tier, which would tell the
annotator whether the scene is ambiguous — reaches the browser. If it did, the
referential- and directional-accuracy numbers the scorer reports would be
measuring the annotator's reading comprehension rather than their perception.
`score_human_sketches.py` rejoins truth from disk on `(suite, dir)`.

Frames are base64-embedded because a `file://` page cannot `fetch()` sibling
PNGs, and requiring `python -m http.server` defeats the point of handing one
file to a collaborator. Thirty-seven 128x128 PNGs cost about 1.1 MB.

    cd C:\\Users\\Admin\\sketch_prompted_vla
    python scripts/build_human_study_bundle.py            # stratified 36-scene bundle
    python scripts/build_human_study_bundle.py --smoke    # 3-scene vertical slice
    python scripts/build_human_study_bundle.py --all      # census: 113 scored + 1 practice

`--all` writes to `outputs/human_study/full114/` rather than over the sampled
bundle, so both rosters can exist side by side and a response file can never be
scored against the wrong one. Score it with
`score_human_sketches.py --study-dir outputs/human_study/full114`.
"""

import argparse
import base64
import collections
import datetime
import json
import os
import random
import sys

# ----------------------------------------------------------------- constants --
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rename-proof
ROOT = os.path.join(_REPO, "outputs")
OUT_DIR = os.path.join(ROOT, "human_study")

SUITES = {
    "spatial": "validation_set_spatial",
    "object":  "validation_set_object",
    "goal":    "validation_set_goal",
}

# The sample seed. Recorded in scene_subset.json and echoed into every annotator
# response file, so a response can always be traced back to the roster it answers.
SUBSET_SEED = 20260802

# 12 scenes per suite, allocated across the four tiers roughly in proportion to
# each suite's control 5 / referential 12 / directional 9 / both 12 split.
# Exact proportional allocation is 1.58 / 3.79 / 2.84 / 3.79; largest-remainder
# rounding would give 1 control. Control is bumped to 2 at the expense of `both`
# because control scenes are the only ones where a wrong circle is unambiguously
# annotator error rather than genuine scene ambiguity — they are the calibration
# baseline for every other tier, and one per suite is too thin to read.
N_PER_SUITE = 12
TIER_QUOTA = {"control": 2, "referential": 4, "directional": 3, "both": 3}

# Vertical slice: one scene per suite, control tier, so the round trip can be
# checked end-to-end before generating the full bundle.
SMOKE_QUOTA = {"control": 1, "referential": 0, "directional": 0, "both": 0}
SMOKE_N_PER_SUITE = 1

# Census mode (`--all`): every scene on disk, no sampling. 113 scored + 1
# practice held out, because the practice scene must stay outside the scored set
# and there is no "outside" once the census is taken. Written to its own
# directory so it cannot clobber the sampled 36-scene bundle, whose roster other
# annotators may already hold — see the note in HUMAN_STUDY.md on why every
# annotator must receive the same build.
ALL_SUBDIR = "full114"

# Match the auto drawer exactly (build_validation_set_*.py, draw_circle /
# draw_arrow) so a human sketch is a drop-in replacement for an auto one.
CIRCLE_RGB = (0, 200, 0)
ARROW_RGB = (200, 50, 50)

IMG_PX = 128
DISPLAY_SCALE = 4          # 128 -> 512, image-rendering: pixelated

# Truth fields lifted out of meta.json into scene_subset.json. The scorer reads
# meta.json from disk as the authority and cross-checks it against these.
TRUTH_FIELDS = ("target", "destination", "destination_region", "goal_predicate",
                "pick_px", "place_px", "radius", "symbolic_tokens",
                "all_pixels", "px_extent", "seed")

# Verbatim annotator-facing copy. Lives here rather than in the template so that
# HUMAN_STUDY.md can quote it and this file stays the single source of truth.
INTRO_HTML = """
<p>You are looking at what a robot's camera sees: a small 128&times;128 photo of a
table or floor with a few objects on it, shown here blown up four times so the
individual pixels are visible. It will look blocky. That is the real resolution
the robot works from, so it is the resolution you should judge from too.</p>

<p>Underneath each picture is a short instruction, such as
&ldquo;put the bowl on the plate&rdquo;. The instruction is deliberately vague.
Often there will be several bowls, or several plates, and the words alone will
not tell you which one is meant. That is the point of the study.</p>

<p><strong>Your job is to answer the instruction with two marks:</strong></p>
<ol>
  <li><strong>Draw a circle</strong> around the one object you would pick up.</li>
  <li><strong>Draw an arrow</strong> from that object to the place you would put it.
      Start the arrow at the object and finish it where the object should end up
      &mdash; the arrowhead appears wherever you lift your finger or mouse.</li>
</ol>

<p>There is no correct answer to look up and nothing is being tested about you.
If the instruction is ambiguous, pick whichever object or destination seems most
natural to you and move on. Draw the way you would draw on a whiteboard: quickly
and roughly. A wobbly circle is a good circle. Do not zoom in, count pixels or
deliberate &mdash; how fast and how loosely people do this is one of the things
being measured.</p>

<p>If you truly cannot tell what is being asked, press <strong>Skip</strong> and
say why. A skip is a useful result, not a failure; it tells me the picture alone
is not enough for that scene.</p>

<p>There are __N_SCENES__ scenes, preceded by one practice scene to get the feel
of the controls. Your progress is saved in this browser after every scene, so you
can close the tab and come back. At the end you will get a
<strong>Download JSON</strong> button &mdash; send me that file.</p>

<p>Keyboard: <kbd>c</kbd> circle tool, <kbd>a</kbd> arrow tool,
<kbd>u</kbd> undo, <kbd>Enter</kbd> next scene.</p>
"""

SKIP_REASONS = [
    "cannot tell which object",
    "cannot tell where",
    "image unclear",
    "other",
]


# -------------------------------------------------------------------- helpers --
def load_scenes():
    """Read every scene's meta.json from disk. meta.json is the authority; the
    combined manifest carries no `instruction`."""
    scenes = []
    for suite, subdir in SUITES.items():
        suite_root = os.path.join(ROOT, subdir)
        if not os.path.isdir(suite_root):
            sys.exit("missing suite directory: %s" % suite_root)
        dirs = sorted(d for d in os.listdir(suite_root) if d.startswith("scene_"))
        for d in dirs:
            with open(os.path.join(suite_root, d, "meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            scenes.append({"suite": suite, "dir": d, "tier": meta["tier"],
                           "instruction": meta["instruction"], "meta": meta,
                           "path": os.path.join(suite_root, d)})
    return scenes


def stratified_sample(scenes, quota, n_per_suite, rng):
    """Draw `quota` scenes of each tier from each suite. Sampling is done on a
    sorted candidate list so the result depends only on the seed."""
    chosen = []
    for suite in SUITES:
        pool = [s for s in scenes if s["suite"] == suite]
        got = []
        for tier, k in quota.items():
            if k == 0:
                continue
            cand = sorted((s for s in pool if s["tier"] == tier), key=lambda s: s["dir"])
            if len(cand) < k:
                sys.exit("suite %s has %d %s scenes, need %d" % (suite, len(cand), tier, k))
            got.extend(rng.sample(cand, k))
        if len(got) != n_per_suite:
            sys.exit("suite %s: quota sums to %d, expected %d" % (suite, len(got), n_per_suite))
        chosen.extend(got)
    return chosen


def pick_practice(scenes, chosen, rng):
    """A control-tier scene from OUTSIDE the subset. Control tier because the
    practice scene should teach the controls without also teaching the annotator
    that ambiguity is expected; outside the subset so no scored scene carries a
    warm-up annotation."""
    taken = {(s["suite"], s["dir"]) for s in chosen}
    cand = sorted((s for s in scenes
                   if (s["suite"], s["dir"]) not in taken and s["tier"] == "control"),
                  key=lambda s: (s["suite"], s["dir"]))
    if not cand:
        sys.exit("no control-tier scene left outside the subset for practice")
    return rng.choice(cand)


def frame_b64(scene):
    with open(os.path.join(scene["path"], "frame0.png"), "rb") as fh:
        raw = fh.read()
    if not raw.endswith(b"IEND\xaeB`\x82"):
        sys.exit("truncated PNG (no IEND): %s" % scene["path"])
    return base64.b64encode(raw).decode("ascii")


def annotator_view(scene, order):
    """Exactly what reaches the browser. Audited by the verification block below."""
    return {"order": order, "suite": scene["suite"], "dir": scene["dir"],
            "instruction": scene["instruction"], "frame_b64": frame_b64(scene)}


def truth_view(scene, order):
    meta = scene["meta"]
    truth = {k: meta[k] for k in TRUTH_FIELDS if k in meta}
    # 11 Goal scenes have a fixture/region destination (flat_stove_1, wine_rack_1,
    # wooden_cabinet_1, main_table) that is NOT a key in all_pixels, because
    # all_pixels enumerates movable instances only. The scorer's nearest-candidate
    # directional metric would be undefined there, so it injects the destination at
    # place_px. Flagged here so that injection is visible in the roster rather than
    # buried in the scorer.
    truth["dest_in_all_pixels"] = meta["destination"] in meta["all_pixels"]
    return {"order": order, "suite": scene["suite"], "dir": scene["dir"],
            "tier": scene["tier"], "instruction": scene["instruction"],
            "truth": truth}


# ------------------------------------------------------------- HTML template --
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sketch-Prompted VLA &mdash; sketch collection</title>
<style>
  :root {
    --circle: rgb(0,200,0);
    --arrow: rgb(200,50,50);
    --ink: #16181d;
    --muted: #5d6470;
    --line: #d8dbe2;
    --bg: #f6f7f9;
    --panel: #ffffff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 980px; margin: 0 auto; padding: 28px 22px 60px; }
  h1 { font-size: 21px; margin: 0 0 4px; letter-spacing: -0.01em; }
  h2 { font-size: 16px; margin: 26px 0 8px; }
  .sub { color: var(--muted); font-size: 13px; margin: 0 0 22px; }
  .panel {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 22px 24px;
  }
  ol, ul { padding-left: 22px; }
  kbd {
    font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: #eceef2; border: 1px solid var(--line); border-bottom-width: 2px;
    border-radius: 4px; padding: 1px 5px;
  }
  button {
    font: inherit; font-size: 14px; padding: 7px 14px; border-radius: 7px;
    border: 1px solid var(--line); background: #fff; color: var(--ink);
    cursor: pointer;
  }
  button:hover:not(:disabled) { background: #eef0f4; }
  button:disabled { opacity: .4; cursor: default; }
  button.primary { background: var(--ink); color: #fff; border-color: var(--ink); }
  button.primary:hover:not(:disabled) { background: #333842; }
  button.tool[aria-pressed="true"] { border-width: 2px; font-weight: 600; }
  button#tool-circle[aria-pressed="true"] { border-color: var(--circle); color: #0a6b26; }
  button#tool-arrow[aria-pressed="true"] { border-color: var(--arrow); color: #a32020; }
  input[type=text] {
    font: inherit; padding: 7px 10px; border: 1px solid var(--line);
    border-radius: 7px; width: 260px;
  }
  select { font: inherit; padding: 6px 8px; border: 1px solid var(--line); border-radius: 7px; }

  .stage { display: flex; gap: 26px; align-items: flex-start; flex-wrap: wrap; }
  #canvas {
    width: 512px; height: 512px; display: block;
    image-rendering: pixelated; image-rendering: crisp-edges;
    border: 1px solid var(--line); border-radius: 6px;
    background: #fff; touch-action: none; cursor: crosshair;
  }
  .side { flex: 1; min-width: 260px; }
  .instruction {
    font-size: 19px; font-weight: 600; margin: 0 0 4px; line-height: 1.35;
  }
  .scene-id { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
  .progress { color: var(--muted); font-size: 13px; margin: 0 0 16px; font-variant-numeric: tabular-nums; }
  .status { margin: 14px 0 0; font-size: 13px; min-height: 20px; }
  .status .ok { color: #0a6b26; }
  .status .todo { color: var(--muted); }
  .status .warn { color: #a32020; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
            vertical-align: baseline; margin-right: 5px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0 0; }
  .row.tight { margin-top: 8px; }
  hr { border: 0; border-top: 1px solid var(--line); margin: 20px 0; }
  /* The `hidden` attribute is only honoured by the UA rule `[hidden]{display:none}`,
     which loses to ANY author rule that sets `display` on the same element — class
     selectors outrank the UA sheet. `.practice-flag` sets `display:inline-block`,
     so `el.hidden = true` had no visual effect and the practice badge showed on
     every scored scene. Restated with !important so the attribute wins wherever
     the JS toggles it. Cosmetic only: the exported `practice` flag is read from
     the bundle, never from the badge, so no response file was ever mislabelled. */
  [hidden] { display: none !important; }
  .practice-flag {
    display: inline-block; background: #fff4d6; border: 1px solid #e5cf94;
    color: #7a5b0a; border-radius: 5px; padding: 1px 7px; font-size: 12px;
    font-weight: 600; margin-bottom: 6px;
  }
  .skipbox { margin-top: 12px; display: none; }
  .skipbox.on { display: block; }
  code { font: 12.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
</style>
</head>
<body>
<div class="wrap">

<!-- ------------------------------------------------------------- intro --- -->
<section id="screen-intro">
  <h1>Sketch-Prompted VLA &mdash; sketch collection</h1>
  <p class="sub">Circle the object, arrow to the destination. About __EST_MIN__ minutes.</p>
  <div class="panel">
    __INTRO_HTML__
    <hr>
    <p><label for="annotator">Your name</label> &mdash; Latin letters and spaces.
    Accents are stripped automatically, so <em>Jos&eacute;</em> is fine to type.</p>
    <div class="row tight">
      <input type="text" id="annotator" placeholder="e.g. Alex Smith" autocomplete="name">
      <button class="primary" id="btn-start">Start</button>
    </div>
    <p class="status" id="intro-status"></p>
  </div>
</section>

<!-- -------------------------------------------------------------- draw --- -->
<section id="screen-draw" hidden>
  <div class="stage">
    <div>
      <canvas id="canvas" width="512" height="512"></canvas>
      <div class="row tight">
        <button class="tool" id="tool-circle" aria-pressed="true">Circle <kbd>c</kbd></button>
        <button class="tool" id="tool-arrow" aria-pressed="false">Arrow <kbd>a</kbd></button>
        <button id="btn-undo">Undo <kbd>u</kbd></button>
        <button id="btn-clear-cur">Clear this stroke</button>
        <button id="btn-clear-all">Clear all</button>
      </div>
    </div>
    <div class="side">
      <div id="practice-flag" class="practice-flag" hidden>Practice scene &mdash; not scored</div>
      <p class="progress" id="progress"></p>
      <p class="instruction" id="instruction"></p>
      <p class="scene-id" id="scene-id"></p>
      <p class="status" id="status"></p>
      <div class="row">
        <button id="btn-prev">Previous</button>
        <button class="primary" id="btn-next">Next <kbd>&crarr;</kbd></button>
        <button id="btn-skip">Skip this scene</button>
      </div>
      <div class="skipbox" id="skipbox">
        <p style="margin:0 0 6px"><label for="skip-reason">Why are you skipping?</label></p>
        <div class="row tight">
          <select id="skip-reason">__SKIP_OPTIONS__</select>
          <button id="btn-skip-confirm">Confirm skip</button>
          <button id="btn-skip-cancel">Cancel</button>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- --------------------------------------------------------------- done --- -->
<section id="screen-done" hidden>
  <h1>Done &mdash; thank you</h1>
  <p class="sub" id="done-summary"></p>
  <div class="panel">
    <p>Press the button to save your responses, then send me the file. It contains
    only your strokes and timings, plus the name you typed.</p>
    <div class="row tight">
      <button class="primary" id="btn-download">Download JSON</button>
      <button id="btn-back">Back to the last scene</button>
    </div>
    <p class="status" id="done-status"></p>
  </div>
</section>

</div>
<script>
"use strict";

// ------------------------------------------------------------------ bundle --
const BUNDLE = __BUNDLE_JSON__;
const SCENES = BUNDLE.scenes;            // practice first, then the scored roster
const N_SCORED = BUNDLE.n_scored;
const SCALE = BUNDLE.display_scale;
const IMG = BUNDLE.img_px;
const CIRCLE_CSS = BUNDLE.circle_css;
const ARROW_CSS = BUNDLE.arrow_css;
const LS_PREFIX = "spvla_human_study_v1:";

// ------------------------------------------------------------------- state --
let annotatorId = "";        // filename- and path-safe slug, e.g. "alex_smith"
let annotatorName = "";      // as typed, normalised to plain Latin
let idx = 0;                 // index into SCENES
let startedUtc = null;
let state = [];              // per-scene working state, parallel to SCENES
let tool = "circle";
let drawing = false;
let active = null;           // stroke being drawn
let sceneShownAt = 0;
const images = [];           // decoded frames

// ------------------------------------------------------------------- utils --
const $ = (id) => document.getElementById(id);
const nowMs = () => performance.now();
const utc = () => new Date().toISOString();

// Names are folded to plain Latin rather than rejected, so someone called Jose or
// Nguyen can type their name the way they spell it and still produce an id that is
// safe as a filename and as a directory name in the export tree. NFD splits most
// accents into combining marks that can be stripped; Vietnamese d-with-stroke does
// not decompose and is mapped explicitly.
function normaliseName(raw) {
  return (raw || "")
    .normalize("NFD").replace(/[̀-ͯ]/g, "")   // strip combining accents
    .replace(/Đ/g, "D").replace(/đ/g, "d")    // Vietnamese D-stroke
    .replace(/[Æ]/g, "AE").replace(/[æ]/g, "ae")
    .replace(/[Ø]/g, "O").replace(/[ø]/g, "o")
    .replace(/[‘’'`]/g, "")                   // apostrophes: O'Neill -> ONeill
    .replace(/[‐-―-]/g, " ")                  // hyphens become spaces
    .replace(/\s+/g, " ").trim();
}
const slugify = (s) => s.toLowerCase().replace(/ /g, "_");

function blankScene() {
  return {
    circle: null, arrow: null, skipped: false, skip_reason: null,
    history: [],               // tool names in completion order, for undo
    effort: { time_to_first_stroke_ms: null, time_total_ms: 0, n_undo: 0,
              n_redraw: 0, n_points_circle: 0, n_points_arrow: 0 },
    visited: false
  };
}

// -------------------------------------------------------------- ellipse fit --
// Least squares over the stroke points, allowing rotation. The conic
// A x^2 + B xy + C y^2 + D x + E y + F = 0 is fitted with F fixed to -1 after
// the points have been centred and scaled to unit RMS radius; without that
// normalisation the F = -1 constraint is badly conditioned for strokes far from
// the origin. Falls back to a polygon-moment fit if the solve is singular or the
// result is not an ellipse (discriminant >= 0), which happens on near-straight
// or self-crossing strokes.
function solve(M, b, n) {
  const A = M.map((r, i) => r.slice().concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12) return null;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return A.map((r, i) => r[n] / r[i]);   // r === A[i], so r[i] is the pivot
}

function polygonMomentFit(pts) {
  // Green's-theorem moments of the closed polygon. Parameterisation-independent,
  // so uneven point spacing does not bias it. Exact for a true ellipse.
  let a2 = 0, cx = 0, cy = 0;
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const [x0, y0] = pts[i], [x1, y1] = pts[(i + 1) % n];
    const cr = x0 * y1 - x1 * y0;
    a2 += cr; cx += (x0 + x1) * cr; cy += (y0 + y1) * cr;
  }
  const area = a2 / 2;
  if (Math.abs(area) < 1e-6) return null;
  cx /= (3 * a2); cy /= (3 * a2);
  let m20 = 0, m02 = 0, m11 = 0;
  for (let i = 0; i < n; i++) {
    const x0 = pts[i][0] - cx, y0 = pts[i][1] - cy;
    const x1 = pts[(i + 1) % n][0] - cx, y1 = pts[(i + 1) % n][1] - cy;
    const cr = x0 * y1 - x1 * y0;
    m20 += (x0 * x0 + x0 * x1 + x1 * x1) * cr;
    m02 += (y0 * y0 + y0 * y1 + y1 * y1) * cr;
    m11 += (2 * x0 * y0 + x0 * y1 + x1 * y0 + 2 * x1 * y1) * cr;
  }
  const A2 = Math.abs(area);
  m20 = Math.abs(m20 / (12 * area)); m02 = Math.abs(m02 / (12 * area));
  m11 = m11 / (24 * area);
  return eigenToEllipse(cx, cy, m20, m11, m02, 4.0);
}

function eigenToEllipse(cx, cy, m20, m11, m02, k) {
  // Semi-axes from the 2x2 second-moment matrix [[m20,m11],[m11,m02]].
  // For a filled ellipse the central moments are rx^2/4 and ry^2/4, hence k = 4.
  const tr = m20 + m02, dt = m20 * m02 - m11 * m11;
  const s = Math.sqrt(Math.max(0, tr * tr / 4 - dt));
  const l1 = tr / 2 + s, l2 = tr / 2 - s;
  let vx = 1, vy = 0;
  if (Math.abs(m11) > 1e-12) { vx = l1 - m02; vy = m11; }
  else if (m02 > m20) { vx = 0; vy = 1; }
  const nrm = Math.hypot(vx, vy) || 1;
  return finaliseEllipse(cx, cy, Math.sqrt(Math.max(0, k * l1)),
                         Math.sqrt(Math.max(0, k * l2)), vx / nrm, vy / nrm);
}

function finaliseEllipse(cx, cy, s1, s2, vx, vy) {
  // The auto tokens are axis-aligned {cx, cy, rx, ry}. To stay directly
  // comparable, rx is reported as the semi-axis whose direction is NEARER the
  // horizontal, so theta_deg lands in [-45, 45] and rx/ry line up with the auto
  // pair. Auto circles are near-round (radius +/- 2..3 px) so this never has to
  // arbitrate between a long and a short axis.
  let rx, ry, th;
  if (Math.abs(vx) >= Math.abs(vy)) { rx = s1; ry = s2; th = Math.atan2(vy, vx); }
  else { rx = s2; ry = s1; th = Math.atan2(vx, -vy); }
  let deg = th * 180 / Math.PI;
  while (deg > 90) deg -= 180;
  while (deg < -90) deg += 180;
  if (deg > 45) deg -= 90; else if (deg < -45) deg += 90;
  if (!isFinite(rx) || !isFinite(ry) || rx <= 0 || ry <= 0) return null;
  return { cx: cx, cy: cy, rx: rx, ry: ry, theta_deg: deg };
}

function fitEllipse(pts) {
  const n = pts.length;
  if (n < 6) return null;
  let mx = 0, my = 0;
  for (const p of pts) { mx += p[0]; my += p[1]; }
  mx /= n; my /= n;
  let sc = 0;
  for (const p of pts) sc += (p[0] - mx) ** 2 + (p[1] - my) ** 2;
  sc = Math.sqrt(sc / n);
  if (!(sc > 1e-9)) return null;

  const M = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]];
  const rhs = [0,0,0,0,0];
  for (const p of pts) {
    const x = (p[0] - mx) / sc, y = (p[1] - my) / sc;
    const r = [x*x, x*y, y*y, x, y];
    for (let i = 0; i < 5; i++) {
      rhs[i] += r[i];
      for (let j = 0; j < 5; j++) M[i][j] += r[i] * r[j];
    }
  }
  const sol = solve(M, rhs, 5);
  if (!sol) return polygonMomentFit(pts);
  const [A, B, C, D, E] = sol, F = -1;
  if (!(B * B - 4 * A * C < 0)) return polygonMomentFit(pts);

  const den = 4 * A * C - B * B;
  const x0 = (B * E - 2 * C * D) / den, y0 = (B * D - 2 * A * E) / den;
  const Fp = F + (D * x0 + E * y0) / 2;
  if (!(Fp < 0)) return polygonMomentFit(pts);

  // Eigen-decompose [[A, B/2],[B/2, C]]; semi-axis along eigenvector v is
  // sqrt(-Fp / lambda).
  const m11 = B / 2;
  const tr = A + C, dt = A * C - m11 * m11;
  const s = Math.sqrt(Math.max(0, tr * tr / 4 - dt));
  const l1 = tr / 2 + s, l2 = tr / 2 - s;
  if (!(l1 > 0 && l2 > 0)) return polygonMomentFit(pts);
  let vx = 1, vy = 0;
  if (Math.abs(m11) > 1e-12) { vx = l1 - C; vy = m11; }
  else if (C > A) { vx = 0; vy = 1; }
  const nr = Math.hypot(vx, vy) || 1; vx /= nr; vy /= nr;
  // l1 is the LARGER eigenvalue, so its axis is the SHORTER one.
  const fit = finaliseEllipse(x0, y0, Math.sqrt(-Fp / l1), Math.sqrt(-Fp / l2), vx, vy);
  if (!fit) return polygonMomentFit(pts);
  fit.cx = fit.cx * sc + mx; fit.cy = fit.cy * sc + my;
  fit.rx *= sc; fit.ry *= sc;
  // A runaway conic fit (very elongated, or centred off the image) means the
  // stroke was not loop-like; the moment fit is the safer answer there.
  if (fit.rx / fit.ry > 8 || fit.ry / fit.rx > 8 ||
      fit.cx < -IMG || fit.cx > 2 * IMG || fit.cy < -IMG || fit.cy > 2 * IMG) {
    return polygonMomentFit(pts) || fit;
  }
  return fit;
}

// ------------------------------------------------------------------ drawing --
const cv = $("canvas"), ctx = cv.getContext("2d");

function toImg(ev) {
  // Display pixel -> 128-space float. Never rounded to the display grid: a 4x
  // upscale would otherwise quantise every recorded coordinate to 0.25 px and
  // make the human strokes look artificially crisper than they are.
  const r = cv.getBoundingClientRect();
  const x = (ev.clientX - r.left) * (cv.width / r.width) / SCALE;
  const y = (ev.clientY - r.top) * (cv.height / r.height) / SCALE;
  return [Math.min(IMG, Math.max(0, x)), Math.min(IMG, Math.max(0, y))];
}

function strokePath(pts, colour, close) {
  if (pts.length < 2) {
    if (pts.length === 1) {
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(pts[0][0] * SCALE, pts[0][1] * SCALE, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    return;
  }
  ctx.strokeStyle = colour; ctx.lineWidth = 4; ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(pts[0][0] * SCALE, pts[0][1] * SCALE);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * SCALE, pts[i][1] * SCALE);
  if (close) ctx.closePath();
  ctx.stroke();
}

function arrowHead(pts) {
  if (pts.length < 2) return;
  const h = pts[pts.length - 1];
  // Take the direction from a point a little way back so a jittery final sample
  // does not spin the head.
  let b = pts[0];
  for (let i = pts.length - 2; i >= 0; i--) {
    if (Math.hypot(h[0] - pts[i][0], h[1] - pts[i][1]) > 3) { b = pts[i]; break; }
    b = pts[i];
  }
  const ang = Math.atan2(h[1] - b[1], h[0] - b[0]);
  const L = 26, W = 0.42;
  const hx = h[0] * SCALE, hy = h[1] * SCALE;
  ctx.fillStyle = ARROW_CSS;
  ctx.beginPath();
  ctx.moveTo(hx, hy);
  ctx.lineTo(hx - L * Math.cos(ang - W), hy - L * Math.sin(ang - W));
  ctx.lineTo(hx - L * Math.cos(ang + W), hy - L * Math.sin(ang + W));
  ctx.closePath(); ctx.fill();
}

function render() {
  const st = state[idx];
  if (!st) return;          // a frame can finish decoding before start() builds state
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, cv.width, cv.height);
  const im = images[idx];
  if (im && im.complete) ctx.drawImage(im, 0, 0, cv.width, cv.height);
  if (st.circle) strokePath(st.circle.points, CIRCLE_CSS, true);
  if (st.arrow) { strokePath(st.arrow.points, ARROW_CSS, false); arrowHead(st.arrow.points); }
  if (drawing && active && active.points.length) {
    const col = active.tool === "circle" ? CIRCLE_CSS : ARROW_CSS;
    strokePath(active.points, col, false);
  }
}

// -------------------------------------------------------------- interaction --
function onDown(ev) {
  if (state[idx].skipped) return;
  cv.setPointerCapture(ev.pointerId);
  drawing = true;
  active = { tool: tool, points: [toImg(ev)] };
  const eff = state[idx].effort;
  if (eff.time_to_first_stroke_ms === null) {
    eff.time_to_first_stroke_ms = Math.round(nowMs() - sceneShownAt);
  }
  render();
}

function onMove(ev) {
  if (!drawing) return;
  const p = toImg(ev), last = active.points[active.points.length - 1];
  if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 0.4) return;   // thin, keep floats
  active.points.push(p);
  render();
}

function onUp(ev) {
  if (!drawing) return;
  drawing = false;
  const p = toImg(ev), last = active.points[active.points.length - 1];
  if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 1e-6) active.points.push(p);
  commit(active);
  active = null;
  render(); refreshStatus(); save();
}

function commit(stroke) {
  const st = state[idx], pts = stroke.points;
  if (stroke.tool === "circle") {
    if (pts.length < 4) return;                 // a tap is not a circle
    if (st.circle) st.effort.n_redraw++;
    st.circle = { points: pts, fit: fitEllipse(pts) };
    st.effort.n_points_circle = pts.length;
  } else {
    if (pts.length < 2) return;
    if (st.arrow) st.effort.n_redraw++;
    const a = pts[0], b = pts[pts.length - 1];
    st.arrow = { points: pts, x0: a[0], y0: a[1], x1: b[0], y1: b[1] };
    st.effort.n_points_arrow = pts.length;
  }
  st.history.push(stroke.tool);
}

function undo() {
  const st = state[idx];
  if (!st.history.length) return;
  const t = st.history.pop();
  if (t === "circle") { st.circle = null; st.effort.n_points_circle = 0; }
  else { st.arrow = null; st.effort.n_points_arrow = 0; }
  st.effort.n_undo++;
  render(); refreshStatus(); save();
}

function setTool(t) {
  tool = t;
  $("tool-circle").setAttribute("aria-pressed", String(t === "circle"));
  $("tool-arrow").setAttribute("aria-pressed", String(t === "arrow"));
}

// ------------------------------------------------------------- scene chrome --
function refreshStatus() {
  const st = state[idx];
  const dot = (c) => '<span class="swatch" style="background:' + c + '"></span>';
  if (st.skipped) {
    $("status").innerHTML = '<span class="warn">Skipped &mdash; ' + st.skip_reason + '</span>';
  } else {
    $("status").innerHTML =
      dot(CIRCLE_CSS) + '<span class="' + (st.circle ? "ok" : "todo") + '">circle ' +
      (st.circle ? "drawn" : "not yet") + '</span> &nbsp;&nbsp; ' +
      dot(ARROW_CSS) + '<span class="' + (st.arrow ? "ok" : "todo") + '">arrow ' +
      (st.arrow ? "drawn" : "not yet") + '</span>';
  }
  $("btn-prev").disabled = (idx === 0);
  $("btn-undo").disabled = !st.history.length;
  $("btn-next").textContent = (idx === SCENES.length - 1) ? "Finish" : "Next \u21B5";
}

function showScene(i) {
  if (idx !== i && state[idx]) {
    state[idx].effort.time_total_ms += Math.round(nowMs() - sceneShownAt);
  }
  idx = i;
  const sc = SCENES[i], st = state[i];
  st.visited = true;
  sceneShownAt = nowMs();
  $("practice-flag").hidden = !sc.practice;
  $("progress").textContent = sc.practice
    ? "Practice scene"
    : "Scene " + sc.order_scored + " of " + N_SCORED;
  $("instruction").textContent = "\u201C" + sc.instruction + "\u201D";
  $("scene-id").textContent = sc.suite + " / " + sc.dir;
  $("skipbox").classList.remove("on");
  setTool("circle");
  render(); refreshStatus(); save();
}

function next() {
  const st = state[idx];
  if (!st.skipped && (!st.circle || !st.arrow)) {
    $("status").innerHTML = '<span class="warn">Draw both a circle and an arrow, ' +
      'or press Skip if you cannot tell.</span>';
    return;
  }
  if (idx === SCENES.length - 1) { finish(); return; }
  showScene(idx + 1);
}

function finish() {
  state[idx].effort.time_total_ms += Math.round(nowMs() - sceneShownAt);
  save();
  const done = state.filter((s, i) => !SCENES[i].practice && !s.skipped).length;
  const skipped = state.filter((s, i) => !SCENES[i].practice && s.skipped).length;
  $("done-summary").textContent = done + " scenes annotated, " + skipped + " skipped.";
  $("screen-draw").hidden = true;
  $("screen-done").hidden = false;
}

// ---------------------------------------------------------------- persistence --
function lsKey() { return LS_PREFIX + BUNDLE.subset_seed + ":" + annotatorId; }

function save() {
  try {
    localStorage.setItem(lsKey(), JSON.stringify({
      annotator_id: annotatorId, annotator_name: annotatorName,
      subset_seed: BUNDLE.subset_seed,
      started_utc: startedUtc, idx: idx, state: state
    }));
  } catch (e) { /* private browsing / quota - the session still works, just no resume */ }
}

function restore() {
  try {
    const raw = localStorage.getItem(lsKey());
    if (!raw) return false;
    const d = JSON.parse(raw);
    if (!Array.isArray(d.state) || d.state.length !== SCENES.length) return false;
    state = d.state;
    for (const s of state) {                        // tolerate an older partial save
      s.history = s.history || []; s.effort = s.effort || blankScene().effort;
    }
    startedUtc = d.started_utc || utc();
    idx = Math.min(Math.max(0, d.idx | 0), SCENES.length - 1);
    return true;
  } catch (e) { return false; }
}

// ---------------------------------------------------------------- export --
function buildExport() {
  const out = [];
  for (let i = 0; i < SCENES.length; i++) {
    const sc = SCENES[i], st = state[i];
    if (!st.visited) continue;
    out.push({
      suite: sc.suite, dir: sc.dir,
      tier: null,          // withheld from the browser; the scorer joins it from disk
      practice: !!sc.practice,
      skipped: !!st.skipped,
      skip_reason: st.skipped ? st.skip_reason : null,
      circle: st.circle ? { points: round(st.circle.points), fit: roundFit(st.circle.fit) } : null,
      arrow: st.arrow ? {
        points: round(st.arrow.points),
        x0: r3(st.arrow.x0), y0: r3(st.arrow.y0), x1: r3(st.arrow.x1), y1: r3(st.arrow.y1)
      } : null,
      effort: {
        time_to_first_stroke_ms: st.effort.time_to_first_stroke_ms,
        time_total_ms: st.effort.time_total_ms,
        n_undo: st.effort.n_undo, n_redraw: st.effort.n_redraw,
        n_points_circle: st.effort.n_points_circle,
        n_points_arrow: st.effort.n_points_arrow
      }
    });
  }
  return {
    schema_version: "1.0",
    annotator_id: annotatorId,
    annotator_name: annotatorName,
    subset_seed: BUNDLE.subset_seed,
    bundle_generated_utc: BUNDLE.generated_utc,
    user_agent: navigator.userAgent,
    started_utc: startedUtc,
    finished_utc: utc(),
    annotations: out
  };
}
const r3 = (v) => (v === null || v === undefined) ? null : Math.round(v * 1000) / 1000;
const round = (pts) => pts.map((p) => [r3(p[0]), r3(p[1])]);
function roundFit(f) {
  if (!f) return null;
  return { cx: r3(f.cx), cy: r3(f.cy), rx: r3(f.rx), ry: r3(f.ry), theta_deg: r3(f.theta_deg) };
}

function download() {
  const data = buildExport();
  const blob = new Blob([JSON.stringify(data, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "human_sketches_" + (annotatorId || "anon") + ".json";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  $("done-status").innerHTML = '<span class="ok">Saved ' + data.annotations.length +
    ' records. Send me <code>' + a.download + '</code>.</span>';
}

// ------------------------------------------------------------------- wiring --
function start() {
  const name = normaliseName($("annotator").value);
  if (!/^[A-Za-z][A-Za-z ]{1,48}$/.test(name)) {
    $("intro-status").innerHTML =
      '<span class="warn">Please give your name using Latin letters and spaces only ' +
      '&mdash; at least two characters, no digits or punctuation.</span>';
    return;
  }
  annotatorName = name;
  annotatorId = slugify(name);
  if (name !== $("annotator").value.trim()) $("annotator").value = name;
  state = SCENES.map(blankScene);
  const resumed = restore();
  if (!startedUtc) startedUtc = utc();
  $("screen-intro").hidden = true;
  $("screen-draw").hidden = false;
  $("screen-done").hidden = true;
  showScene(resumed ? idx : 0);
  if (resumed) {
    $("status").innerHTML = '<span class="ok">Resumed your saved session.</span>';
  }
}

$("btn-start").addEventListener("click", start);
$("annotator").addEventListener("keydown", (e) => { if (e.key === "Enter") start(); });
$("tool-circle").addEventListener("click", () => setTool("circle"));
$("tool-arrow").addEventListener("click", () => setTool("arrow"));
$("btn-undo").addEventListener("click", undo);
$("btn-clear-cur").addEventListener("click", () => {
  const st = state[idx];
  if (tool === "circle") { st.circle = null; st.effort.n_points_circle = 0; }
  else { st.arrow = null; st.effort.n_points_arrow = 0; }
  st.history = st.history.filter((t) => t !== tool);
  render(); refreshStatus(); save();
});
$("btn-clear-all").addEventListener("click", () => {
  const st = state[idx];
  st.circle = null; st.arrow = null; st.history = [];
  st.effort.n_points_circle = 0; st.effort.n_points_arrow = 0;
  st.skipped = false; st.skip_reason = null;
  $("skipbox").classList.remove("on");
  render(); refreshStatus(); save();
});
$("btn-prev").addEventListener("click", () => { if (idx > 0) showScene(idx - 1); });
$("btn-next").addEventListener("click", next);
$("btn-skip").addEventListener("click", () => $("skipbox").classList.add("on"));
$("btn-skip-cancel").addEventListener("click", () => $("skipbox").classList.remove("on"));
$("btn-skip-confirm").addEventListener("click", () => {
  const st = state[idx];
  st.skipped = true; st.skip_reason = $("skip-reason").value;
  st.circle = null; st.arrow = null; st.history = [];
  $("skipbox").classList.remove("on");
  render(); refreshStatus(); save();
  if (idx === SCENES.length - 1) finish(); else showScene(idx + 1);
});
$("btn-download").addEventListener("click", download);
$("btn-back").addEventListener("click", () => {
  $("screen-done").hidden = true; $("screen-draw").hidden = false;
  showScene(SCENES.length - 1);
});

cv.addEventListener("pointerdown", onDown);
cv.addEventListener("pointermove", onMove);
cv.addEventListener("pointerup", onUp);
cv.addEventListener("pointercancel", onUp);

document.addEventListener("keydown", (e) => {
  if ($("screen-draw").hidden) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "c") { setTool("circle"); e.preventDefault(); }
  else if (e.key === "a") { setTool("arrow"); e.preventDefault(); }
  else if (e.key === "u") { undo(); e.preventDefault(); }
  else if (e.key === "Enter") { next(); e.preventDefault(); }
});

// Decode every frame up front; they are already in the document as base64.
for (const sc of SCENES) {
  const im = new Image();
  im.onload = () => { if (!$("screen-draw").hidden) render(); };
  im.src = "data:image/png;base64," + sc.frame_b64;
  images.push(im);
}

// Exposed only so the round-trip smoke test can drive the tool headlessly. No
// ground truth passes through it.
window.__spvla = {
  bundle: BUNDLE, get state() { return state; }, get idx() { return idx; },
  setTool: setTool, next: next, finish: finish, buildExport: buildExport,
  fitEllipse: fitEllipse, toImg: toImg,
  normaliseName: normaliseName, slugify: slugify,
  setAnnotator: (v) => { $("annotator").value = v; }
};
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description="Build the human sketch-collection bundle.")
    ap.add_argument("--smoke", action="store_true",
                    help="3-scene vertical slice (1 control scene per suite) into "
                         "outputs/human_study/smoke/")
    ap.add_argument("--all", action="store_true",
                    help="census: every scene on disk, no sampling (113 scored + 1 "
                         "practice), into outputs/human_study/%s/" % ALL_SUBDIR)
    ap.add_argument("--seed", type=int, default=SUBSET_SEED)
    args = ap.parse_args()
    if args.smoke and args.all:
        sys.exit("--smoke and --all are mutually exclusive")

    quota = SMOKE_QUOTA if args.smoke else TIER_QUOTA
    n_per_suite = SMOKE_N_PER_SUITE if args.smoke else N_PER_SUITE
    mode = "smoke" if args.smoke else "all" if args.all else "full"
    out_dir = (os.path.join(OUT_DIR, "smoke") if args.smoke else
               os.path.join(OUT_DIR, ALL_SUBDIR) if args.all else OUT_DIR)

    print("Sketch-Prompted VLA - human study bundle")
    print("  mode         : %s" % {"smoke": "SMOKE (vertical slice)",
                                   "all": "ALL (census, no sampling)",
                                   "full": "full (stratified 36)"}[mode])
    print("  seed         : %d" % args.seed)
    print("  out          : %s" % out_dir)

    scenes = load_scenes()
    print("  scenes read  : %d" % len(scenes))
    if len(scenes) != 114 and not args.smoke:
        print("  ! expected 114 scenes on disk, found %d" % len(scenes))

    rng = random.Random(args.seed)
    if args.all:
        # Hold the practice scene out FIRST, then take everything else. Reversing
        # the order of the sampled path is deliberate: `pick_practice` wants a
        # control scene outside the scored set, and once the census is taken
        # there is nothing outside it to draw from.
        practice = pick_practice(scenes, [], rng)
        chosen = [s for s in scenes
                  if (s["suite"], s["dir"]) != (practice["suite"], practice["dir"])]
        chosen.sort(key=lambda s: (s["suite"], s["dir"]))   # seed-only determinism
    else:
        chosen = stratified_sample(scenes, quota, n_per_suite, rng)
        practice = pick_practice(scenes, chosen, rng)
    rng.shuffle(chosen)        # interleave suites so no annotator sees them blocked

    tally = collections.Counter((s["suite"], s["tier"]) for s in chosen)
    print("  sampled      : %d scenes" % len(chosen))
    for suite in SUITES:
        row = "  %-12s" % ("    " + suite)
        for tier in TIER_QUOTA:
            row += "  %s=%d" % (tier[:4], tally[(suite, tier)])
        print(row)
    print("  practice     : %s / %s  (%s, %s)"
          % (practice["suite"], practice["dir"], practice["tier"],
             "held out of the census" if args.all else "outside the subset"))

    os.makedirs(os.path.join(out_dir, "responses"), exist_ok=True)
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    # --- scene_subset.json: the roster PLUS the truth --------------------------
    subset = {
        "schema_version": "1.0",
        "generated_utc": generated,
        "subset_seed": args.seed,
        "mode": mode,
        "n_scored": len(chosen),
        "tier_quota_per_suite": quota,
        "img_px": IMG_PX,
        "display_scale": DISPLAY_SCALE,
        "skip_reasons": SKIP_REASONS,
        "practice": truth_view(practice, -1),
        "scenes": [truth_view(s, i) for i, s in enumerate(chosen)],
    }
    subset_path = os.path.join(out_dir, "scene_subset.json")
    with open(subset_path, "w", encoding="utf-8") as fh:
        json.dump(subset, fh, indent=1)
    n_injected = sum(1 for s in subset["scenes"] if not s["truth"]["dest_in_all_pixels"])
    print("  wrote        : scene_subset.json (%d bytes)" % os.path.getsize(subset_path))
    print("                 %d scene(s) have a region destination absent from "
          "all_pixels" % n_injected)

    # --- sketch_tool.html: the roster, MINUS the truth -------------------------
    browser_scenes = [dict(annotator_view(practice, -1), practice=True, order_scored=0)]
    for i, s in enumerate(chosen):
        browser_scenes.append(dict(annotator_view(s, i), practice=False, order_scored=i + 1))

    bundle = {
        "schema_version": "1.0",
        "generated_utc": generated,
        "subset_seed": args.seed,
        "n_scored": len(chosen),
        "img_px": IMG_PX,
        "display_scale": DISPLAY_SCALE,
        "circle_css": "rgb(%d,%d,%d)" % CIRCLE_RGB,
        "arrow_css": "rgb(%d,%d,%d)" % ARROW_RGB,
        "scenes": browser_scenes,
    }

    # Audit the payload BEFORE it is written. Every key that reaches the browser
    # must be on this list; a future edit that widens annotator_view() fails here
    # rather than silently invalidating the study.
    allowed = {"order", "suite", "dir", "instruction", "frame_b64", "practice",
               "order_scored"}
    for sc in browser_scenes:
        extra = set(sc) - allowed
        if extra:
            sys.exit("ground truth would leak into the HTML: %s" % sorted(extra))
    blob = json.dumps(bundle, separators=(",", ":"))
    for s in chosen + [practice]:
        m = s["meta"]
        for token in (m["target"], m["destination"], m["destination_region"]):
            if token in blob:
                sys.exit("ground-truth string %r found in the browser payload" % token)
    est_min = max(4, round(len(browser_scenes) * 0.35))
    html = (TEMPLATE
            .replace("__BUNDLE_JSON__", blob)
            .replace("__INTRO_HTML__", INTRO_HTML
                     .replace("__N_SCENES__", str(len(chosen))))
            .replace("__EST_MIN__", str(est_min))
            .replace("__SKIP_OPTIONS__",
                     "".join('<option value="%s">%s</option>' % (r, r) for r in SKIP_REASONS)))
    html_path = os.path.join(out_dir, "sketch_tool.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    size = os.path.getsize(html_path)
    print("  wrote        : sketch_tool.html (%.2f MB, %d scenes incl. practice)"
          % (size / 1048576.0, len(browser_scenes)))
    print("  leak audit   : clean - no target/destination string in the payload")
    print("\nHand the annotator sketch_tool.html alone. Returned JSON goes in %s"
          % os.path.join(out_dir, "responses"))


if __name__ == "__main__":
    main()
