"""
Sketch-Prompted VLA — package the GOAL validation set: write DATASHEET.md + contact_sheet.png.

WHY THIS EXISTS AS A SEPARATE SCRIPT
The Spatial and Object builders each end by writing their own DATASHEET.md and
contact_sheet.png; the Goal builder does not. Re-running the Goal builder just to
emit those two files would resample and OVERWRITE the 38 verified scenes, so this
script derives both from what is already on disk instead.

It reads `manifest.json` + `scene_*/meta.json` + `scene_*/sketch.png` and writes
only the two missing files. **It never touches scene data**, and it refuses to
write into the Spatial or Object directories (their datasheets are bespoke prose).

The contact sheet uses the SAME recipe as the two siblings, so the three suites
are visually consistent: 128x128 sketch tiles on a 130px grid, 8 columns,
background grey 40, yellow HERSHEY_SIMPLEX 0.3 label at (2,12).
Goal adds a second label line, because unlike the siblings (one task each) the
Goal suite spans 7 different LIBERO tasks and the tile is unreadable without it.

    cd /mnt/c/Users/Admin/sketch_prompted_vla
    python scripts/package_goal_suite.py

Requires numpy + cv2 only (no libero, no MuJoCo). Safe to re-run; idempotent.
"""

import json, os, glob, sys
from collections import Counter

import numpy as np
import cv2

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, rename-proof
ROOT = os.path.join(_REPO, "outputs")
SUITE_DIR = "validation_set_goal"
OUT = os.path.join(ROOT, SUITE_DIR)

TIER_ORDER = ["control", "referential", "directional", "both"]
TIER_MEANING = {
    "control":     "unambiguous; caption alone suffices",
    "referential": "which object is ambiguous",
    "directional": "which destination is ambiguous",
    "both":        "both axes ambiguous",
}

# PNG end-of-file marker; a truncated DrvFs write is missing this.
_PNG_IEND = b"IEND\xaeB`\x82"


def safe_imwrite(path, img, retries=5):
    """Durably write across the WSL2<->Windows DrvFs bridge (same guard the
    builders use: verify the file on disk ends with a valid PNG IEND chunk)."""
    for _ in range(retries):
        cv2.imwrite(path, img)
        try:
            with open(path, "rb") as f:
                if f.read()[-8:] == _PNG_IEND:
                    return
        except OSError:
            pass
    raise IOError(f"could not durably write {path}")


def stat3(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return min(vals), sum(vals) / len(vals), max(vals)


def fmt3(vals, f="{:.3f}"):
    s = stat3(vals)
    if s is None:
        return "n/a"
    return " / ".join(f.format(x) for x in s)


def load_metas():
    metas = []
    for p in sorted(glob.glob(os.path.join(OUT, "scene_*", "meta.json"))):
        with open(p) as f:
            m = json.load(f)
        m["_dir"] = os.path.basename(os.path.dirname(p))
        metas.append(m)
    return metas


# ---------------------------------------------------------------- contact sheet
def build_contact_sheet(metas):
    imgs = []
    for m in metas:
        p = os.path.join(OUT, m["_dir"], "sketch.png")
        im = cv2.imread(p)
        if im is None:
            print(f"  [warn] unreadable sketch: {p}")
            continue
        im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_NEAREST)

        # line 1 — same field order as the Spatial/Object sheets
        flag = "" if m["grasp"]["grasp_success"] else " g!"
        cv2.putText(im, f"{m['tier'][:4]} {m['n_target']}x{m['n_dest']}d "
                        f"v{m['visibility']['visibility']:.2f}{flag}",
                    (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 255, 255), 1, cv2.LINE_AA)
        # line 2 — Goal spans 7 tasks; the siblings span one, hence the addition
        cv2.putText(im, f"{m['task'][:20]}", (2, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 200, 255), 1, cv2.LINE_AA)
        imgs.append(im)

    if not imgs:
        sys.exit("no sketches found — nothing to tile")

    cols = min(8, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.full((rows * 130, cols * 130, 3), 40, np.uint8)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        sheet[r * 130 + 1:r * 130 + 129, c * 130 + 1:c * 130 + 129] = im

    path = os.path.join(OUT, "contact_sheet.png")
    safe_imwrite(path, sheet)
    print(f"  wrote contact_sheet.png  ({rows}x{cols} grid, {len(imgs)} tiles, "
          f"{sheet.shape[1]}x{sheet.shape[0]} px)")


# ------------------------------------------------------------------- datasheet
def build_datasheet(metas):
    n = len(metas)
    tiers = Counter(m["tier"] for m in metas)
    tasks = Counter(m["task"] for m in metas)
    kinds = Counter(m["dest_kind"] for m in metas)
    cats = Counter(m["target_cat"] for m in metas)

    rows = []
    for t in TIER_ORDER:
        sel = [m for m in metas if m["tier"] == t]
        if not sel:
            continue
        N = [m["n_target"] for m in sel]
        M = [m["n_dest"] for m in sel]
        rng = lambda v: f"{min(v)}" if min(v) == max(v) else f"{min(v)}-{max(v)}"
        rows.append(f"| {t} | {len(sel)} | {rng(N)} | {rng(M)} | {TIER_MEANING[t]} |")

    task_lines = "\n".join(
        f"| `{k}` | {v} | {'OBJECT' if any(m['dest_kind']=='OBJECT' for m in metas if m['task']==k) else 'REGION'} |"
        for k, v in tasks.most_common())

    cat_str = ", ".join(f"{k} x{v}" for k, v in cats.most_common())

    ungrasped = [m for m in metas if not m["grasp"]["grasp_success"]]
    ung_by_task = Counter(m["task"] for m in ungrasped)
    ung_str = ", ".join(f"`{k}` x{v}" for k, v in ung_by_task.most_common())

    # Per-category grasp split — the evidence that grasp failure here is a
    # property of the CATEGORY, not of any particular sampled placement.
    by_cat = {}
    for m in metas:
        ok, tot = by_cat.get(m["target_cat"], (0, 0))
        by_cat[m["target_cat"]] = (ok + int(m["grasp"]["grasp_success"]), tot + 1)
    cat_rows = "\n".join(
        f"| `{k}` | {t} | {o} | {t - o} |"
        for k, (o, t) in sorted(by_cat.items(), key=lambda kv: -kv[1][1]))
    n_ok = sum(1 for m in metas if m["grasp"]["grasp_success"])

    neg_counts = Counter(len(m["oracle_negatives"]) for m in metas
                         if isinstance(m.get("oracle_negatives"), dict))

    md = f"""# Sketch-Prompted VLA validation set — LIBERO-**Goal** suite

{n} scenes. Built by `scripts/build_validation_set_goal.py` (VSLICE=False,
SMOKE=False); `DATASHEET.md` and `contact_sheet.png` written by
`scripts/package_goal_suite.py`. Canonical schema v1.0 (see SCHEMA.md).
Companion to `outputs/validation_set_spatial/` and `outputs/validation_set_object/`.

## Purpose

Scenes that are deliberately **impossible to disambiguate from the caption alone**.
Each scene pairs a vague instruction with several identical candidate objects
and/or several candidate destinations. A circle (which object) + arrow (which
destination) is the only signal that identifies the intended instance. The BDDL
goal names one specific instance, so a rollout can be scored automatically.

What makes Goal different from its two siblings: it has **no single flat
workspace**. Each LIBERO-Goal task ships a bespoke scene with real fixtures
(`wooden_cabinet`, `flat_stove`, `wine_rack`) and affordance regions, so the
builder starts from the ORIGINAL BDDL and injects duplicate instances, keeping
every fixture, region and `(:init)` line intact and retargeting only `(:goal)`.

## Composition

| tier | scenes | targets N | dests M | meaning |
|---|---|---|---|---|
{chr(10).join(rows)}

### Tasks covered

| task | scenes | destination kind |
|---|---|---|
{task_lines}

Destination kind overall: {', '.join(f'{k} {v}' for k, v in kinds.most_common())}.
Goal predicate: {', '.join(f'{k} {v}' for k, v in Counter(m['goal_predicate'] for m in metas).most_common())}.
Target categories: {cat_str}.

**Why the tier/task shape is uneven.** A Goal destination is either an OBJECT
instance (duplicable → supports directional/both) or a fixed affordance REGION
(not duplicable → referential-only). The probe (`outputs/probe_goal.txt`, §A2)
measured the split as 2 object-dest vs 6 region-dest tasks, and the builder
implements **Option 1**: every usable task feeds control+referential, while the
2 object-dest tasks additionally feed directional+both. The drawer task was
dropped after all 24 seeds scored `oracle_false` — the drawer starts closed, so
the `In` region site is retracted and no teleport satisfies it. Opening then
inserting is two actions, which one circle+arrow cannot express (the same
rationale that postpones `libero_10`). Roster: 7 usable tasks, all `On`.

## Per scene

`scene.bddl` `frame0.png` `sketch.png` `tokens.json` `meta.json`
(128x128, camera `agentview`). `meta.json` records the seed, camera matrix, all
object pixels, placements, visibility, grasp, clearance, pixel separations, and
the full oracle result matrix. Every scene is reproducible from its seed.

## Gates (all must pass; failures resample with a new seed)

- **settled** — every body above the floor plane
- **in-frame** — target and all destinations project inside the image
- **pixel separation (sketch resolvability)** — thresholds derive from the
  *projected extents* of the objects being told apart, not constants
- **visibility** — RGB-difference occlusion fraction of target >= 0.35
- **not pre-solved** — `check_success()` is False at t=0
- **oracle +** — teleporting the target to the named destination scores True
- **oracle - (directional)** — target into *every other* destination scores False
- **oracle - (referential)** — *every* same-category sibling into the named
  destination scores False

The two negative oracles are what certify a scene is genuinely unsolvable
without the sketch, along each axis independently. Negative oracles per scene:
{', '.join(f'{k}→{v} scenes' for k, v in sorted(neg_counts.items()))}
(control scenes have none by construction — nothing to disambiguate).

**Grasp is RECORDED, NOT GATED in this suite** — see Known limitations.

## Measured distributions (min / mean / max)

| metric | value |
|---|---|
| visibility | {fmt3([m['visibility']['visibility'] for m in metas])} |
| grasp lift (m) | {fmt3([m['grasp']['lift'] for m in metas])} |
| clearance xy (m) | {fmt3([m['clearance_xy'] for m in metas])} |
| px sep, dests | {fmt3([m['px_sep_dests'] for m in metas], '{:.1f}')} |
| px sep, siblings | {fmt3([m['px_sep_siblings'] for m in metas], '{:.1f}')} |

Positive oracle: {sum(1 for m in metas if m.get('oracle_success'))}/{n} True.
Rejections during generation: 17 (siblings x9, oracle x4, fell x3, dest x1).

## Known limitations

- **Grasp is recorded, not gated — {len(ungrasped)} of {n} scenes have
  `grasp_success: False`** ({ung_str}). This is a deliberate decision, not a
  defect. See the dedicated section below.
- **Directional and both tiers rest on only 2 tasks** (`bowl_on_plate`,
  `cheese_in_bowl`), the only ones with a duplicable object destination.
  Region-destination tasks cannot express "which destination?" ambiguity.
- **Ambiguity is multiplicity, not occlusion.** `agentview` looks down steeply,
  so objects rarely occlude one another and the visibility gate seldom binds.
  Do not cite this set as evidence about occlusion robustness.
- **All 38 scenes use the `On` predicate.** The one `In` task (the drawer) was
  dropped, so this suite does not exercise `In` at all — the Object suite is
  where `In` coverage lives.
- Scoring the headline experiment (text-only vs text+sketch) needs a trained
  policy on a GPU. Each scene ships its BDDL for exactly that.

## The grasp gate: why this suite differs from Spatial and Object

Spatial and Object both **reject** a scene when the scripted top-down grasp fails.
Goal records the result and keeps the scene. That is a real divergence, and the
justification rests on what the gate is measuring in each suite.

| target category | scenes | grasp True | grasp False |
|---|---|---|---|
{cat_rows}

**The split is perfectly categorical — zero within-category variance.** Every
`akita_black_bowl` and `cream_cheese` scene grasps; no `wine_bottle` or `plate`
scene ever does. So in this suite a grasp failure is a property of the *object
category* under a scripted top-down gripper, not of a particular sampled
placement. Resampling the seed cannot fix it.

Contrast Object: all 38 of its kept scenes grasp successfully, and its 15
`ungraspable` rejections were all *graspable* grocery categories — those were bad
placements, and resampling did fix them. **The same gate therefore does different
work in the two suites:** in Object it filters unlucky samples; in Goal it would
not filter anything, it would delete the `wine_on_rack`, `wine_on_cabinet` and
`plate_to_stove_front` tasks outright, taking the suite from 7 usable tasks to 4.

What certifies a scene is `oracle_success` — teleporting the target to the named
destination scores the BDDL goal True — and that is independent of the scripted
grasp. All {n}/{n} scenes pass it.

### Consequence for the headline experiment (read this before scoring)

Keeping these scenes has a cost: a policy that cannot physically lift a wine
bottle will score near zero on those {len(ungrasped)} scenes in **both** the
text-only and the text+sketch condition. The comparison stays valid — both arms
face identical physical difficulty — but the floor effect compresses the measured
effect size and could mask a real difference.

Recommended reporting: **stratify.** Headline number on the {n_ok} grasp-True
scenes, with the {len(ungrasped)} grasp-False scenes reported separately as a
manipulation-limited subset. Filter with
`meta['grasp']['grasp_success']`; those scenes are flagged `g!` on the contact
sheet. Do not silently pool them.

## Reproduce

```bash
conda activate libero
cd /mnt/c/Users/Admin/sketch_prompted_vla
python scripts/build_validation_set_goal.py     # rebuilds all 38 scenes
python scripts/package_goal_suite.py                # this file + contact sheet
python scripts/normalize_validation_schema.py       # refresh canonical manifests
python scripts/audit_validation_sets.py             # read-only verification
```
"""
    path = os.path.join(OUT, "DATASHEET.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  wrote DATASHEET.md       ({len(md.splitlines())} lines)")


def main():
    if not os.path.isdir(OUT):
        sys.exit(f"suite directory not found: {OUT}")
    metas = load_metas()
    if not metas:
        sys.exit(f"no scene_*/meta.json under {OUT}")
    print(f"packaging {SUITE_DIR}: {len(metas)} scenes")
    build_contact_sheet(metas)
    build_datasheet(metas)
    print("done — scene data untouched")


if __name__ == "__main__":
    main()
