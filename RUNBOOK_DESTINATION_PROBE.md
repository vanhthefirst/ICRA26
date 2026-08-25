# Runbook — destination-displacement probe (the mirror of the target probe)

Aaron, 25 Aug 2026. Mirror of the run recorded in
`claude/displacement_probe_result.md`. Same tasks, same caption, same dropped
instance, same gates, same radii — the only difference is that `plate_1` is
translated instead of `akita_black_bowl_1`.

The point is one figure: two curves on one x-axis. The target arm is flat at
99.3% out to 12 cm. If the destination arm is a cliff, the whole position
argument is a single panel instead of a pair of tables.

Budget: about 45 minutes of GPU, matching the target arm.

## 0. Pod

Needs the libero client env and the pi0.5 policy server, exactly as the baselines
did. If the pod was terminated, `pod_bootstrap.sh` + `libero_env.sh` rebuild the
container disk (about an hour, mostly unattended).

    source $OPENPI/examples/libero/.venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:$OPENPI/third_party/libero
    export MUJOCO_GL=egl
    export LIBERO_SPATIAL_BDDL=$OPENPI/third_party/libero/libero/libero/bddl_files/libero_spatial
    cd $REPO

Policy server on :8000, same as every previous run.

## 1. Build

    mkdir -p outputs/validation_set_destination
    python scripts/build_displacement_probe.py --move destination 2>&1 \
        | tee outputs/validation_set_destination/build_log.txt

Writes `outputs/validation_set_destination/`, suite name `destination`.

**Expected coverage.** I checked the two geometry gates offline against the
shipped rectangles, so the build should not surprise me: 26 of 34 offsets clear
reach and `D_CROSS`. The 8 that do not are `yneg` at 9 and 12 cm in both tasks
(the plate lands on the cookie box) and all four `xneg` offsets in
`between_plate_ramekin` (the plate lands on the bowl, which sits between the
plate and the ramekin by construction). The frame gate is only checkable in the
simulator and will likely take some of the far `xpos` offsets — the plate is
already at pixel x=97 of 128 at its shipped position.

So expect roughly 20-24 scenes. If the build comes back under 18, add
`next_to_ramekin` to `TASKS` and rebuild; it costs about 4 more minutes of
rollout per surviving scene.

**Known asymmetry, stated rather than hidden.** In each arm the moved object's
placement region is re-centred and normalised to `HALF_BOX` (±12 mm) while every
other region stays stock. So in the target arm the plate keeps its ±10 mm region
and in this arm the bowl does. That is 2 mm of placement jitter against 30 mm
steps, and rebuilding the target arm to remove it is not worth 45 minutes of GPU.

## 2. Roll out

    SCENES=$(python -c "import json;print(','.join('destination/'+e['dir'] for e in json.load(open('outputs/validation_set_destination/manifest.json'))))")
    EXTRA_SUITES=destination python scripts/rollout_sketch.py --policy pi05 \
        --conditions text_only --prompt-type explicit \
        --run-id pi05_destination --n-rollouts 14 --scenes "$SCENES" --resume

`EXTRA_SUITES` maps the suite name to `outputs/validation_set_destination/`
without touching the three canonical suites, so `normalize`, `audit` and
`build_prompt_variants` ignore it and the 113-scene claim is unaffected.

## 3. Read it

    python scripts/analyze_displacement.py \
        --run pi05_displacement  --set validation_set_displacement \
        --label "target moved" \
        --compare pi05_destination --compare-set validation_set_destination \
        --compare-label "destination moved"

Writes `report/displacement/fig_displacement_both.png` and folds both arms into
`outputs/rollouts/pi05_displacement/analysis.json`.

## What to look at first

Not the success curve — the **terminal-distance** column the analyser now prints
beside it. `terminal_dist_xy` is how far the carried bowl ended from the plate.

- If the policy reads the scene and simply gets worse with distance, terminal
  distance stays small and success falls: it is going to the right place and
  failing to get there.
- If the policy has memorised the drop point, terminal distance tracks the
  displacement roughly one for one, so the `dist/offset` ratio climbs toward
  1.0. That is the bowl being delivered to where the plate *used to* ship,
  regardless of where it now is.

The second is the claim. Success can only say the policy failed; the ratio says
where it went instead, and it is the sentence the figure caption should carry.

## The result this is predicted to produce

From the fine-tuning coverage, across all ten shipped `libero_spatial` tasks:

| object | distinct table positions | spread |
|---|---|---|
| `akita_black_bowl_1` (target) | 5 on the table | 0.498 m |
| `plate_1` (destination) | 1 | 0 |

A 12 cm target displacement is inside the demonstrated distribution, which is why
the target arm is flat. Any destination displacement is outside it. The anchored
baselines already show 14.8% correct-destination when the goal is a plate at a
novel position, below the 33-50% a guess would score — so the prediction is a
cliff, and a `dist/offset` ratio near 1.

If the destination curve comes back flat too, that is a real result and it kills
the position story rather than the probe: it would mean the anchored-set
destination effect came from having two plates to choose between, not from the
goal plate being in a novel place. Either way the figure is worth the 45 minutes.
