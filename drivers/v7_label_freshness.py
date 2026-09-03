#!/usr/bin/env python3
"""Mark which frame-sweep rows have a still-valid frame-0 target mask.

`target` and `distractor` are ONE mask per episode, rendered at frame 0. The
robot picks up and carries the target bowl, so past the grasp the target mask
no longer covers the bowl and `point_hit_real` scores a correct pointer as a
miss. The distractor bowl is never manipulated, so its mask stays valid all
episode -- which is why `point_hit_swap` holds up while `point_hit_real` falls.

The distractor region is therefore a within-frame control: it absorbs lighting
and arm-occlusion drift. A frame's target label is STALE when the target region
has changed substantially MORE than the distractor region has.

Emits the joined per-row table; scoring lives in v7_sweep_corrected.
"""
import json, sys
import numpy as np

sys.path.insert(0, "/workspace/SketchPromptVLA-Pi/src")
from sketchvla.utils import validation_data  # noqa: E402

DATA = "/workspace/data/sketch_libero_rlds_paired_cf"
OUT = "/workspace/SketchPromptVLA-Pi/outputs/v7_label_freshness.json"

def region_diff(a, b, mask):
    m = mask.squeeze() > 127
    if m.sum() == 0:
        return 0.0
    return float(np.abs(a[m].astype(np.int16) - b[m].astype(np.int16)).mean())

def main():
    rows, ref = [], {}
    for s in validation_data.iter_rlds_samples(
            DATA, "sketch_libero", action_horizon=10, split="val",
            holdout_frac=0.1, max_episodes=41, frames_per_episode=12):
        e = s.element
        img = np.asarray(e["observation/image"])
        if s.episode_key not in ref:
            ref[s.episode_key] = img.copy()          # frame 0 is always picked
        r = ref[s.episode_key]
        rows.append({
            "episode_key": s.episode_key,
            "frame_index": s.frame_index,
            "target_diff": region_diff(img, r, np.asarray(e["observation/target"])),
            "distractor_diff": region_diff(img, r, np.asarray(e["observation/distractor"])),
        })
        if len(rows) % 100 == 0:
            print("scanned %d frames" % len(rows), flush=True)

    json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print("wrote %s (%d rows)" % (OUT, len(rows)))

    td = np.array([r["target_diff"] for r in rows])
    dd = np.array([r["distractor_diff"] for r in rows])
    print("\ntarget_diff     pct: " + "  ".join("p%d=%.1f" % (p, np.percentile(td, p)) for p in (10, 50, 90, 99)))
    print("distractor_diff pct: " + "  ".join("p%d=%.1f" % (p, np.percentile(dd, p)) for p in (10, 50, 90, 99)))
    print("\nfrom frame 0 only (should be ~0 for both):")
    z = [r for r in rows if r["frame_index"] == 0]
    print("  n=%d  target=%.3f  distractor=%.3f"
          % (len(z), np.mean([r["target_diff"] for r in z]), np.mean([r["distractor_diff"] for r in z])))

if __name__ == "__main__":
    main()
