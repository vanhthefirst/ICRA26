"""
Sketch-Prompted VLA — audit a written RLDS export without TensorFlow (runs anywhere).

Answers the two questions that decide whether an export is safe to train on:

  1. IS THERE AN ACTION SIGNAL? A validation export — scenes built from BDDLs and
     pinned at their annotated start frame — has one step per episode and
     `action = zeros(7)`, because no demonstration exists behind it. Fine-tuning
     on that regresses the policy onto a constant zero action: the arm never
     moves and every rollout scores 0%, with nothing raised anywhere along the
     way. That is the failure this script exists to make visible in seconds.

  2. IS THE FRAME THE RIGHT WAY UP? pi0.5-LIBERO was fine-tuned on
     `openvla/modified_libero_rlds`, whose frames sit 180 degrees from raw
     robosuite output. In that orientation the arm hangs from the TOP of the
     frame against the dark back wall and the lit table fills the bottom, so the
     bottom third is measurably brighter than the top. Raw robosuite output is
     that image inverted, and no consumer downstream says so —
     `convert_libero_data_to_lerobot.py` passes the image straight through.

WHY IT PARSES THE TFRECORDS BY HAND

`export_rlds_pack.py --verify` checks the same things through tfds, but tfds
needs tensorflow, which is not installed in either the `libero` client env or the
base env. This script reads the TFRecord framing and the `tf.train.Example`
protobuf directly, so the audit is available on any machine holding the files —
including the one that has to decide whether to re-export.

Run anywhere (stdlib + numpy + Pillow). No simulator, no GPU, no TensorFlow.

    python scripts/audit_rlds_export.py outputs/rlds/sketch_libero_val_spatial_anchored

Exit status is 0 if the export is trainable and correctly oriented, 1 otherwise.
"""

import argparse
import glob
import io
import os
import struct
import sys

import numpy as np
from PIL import Image


# ----------------------------------------------------------- protobuf reader --
def _varint(buf, i):
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, i


def _fields(buf):
    """Yield (field_number, wire_type, payload) for one protobuf message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            val, i = _varint(buf, i)
        elif wt == 2:
            n, i = _varint(buf, i)
            val, i = buf[i:i + n], i + n
        elif wt == 5:
            val, i = buf[i:i + 4], i + 4
        elif wt == 1:
            val, i = buf[i:i + 8], i + 8
        else:
            raise ValueError("unsupported wire type %d" % wt)
        yield fn, wt, val


def _example_features(payload):
    """tf.train.Example -> {feature_name: serialised Feature}."""
    out = {}
    for fn, _, val in _fields(payload):
        if fn != 1:                       # Example.features
            continue
        for fn2, _, entry in _fields(val):
            if fn2 != 1:                  # Features.feature map entry
                continue
            key = value = None
            for fn3, _, v in _fields(entry):
                if fn3 == 1:
                    key = v.decode()
                elif fn3 == 2:
                    value = v
            out[key] = value
    return out


def _values(feature):
    """Feature -> (kind, list). Kinds: bytes / float / int / empty."""
    if feature is None:
        return "empty", []
    for fn, _, val in _fields(feature):
        if fn == 1:                                        # bytes_list
            return "bytes", [v for _, _, v in _fields(val)]
        if fn == 2:                                        # float_list
            out = []
            for _, wt, v in _fields(val):
                if wt == 2:
                    out += list(struct.unpack("<%df" % (len(v) // 4), v))
                elif wt == 5:
                    out += list(struct.unpack("<f", v))
            return "float", out
        if fn == 3:                                        # int64_list
            out = []
            for _, wt, v in _fields(val):
                if wt == 2:
                    j = 0
                    while j < len(v):
                        x, j = _varint(v, j)
                        out.append(x)
                elif wt == 0:
                    out.append(v)
            return "int", out
    return "empty", []


class TruncatedShardError(RuntimeError):
    """A shard ends mid-record — almost always an interrupted download."""


def _records(path):
    """TFRecord framing: uint64 length, uint32 crc, payload, uint32 crc.

    Bounds are checked rather than assumed. A slice past the end of a bytes
    object returns a SHORT result instead of raising, so an interrupted download
    would otherwise surface as a silently smaller episode count and a quietly
    wrong audit -- which is the one thing this file exists to prevent.
    """
    blob = open(path, "rb").read()
    total = len(blob)
    i = 0
    while i < total:
        if i + 12 > total:
            raise TruncatedShardError(
                "%s: %d trailing byte(s), too few for a record header. The file "
                "is incomplete -- re-download it." % (path, total - i))
        (length,) = struct.unpack("<Q", blob[i:i + 8])
        i += 12                                   # length + its masked crc32c
        end = i + length
        if end + 4 > total:
            raise TruncatedShardError(
                "%s: record header declares %d payload bytes ending at %d, but "
                "the file is only %d bytes (short by %d). The file is "
                "incomplete -- re-download it."
                % (path, length, end + 4, total, end + 4 - total))
        yield blob[i:end]
        i = end + 4                               # payload + its masked crc32c


# ------------------------------------------------------------------- audit --
def audit(data_dir, action_key="steps/action", image_key="steps/observation/image"):
    shards = sorted(glob.glob(os.path.join(data_dir, "*.tfrecord-*")))
    if not shards:
        raise SystemExit(
            "no *.tfrecord-* shards under %s\n"
            "Point this at the version directory, e.g. "
            "outputs/rlds/<name>/1.0.0/ (or its parent — that is resolved too)."
            % data_dir)

    n = n_zero_action = n_single_step = n_no_demo = n_upright = 0
    action_absmax = 0.0
    drifts = []

    for shard in shards:
        for rec in _records(shard):
            feat = _example_features(rec)
            n += 1

            _, acts = _values(feat.get(action_key))
            n_steps = max(1, len(acts) // 7)
            if n_steps == 1:
                n_single_step += 1
            amax = max((abs(a) for a in acts), default=0.0)
            action_absmax = max(action_absmax, amax)
            if amax == 0.0:
                n_zero_action += 1

            _, demo = _values(feat.get("episode_metadata/has_demonstration"))
            if demo and not demo[0]:
                n_no_demo += 1

            _, imgs = _values(feat.get(image_key))
            if imgs:
                im = np.asarray(Image.open(io.BytesIO(imgs[0])), np.float32)
                h = im.shape[0]
                if im[-h // 3:].mean() > im[:h // 3].mean():
                    n_upright += 1

            _, circ = _values(feat.get("steps/sketch/circle"))
            _, meta = _values(feat.get("steps/sketch/circle_meta"))
            if circ and len(meta) >= 2:
                mask = np.asarray(Image.open(io.BytesIO(circ[0]))).squeeze()
                ys, xs = np.nonzero(mask > 127)
                # Skip rings clipped by the frame edge: only part of the stroke
                # survives to be averaged, so the centroid moves inboard of the
                # recorded centre for geometric reasons. Across 3,234 steps of
                # sketch_libero_rlds every drift over 3 px was clipped and no
                # unclipped mask passed 0.29 px.
                if len(xs):
                    h_m, w_m = mask.shape[:2]
                    if not (xs.min() == 0 or ys.min() == 0
                            or xs.max() == w_m - 1 or ys.max() == h_m - 1):
                        drifts.append(float(np.hypot(xs.mean() - meta[0],
                                                     ys.mean() - meta[1])))

    return dict(n=n, shards=len(shards), n_zero_action=n_zero_action,
                n_single_step=n_single_step, n_no_demo=n_no_demo,
                n_upright=n_upright, action_absmax=action_absmax,
                drift_max=max(drifts) if drifts else None)


def report(r):
    n = r["n"]
    print("episodes                         %d  (%d shard(s))" % (n, r["shards"]))
    print("single-step episodes             %d / %d" % (r["n_single_step"], n))
    print("episodes with all-zero actions   %d / %d" % (r["n_zero_action"], n))
    print("global max |action|              %.6g" % r["action_absmax"])
    print("has_demonstration = False        %d / %d" % (r["n_no_demo"], n))
    print("frames in pi0.5 orientation      %d / %d" % (r["n_upright"], n))
    if r["drift_max"] is not None:
        print("circle_meta vs mask drift        %.2f px worst (unclipped rings only)"
              % r["drift_max"])
    print()

    ok = True
    if r["n_zero_action"] == n or r["n_no_demo"] == n:
        ok = False
        print("NOT TRAINABLE — every episode carries a zero action and/or is flagged")
        print("  has_demonstration=False. This is a validation export: scenes to be")
        print("  SCORED, not demonstrations to imitate. Fine-tuning on it yields a")
        print("  policy that emits zero actions and scores 0% on every rollout.")
        print("  Use a demonstration-bearing corpus (openvla/modified_libero_rlds")
        print("  with the sketch channel attached) for training.")
    else:
        print("TRAINABLE — an action signal is present.")

    if r["n_upright"] != n:
        ok = False
        print("WRONG ORIENTATION — %d / %d frames are upside-down relative to"
              % (n - r["n_upright"], n))
        print("  openvla/modified_libero_rlds, which is what pi0.5-LIBERO was")
        print("  fine-tuned on and what openpi's eval loop feeds the model.")
        print("  Re-run: python scripts/export_rlds_frames.py --suite <suite>")
        print("          python scripts/export_rlds_pack.py  --suite <suite>")
    else:
        print("ORIENTATION OK — frames match the modified_libero_rlds convention.")

    if r["drift_max"] is not None and r["drift_max"] > 3.0:
        ok = False
        print("COORDINATES ADRIFT — circle_meta is %.1f px from its own mask;"
              % r["drift_max"])
        print("  that is what a half-applied rotation looks like.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir",
                    help="RLDS version directory (e.g. outputs/rlds/<name>/1.0.0), "
                         "or its parent")
    args = ap.parse_args()

    data_dir = args.data_dir
    if not glob.glob(os.path.join(data_dir, "*.tfrecord-*")):
        nested = sorted(glob.glob(os.path.join(data_dir, "*", "*.tfrecord-*")))
        if nested:
            data_dir = os.path.dirname(nested[0])

    print("auditing %s\n" % data_dir)
    return 0 if report(audit(data_dir)) else 1


if __name__ == "__main__":
    sys.exit(main())
