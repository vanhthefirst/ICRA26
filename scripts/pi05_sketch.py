"""
Sketch-Prompted VLA -- getting a sketch into pi0.5, which has no sketch channel.

The baseline report established the headroom: pi0.5-LIBERO scores 34.5% on the
114 scenes against 96.0% on the standard suites, and roughly 45 of the missing
points are referential ambiguity rather than distribution shift. This module is
the first attempt to recover them WITHOUT training anything -- two ways of
smuggling the sketch into a stock checkpoint's existing inputs:

    overlay   the circle and arrow are composited onto the image the model sees
    language  the same circle and arrow are described in words and appended to
              the instruction

They are a matched pair, not alternatives, and the pairing is the point. pi0.5
was never trained on images containing annotation marks, so a null result from
`overlay` alone is ambiguous: it could mean the model cannot read marks, or it
could mean the model cannot use referential information from any source on these
scenes. `language` speaks to the model in the modality it was trained on and so
bounds what ANY prompt could recover here. Read together:

    overlay works                    -> a pretrained VLA can exploit marks zero-shot
    overlay fails, language works    -> the information suffices; the MODALITY is
                                        the barrier, and fine-tuning is the answer
    both fail                        -> the bottleneck is not referential at all,
                                        and the baseline report's attribution of
                                        ~45 points to ambiguity needs revisiting

THE FIREWALL STILL HOLDS
------------------------
Both adapters read `symbolic_tokens` and nothing else -- {circle:{cx,cy,rx,ry},
arrow:{x0,y0,x1,y1}}. Neither ever sees `meta['target']`, `meta['destination']`,
`pick_px` or `place_px`. The language string is derived from the SKETCH's
geometry, so it carries exactly the information a human annotator drew and not
one bit of ground truth beyond it. A language paraphrase built from
`meta['target']` would be a different experiment and a dishonest one.

TWO COORDINATE FACTS THAT MUST NOT BE CONFUSED
----------------------------------------------
Tokens live in the 128x128 `frame0.png` space, in the orientation the projection
path uses. The model is fed a 256-rendered frame, rotated 180 degrees, then
pad-resized to 224. The two adapters therefore handle the rotation differently,
and getting this backwards is silent:

 1. OVERLAY draws on the raw 256 frame BEFORE the rotation, at 2x the token
    coordinates. The marks are then rotated along with the pixels they annotate,
    so they stay attached to their objects automatically. No coordinate
    transform is needed beyond the scale.
 2. LANGUAGE must describe the image AS THE MODEL SEES IT, which is the rotated
    one. A circle at the top-left of `frame0` is at the bottom-right of the
    model's view, so the token coordinates are rotated (x -> W-1-x,
    y -> H-1-y) before being turned into words. Describing the unrotated frame
    would hand the model a confidently inverted instruction -- worse than no
    instruction at all.

Colours match the builders' own scheme (build_validation_set_spatial.py):
circle green (0,200,0), arrow red (200,50,50), in RGB.
"""

import numpy as np

# The builders' sketch palette, RGB. Kept identical so an overlaid frame looks
# like the sketch.png an annotator saw rather than a new visual language.
CIRCLE_RGB = (0, 200, 0)
ARROW_RGB = (200, 50, 50)

TOKEN_CANVAS = 128          # the space symbolic_tokens are expressed in


# ------------------------------------------------------------------ overlay --
def draw_tokens(img_rgb, tokens, canvas=TOKEN_CANVAS, thickness=None):
    """Composite the circle and arrow onto `img_rgb`, returning a new array.

    Marks are REDRAWN from the token geometry at the target resolution rather
    than upsampled from the 128px sketch.png: upsampling a 1px anti-aliased
    stroke by 2x produces a soft grey smear, and whether the model can see a
    mark at all is the entire question being asked.

    `img_rgb` is the raw simulator frame, in frame0's orientation. Call this
    BEFORE any 180-degree rotation -- see the module docstring.
    """
    import cv2

    out = np.ascontiguousarray(np.asarray(img_rgb).copy())
    h, w = out.shape[:2]
    s = float(w) / float(canvas)
    if thickness is None:
        # 1-2px at 128 is the builders' stroke; scale it so the mark occupies
        # the same fraction of the image at any render size.
        thickness = max(1, int(round(2 * s)))

    circ = tokens.get("circle")
    if circ:
        cx, cy = int(round(circ["cx"] * s)), int(round(circ["cy"] * s))
        rx, ry = int(round(circ["rx"] * s)), int(round(circ["ry"] * s))
        cv2.ellipse(out, (cx, cy), (max(rx, 1), max(ry, 1)), 0, 0, 360,
                    CIRCLE_RGB, thickness, cv2.LINE_AA)

    arr = tokens.get("arrow")
    if arr:
        p = (int(round(arr["x0"] * s)), int(round(arr["y0"] * s)))
        q = (int(round(arr["x1"] * s)), int(round(arr["y1"] * s)))
        cv2.arrowedLine(out, p, q, ARROW_RGB, thickness, cv2.LINE_AA,
                        tipLength=0.35)
    return out


# ----------------------------------------------------------------- language --
# A 3x3 grid over the image. Coarse on purpose: the sketch localises an object
# to a few pixels, but a phrase like "at 94, 40" is not language the model was
# trained on, and a nine-cell grid is roughly the granularity a person would use
# unprompted ("the one at the top right").
_COLS = ("left", "centre", "right")
_ROWS = ("top", "middle", "bottom")


def _cell(x, y, canvas=TOKEN_CANVAS):
    ci = min(int(x * 3 // canvas), 2)
    ri = min(int(y * 3 // canvas), 2)
    return ri, ci


def _phrase(x, y, canvas=TOKEN_CANVAS):
    ri, ci = _cell(x, y, canvas)
    row, col = _ROWS[ri], _COLS[ci]
    if row == "middle" and col == "centre":
        return "centre"
    if row == "middle":
        return col                      # "left" / "right"
    if col == "centre":
        return row                      # "top" / "bottom"
    return "%s-%s" % (row, col)         # "top-left" etc.


def rotate180_xy(x, y, canvas=TOKEN_CANVAS):
    """Token coordinates -> the same point in the 180-degree-rotated view the
    model is actually shown."""
    return (canvas - 1 - x), (canvas - 1 - y)


def describe_tokens(instruction, tokens, canvas=TOKEN_CANVAS, rotate180=True):
    """Turn the sketch into a deictic clause appended to the instruction.

    `rotate180=True` describes the frame the MODEL sees. Setting it False
    describes frame0's orientation, which is what a human annotator saw and
    what the projection path uses -- useful only as a deliberate control, since
    feeding it to the model inverts every direction.

    Returns the instruction unchanged if the tokens carry no circle, so a
    malformed sketch degrades to the text-only baseline rather than to a
    confidently wrong sentence.
    """
    circ = tokens.get("circle") if tokens else None
    if not circ:
        return instruction

    cx, cy = float(circ["cx"]), float(circ["cy"])
    if rotate180:
        cx, cy = rotate180_xy(cx, cy, canvas)
    pick_where = _phrase(cx, cy, canvas)

    arr = tokens.get("arrow") if tokens else None
    if arr:
        ax, ay = float(arr["x1"]), float(arr["y1"])     # head, not tail
        if rotate180:
            ax, ay = rotate180_xy(ax, ay, canvas)
        place_where = _phrase(ax, ay, canvas)
        clause = ("Pick the one at the %s of the image, and place it at the %s "
                  "of the image." % (pick_where, place_where))
    else:
        clause = "Pick the one at the %s of the image." % (pick_where,)

    sep = "" if instruction.rstrip().endswith((".", "!", "?")) else "."
    return "%s%s %s" % (instruction.rstrip(), sep, clause)
