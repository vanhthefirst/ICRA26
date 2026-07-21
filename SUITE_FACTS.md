# DrawVLA — hard-won facts about LIBERO scene synthesis

Durable notes for the validation-set builders. These were expensive to derive;
do not re-derive them. Evidence lives in `outputs/probe_*.txt`.

Status: **Spatial** done (`outputs/validation_set_hardened/`, 38 scenes).
**Object** done (`outputs/validation_set_object/`, 38 scenes).
**Goal** next. **Long / libero_10** postponed (multi-predicate goals; a single
circle+arrow cannot express two actions).

---

## 1. General (apply to every suite)

1. `obs["agentview_image"]` is **already correctly oriented** — do not flip it.
   Mirror the row instead: `project_points_from_world_to_camera` returns
   `(row, col)`; use `col = int(p[1])`, `row = (IMG_H-1) - int(p[0])`.
   Flipping the frame double-mirrors and misaligns everything.
2. `OffScreenRenderEnv` has **no** `.action_dim` (use 7) and no `._get_observations()`.
3. Gripper close sign is `-1.0` for the scripted grasp (try both signs; some
   objects need `+1.0`).
4. Object position: use the `contype=0 / conaffinity=0` reference geom's `xpos`
   ("visual centre"), **not** `body_xpos` (which sits at the object's bottom).
   *Exception: predicate evaluation — see §3.2.*
5. Success check: `env.check_success()` or `env.env._check_success()`.
6. Free joints are named `<instance>_joint0`; set pose via `model.jnt_qposadr` /
   `jnt_dofadr`.
7. Segmentation-based ID mapping **fails** (empty masks). Use the RGB-difference
   visibility method. Ratios marginally > 1.0 are a benign shadow artifact.
8. Writing across the WSL↔Windows DrvFs bridge can **silently truncate**. Verify
   every write (bytes > 0, PNG `IEND` present, re-decodes). Read back with
   `cv2.IMREAD_UNCHANGED` — plain `imread` returns 3 channels for a grayscale
   mask and a naive `shape ==` check will false-alarm.
9. Body names may need a `_main` suffix when looking up; resolve both.
10. **Rejection sampling needs RESTARTS, not more tries.** A bad prefix (two
    baskets placed badly) can never be repaired by further attempts at the
    remaining item. Throw the whole layout away and redraw. This bug made 3
    baskets look "geometrically impossible" when they fit comfortably.

## 2. Suite shapes (from `outputs/probe_suites.txt`)

| suite | goals | scene contents |
|---|---|---|
| libero_spatial | 10 × `On` | 2 bowls, cookies, ramekin, plate; table |
| libero_object | 10 × `In` | 1 basket + 6 distinct groceries; **floor** |
| libero_goal | 7 `On`, 1 `In`, 1 `Open`, 1 `Turnon` | bowl, cream cheese, wine bottle, plate |
| libero_10 | 9 of 10 have 2–3 predicates | ~10 distinct scenes, 8 fixture types |

For **goal**: use only the 8 `On`/`In` pick-and-place tasks. Skip "open the
drawer" / "turn on the stove" — no object to circle. Some destinations are
regions (`main_table_stove_front_region`), which limits duplicating destinations;
duplicating the bowl/plate works fine.

## 3. libero_object specifics

### 3.1 Container regions must be DECLARED
The destination is a container region, and it only exists if the BDDL declares it:

```
(contain_region
    (:target basket_1)
)
```

One block per basket. Omitting it raises `KeyError: basket_2_contain_region` at
the first `step()` (inside `reward()` → `_check_success`). Two blocks sharing the
region name `contain_region` do **not** collide — keys are `{target}_{name}`.

### 3.2 `In` tests body_xpos, and is permissive
`In(a,b) = b.check_contact(a) and b.check_contain(a)`. For a **site** region
`check_contact` is hardcoded `True`, so `In` reduces to
`in_box(site_xpos, site_xmat, body_xpos)` — the object's **body origin**, not its
visual centre. The teleport oracle must therefore aim the body origin at the
contain **site**.

The box is half-extent `(0.061, 0.061, 0.070)` centred at z≈0.067, so it reaches
the floor: an object merely standing at a basket's xy counts as contained. The
positive oracle is consequently a weak check — the **negative** oracles (wrong
basket, wrong instance) are what certify a scene.

### 3.3 Floor workspace is banded
Baskets belong at +y (stock bin sits at y≈0.26), groceries in front.
Basket footprint ≈ 0.15 × 0.15 → baskets need ≈ 0.215 m separation.

### 3.4 Camera scale (measured, 223 object pairs)
**≈101 px/m**, ranging 71–140 px/m with depth. Use ≈85 px/m for conservative
sizing. Pixel **row tracks world x**; pixel **col tracks world y**.

### 3.5 Object size caps how many copies fit
Measured projected half-extents (px) and the resulting cap for the 0.38 × 0.235
grocery band:

| category | half-extent | max resolvable copies |
|---|---|---|
| ketchup, orange_juice, salad_dressing, milk | 11.5–12.7 | **3** |
| bbq_sauce, tomato_sauce, alphabet_soup | 8.1–9.0 | 5 |
| chocolate_pudding, cream_cheese, butter | 4.1–5.2 | 5 |

Five identical ketchup bottles **cannot** be resolvably placed in this band at
any seed. Demanding it produced 172 wasted rejections.

### 3.6 Tilt is not a usability signal
`tilt_deg` (body z-axis vs world z) reads ~90° for flat groceries *at rest* —
their mesh frame is z-horizontal. It is a per-category constant, not a pose
defect. Record it; do not gate on it. Gating on it rejected 12 of 16 scenes.

## 4. Design principle that kept paying off

**Satisfy a gate by construction, not by rejection.** Every time a gate produced
mass rejections, the fix was to change *placement* so the constraint holds by
design (basket separation 0.175 → 0.215 m; per-category sibling spacing), not to
loosen the gate. The one exception is when the gate asks for something
geometrically impossible — then the gate itself is wrong (see §3.5, and the
circle criterion: "sibling centre outside the circle", not "circle touches no
pixel of the sibling").

## 5. Verification habit

Do not trust the build log. Re-read the produced artifacts and re-check the
claims independently: goal string vs meta, contain-region declaration count,
negative-oracle *coverage* (not just values), caption never naming an instance,
and — the sharpest check — that no sibling centre falls inside the **actually
drawn** circle using the recorded `rx`/`ry` tokens.
