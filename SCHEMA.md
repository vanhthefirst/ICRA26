# DrawVLA validation-set schema (v1.0)

One format across all suites (Spatial, Object, Goal — all built), so a single
loader / eval harness reads every scene. Established by cross-checking the
Spatial and Object sets; all three are normalised to it by
`scripts/normalize_validation_schema.py` (additive — legacy keys preserved).
Combined: `outputs/validation_manifest_all.json`, 114 scenes (38 × 3).

## Directory layout

```
outputs/validation_set_<suite>/
    scene_0000/ ... scene_00NN/
        scene.bddl           # the LIBERO task (ships so a GPU rollout can score it)
        frame0.png           # 128x128 agentview, settled scene, no sketch
        sketch.png           # frame0 + hand-style circle (target) + arrow (destination)
        target_vismask.png   # 128x128 grayscale RGB-diff silhouette of the target
        tokens.json          # model-facing: instruction + sketch geometry + canonical ids
        meta.json            # full provenance + all gate measurements
    manifest.json            # per-suite summary (legacy shape, suite-specific)
    manifest_canonical.json  # per-suite summary in the canonical shape below
    DATASHEET.md
    contact_sheet.png
outputs/validation_manifest_all.json   # every scene, every suite, canonical shape
```

**Keying the combined manifest.** In `validation_manifest_all.json`, `dir` is
NOT unique — each suite numbers its scenes from `scene_0000`, so every `dir`
collides across suites (e.g. `scene_0000` exists in both spatial and object), and
seeds overlap too. Any consumer of the combined manifest MUST key on the pair
`(suite, dir)`, never `dir` or `seed` alone. Each row carries `suite` for exactly
this reason. (Within a single suite's `manifest_canonical.json`, `dir` alone is
unique.)

## Canonical fields (present in EVERY scene, every suite)

These are the contract. Read these, not the suite-specific/legacy keys.

| field | type | meaning |
|---|---|---|
| `schema_version` | str | `"1.0"` |
| `suite` | str | `"spatial"` \| `"object"` \| `"goal"` |
| `tier` | str | `control` \| `referential` \| `directional` \| `both` |
| `target` | str | instance to pick; the **circle** encloses this |
| `destination` | str | instance to move to; the **arrow** points here |
| `destination_region` | str | exact 2nd argument of the BDDL goal predicate |
| `goal_predicate` | str | `"On"` \| `"In"` |
| `instruction` | str | vague natural-language caption (ambiguous by construction) |
| `symbolic_tokens` | obj | `{circle:{cx,cy,rx,ry}, arrow:{x0,y0,x1,y1}}` (pixels) |
| `pick_px` | [int,int] | target centre in image pixels `[col,row]` |
| `place_px` | [int,int] | destination point in image pixels `[col,row]` |
| `radius` | int | drawn circle radius (px) |
| `camera_matrix` | 4x4 | world→pixel transform for this scene |
| `visibility` | obj | `{v_visible, v_full, visibility}` (RGB-diff occlusion fraction) |
| `grasp` | obj | `{grasp_success, lift, close_sign, ...}` |
| `clearance_xy` | float | target→nearest-neighbour xy distance (m) |
| `seed` | int | reproduces the scene exactly |
| `oracle_success` | bool | teleporting target→destination scores the goal True |

`tokens.json` carries: `instruction`, `target`, `destination`,
`destination_region`, `goal_predicate`, `suite`, `tier`, `symbolic_tokens`.
`meta.json` is a superset (all of the above + suite-specific + legacy).

### `destination_region` — why it differs from `destination`

The BDDL goal's second argument is not always the destination *instance*:

- **Spatial** `(On bowl_1 plate_1)` → `destination = destination_region = plate_1`
- **Object** `(In milk_2 basket_2_contain_region)` →
  `destination = basket_2`, `destination_region = basket_2_contain_region`

A scorer should always compare against `destination_region`; a renderer/UX that
wants the physical object should use `destination`. For Goal, region-typed
destinations (e.g. `main_table_stove_front_region`) will set both equal, like
Spatial's `On`.

## Suite-specific fields (kept, not part of the cross-suite contract)

- **Spatial**: `target_bowl`, `target_plate`, `counts`.
- **Object**: `target_cat`, `dest`, `n_target`, `n_basket`, `distractors`,
  `siblings`, `other_baskets`, `oracle_negatives`, `px_extent`,
  `px_sep_siblings`/`px_req_siblings`, `px_sep_baskets`/`px_req_baskets`,
  `tilt_deg`, `all_pixels`.

Legacy keys (`target_bowl`, `target_plate`, `dest`) remain readable; new code
should prefer the canonical `target` / `destination` / `destination_region`.

## Rules for the Goal builder (to stay in-schema)

1. Emit every canonical field above; set `suite="goal"`, `goal_predicate` from
   the task (`On` or `In`).
2. For an object-instance destination, set `destination` and
   `destination_region` equal (as Spatial does). For a container/region
   destination, set `destination_region` to the exact BDDL arg and
   `destination` to the owning object/fixture.
3. Reuse Object's hardened gate stack: settled, in-frame, visibility ≥ 0.35,
   positive oracle, **negative oracles that gate** (wrong target / wrong
   destination must score False), pixel-separation resolvability, graspable.
4. Run `normalize_validation_schema.py` after building (add `"goal"` to `SETS`)
   to emit `manifest_canonical.json` and refresh `validation_manifest_all.json`.

## Verification

`normalize_validation_schema.py` is idempotent and verifies each write. An
independent audit confirmed, for all 76 current scenes, that canonical `target`
and `destination_region` equal the re-parsed `scene.bddl` goal arguments.
