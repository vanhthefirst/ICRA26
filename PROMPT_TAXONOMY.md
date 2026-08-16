# Prompt taxonomy — Sketch-Prompted VLA

Aaron — 16 August 2026

Two things in this benchmark can make a task hard, and until now I described
both with the word "ambiguous". That was a mistake: it made statements like
"an ambiguous scene with an explicit prompt" sound self-contradictory when they
are ordinary. This file separates them into two independent axes and fixes the
vocabulary. Everything below is the definition the code, the manifests and the
reports now use.

## The two axes

**Axis 1 — prompt quality.** A property of the caption alone. Read the string
with the picture covered up.

**Axis 2 — scene structure.** A property of the scene alone. Count the
candidates with the caption covered up. This is the four-tier split, and it has
not changed.

They are independent. Every one of the 114 scenes now carries both captions, so
all eight combinations of (prompt type x tier) exist and are measured.

---

## Axis 1 — explicit vs ambiguous prompts

| prompt type | definition | example |
|---|---|---|
| **explicit** | The caption names the target and the destination by category. | *"pick up the black bowl and place it on the plate"* |
| **ambiguous** | The caption names neither. It refers to both by pointing words alone. | *"move this onto that"* |

The test is one question: **does the caption say what kind of thing to pick up
and what kind of thing to put it on?** If yes it is explicit, if no it is
ambiguous. Nothing about the scene enters the test.

"Explicit" is the new name for what I have been calling "unambiguous". Every
caption authored by the three builders is an explicit caption, so the dataset as
it stood in July was an all-explicit dataset. The ambiguous captions are new.

### The ambiguous captions

Two strings, chosen by the scene's goal predicate:

| goal predicate | ambiguous caption |
|---|---|
| `On` | `move this onto that` |
| `In` | `move this into that` |

That is the entire vocabulary — 114 scenes draw on two strings. A per-scene
paraphrase would add a second uncontrolled variable next to the one I am
measuring, so the wording is held fixed and only the referring expressions
change. The sentence shape (one verb, one object slot, one destination slot)
matches the explicit caption, so what differs between the arms is the referring
expressions and nothing else.

"this" is the target the circle encloses; "that" is the destination the arrow
points at. The pair is deliberate: it is exactly the information a sketch
supplies, so the ambiguous arm is the condition a sketch is supposed to rescue.

`scripts/build_prompt_variants.py` derives them. It validates every caption
against a hard-coded allow-list of the two strings above, so a caption cannot
leak an object name through a template bug.

### What does *not* make a prompt ambiguous

**Adding objects to the scene does not.** *"pick up the black bowl and place it
on the plate"* stays explicit when I put five black bowls on the table. The
caption still says *black bowl* and *plate*; the scene just makes that
description match more than one thing. That is axis 2, and it is why the old
single word was wrong.

---

## Axis 2 — the four tiers

A tier counts candidates in the scene. Write it as (candidate targets to
candidate destinations):

| tier | shape | meaning |
|---|---|---|
| `control` | one-to-one | one candidate target, one candidate destination |
| `referential` | many-to-one | several interchangeable targets, one destination |
| `directional` | one-to-many | one target, several plausible destinations |
| `both` | many-to-many | several of each |

A "candidate" is an instance the explicit caption's category words match — the
same-category siblings for the target, the destination-typed instances for the
destination. Counts per suite are unchanged: control 5, referential 12,
directional 9, both 12, in each of the three suites.

Note on the name: the tier is written `directional` everywhere in the code and
the manifests. I sometimes say "direction" in conversation; the string on disk
is `directional`.

---

## The eight cells, and what each one measures

|  | explicit prompt | ambiguous prompt |
|---|---|---|
| **control** | fully specified — the ceiling for this scene set | can the policy act at all with no names? |
| **referential** | which of the identical targets? | no names *and* several targets |
| **directional** | which of the destinations? | no names *and* several destinations |
| **both** | both open | the hardest cell |

Two cells carry most of the argument:

- **control x explicit** is the only cell with nothing missing. Whatever it
  loses against the standard LIBERO suites is distribution shift, not ambiguity.
- **control x ambiguous** is new and it is the clean measurement of the caption
  on its own. There is one bowl and one plate, so the scene answers the question
  the caption refuses to; if the policy still fails here, the failure is about
  the words, not about choosing between candidates.

---

## What this changed on disk

Nothing physical. Same 114 BDDL files, same `init_state.npz`, same sketches,
same tiers. Only captions were added:

- Every `tokens.json` and `meta.json` gains `instruction_explicit`,
  `instruction_ambiguous` and `prompt_template_id`. The legacy `instruction` key
  is untouched and still equals the explicit caption, so nothing that reads it
  breaks.
- `outputs/validation_set_<suite>/evaluation_rows.json` — 76 rows each.
- `outputs/evaluation_rows_all.json` — **228 rows**, keyed by
  `(suite, dir, prompt_type)` with a `scene_id` of the form
  `spatial/scene_0000#ambiguous`.

The scene manifests (`manifest_canonical.json`,
`validation_manifest_all.json`) are untouched and still list the 114 physical
scenes keyed on `(suite, dir)`. The evaluation manifests are the ones with the
prompt axis in them. Two files, two jobs — not two versions of one file.

Because the two arms share the physical scene, they are paired by construction.
The harness seeds rollout *k* of a scene identically in both arms
(`stable_seed` does not read the prompt type), so an explicit row and an
ambiguous row can be compared one to one.

## Running an arm

```bash
python scripts/rollout_sketch.py --policy pi05 --conditions auto \
    --prompt-type explicit  --run-id pi05_explicit_532  --n-rollouts 14
python scripts/rollout_sketch.py --policy pi05 --conditions auto \
    --prompt-type ambiguous --run-id pi05_ambiguous_532 --n-rollouts 14
```

One `--run-id` per arm, and the harness enforces it: both arms write the same
condition label, so a shared run-id would let the second arm resume as
already-done and silently report the first arm twice. Full sequence in
`RUNBOOK_BASELINES.md`.
