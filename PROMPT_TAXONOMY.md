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
| **ambiguous** | The caption names neither. It refers to both by pointing words alone. | *"pick this up and place it on that"* |

The test is one question: **does the caption say what kind of thing to pick up
and what kind of thing to put it on?** If yes it is explicit, if no it is
ambiguous. Nothing about the scene enters the test.

"Explicit" is the new name for what I have been calling "unambiguous". Every
caption authored by the three builders is an explicit caption, so the dataset as
it stood in July was an all-explicit dataset. The ambiguous captions are new.

### The ambiguous captions

An ambiguous caption must remove the object identities **and nothing else**. If
it also changed the verb, the two arms would differ in what the robot is asked
to *do* as well as what it is asked to do it *to*, and the arm would stop
isolating referring expressions.

So the bank is keyed on the shape of the scene's own explicit caption, and each
scene draws only from its own bucket:

| bucket | explicit caption looks like | scenes | templates |
|---|---|---|---|
| `two_clause_On` | *pick up the black bowl and place it on the plate* | 38 | 6 |
| `two_clause_In` | *grab the milk and put it in the basket* | 38 | 6 |
| `one_clause_On` | *put the bowl on the stove* | 36 | 6 |
| `push` | *push the plate to the front of the stove* | 2 | 2 |
| `one_clause_In` | — no scene currently | 0 | 3 |

**20 distinct ambiguous captions in use.** Examples: *"pick this up and place it
on that"*, *"grab it and place it in there"*, *"take this and set it on that"*,
*"put it on that one"*, *"push it towards that one"*. The verb synonyms are the
ones the explicit captions already use — pick up / grab / take, put / place /
set — so the two arms draw on the same verb vocabulary.

Why a bank rather than one string per predicate: with one string the arm
measures the policy's response to *that sentence*, and any lexical quirk of it
becomes the result. A bank measures the response to ambiguity as a class, and
`prompt_template_id` on every row lets the analysis check whether wording
mattered at all.

**Word order is fixed.** The first deictic is the target the circle encloses,
the second is the destination the arrow points at. The stage-5 sketch arms
append geometry to these strings and depend on that order.

**No template may contain a word that names anything in a scene.**
`check_no_identity_leak()` builds the vocabulary from the object instances
themselves — 41 words — and refuses any template that collides. This is why
*"on top of that"* and *"to the front of that"* are absent however natural they
read: `top` and `front` appear in `wooden_cabinet_1_top_region` and
`main_table_stove_front_region`.

**Assignment is deterministic and balanced.** Round-robin within
(bucket, suite, tier), scenes sorted by directory, each group starting at its
own blake2b-derived offset in the bank. Round-robin within tier so no template
can concentrate in one tier and confound the tier comparison; the offset so a
group smaller than its bank still reaches the tail of it. blake2b rather than
`hash()`, which Python salts per process — the bug that cost this project a
reproducible baseline once already.

Every tier sees 15 or more distinct templates, and no template is used more
than 7 times.

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
