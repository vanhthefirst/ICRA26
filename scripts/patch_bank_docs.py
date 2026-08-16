"""One-shot: bring the docs onto the ambiguous-caption BANK (20 templates in 5
buckets) after they were written for a two-string version. Delete once run."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TAXONOMY_OLD = """### The ambiguous captions

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
"""

TAXONOMY_NEW = """### The ambiguous captions

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
"""

EDITS = {
    "PROMPT_TAXONOMY.md": [
        (TAXONOMY_OLD, TAXONOMY_NEW),
        ("| ambiguous | The caption names neither. It refers to both by pointing words alone. | *\"move this onto that\"* |",
         "| ambiguous | The caption names neither. It refers to both by pointing words alone. | *\"pick this up and place it on that\"* |"),
    ],
    "RUNBOOK_BASELINES.md": [
        ("""**The gate:** the policy echoes `prompt -> '...'` whenever the string it sends
the server changes. For the ambiguous arm every line must read
`move this onto that` or `move this into that` and must contain no object name.
If it prints `pick up the black bowl ...`, the arm is running on explicit
captions and the whole ambiguous baseline would be a duplicate of the explicit
one. Stop and fix it before Part D.""",
         """**The gate:** the policy echoes `prompt -> '...'` whenever the string it sends
the server changes. For the ambiguous arm every line must name **no object** —
only pointing words (*this*, *that*, *it*, *that one*, *there*). There are 20
templates, so the strings differ between scenes; what must never appear is a
category noun. If it prints `pick up the black bowl ...`, the arm is running on
explicit captions and the whole ambiguous baseline would be a duplicate of the
explicit one. Stop and fix it before Part D."""),
    ],
    "SCHEMA.md": [
        ("| `instruction_ambiguous` | str | caption naming neither (`\"move this onto that\"`) |",
         "| `instruction_ambiguous` | str | caption naming neither (`\"pick this up and place it on that\"`); one of 20 templates |"),
        ("| `prompt_template_id` | obj | `{explicit, ambiguous}` — provenance of each caption |",
         "| `prompt_template_id` | obj | `{explicit, ambiguous}`; the ambiguous value is `<bucket>/<index>`, e.g. `two_clause_On/03` |"),
    ],
}

DATASHEET_OLD = "| ambiguous | `instruction_ambiguous` | `move this onto that` / `move this into that` |"
DATASHEET_NEW = "| ambiguous | `instruction_ambiguous` | one of 20 templates, e.g. `pick this up and place it on that` |"
for _s in ("spatial", "object", "goal"):
    EDITS[f"outputs/validation_set_{_s}/DATASHEET.md"] = [(DATASHEET_OLD, DATASHEET_NEW)]

missed = []
for rel, edits in EDITS.items():
    path = os.path.join(_REPO, rel)
    text = original = open(path, encoding="utf-8").read()
    for old, new in edits:
        if new in text and old not in text:
            continue
        if old not in text:
            missed.append(f"{rel}: {old.splitlines()[0][:70]!r}")
            continue
        text = text.replace(old, new, 1)
    if text != original:
        open(path, "w", encoding="utf-8").write(text)
        print("patched", rel)

for m in missed:
    print("  NOT FOUND:", m, file=sys.stderr)
sys.exit(1 if missed else 0)
