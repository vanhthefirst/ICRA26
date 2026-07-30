# Meeting script — data creation method (Sketch-Prompted VLA validation set)

Aaron · 2026-07-30 · spoken script, ~4 minutes. Not for commit.

---

## 1. The task, and how I framed it

The brief was: build a new dataset by augmenting LIBERO — for example, a scene
that says "pick and place the bowl in the middle of the table" gets one or more
identical bowls added — to prove that text alone is often not enough to specify
the task, and that a sketch modality is needed for the model to capture spatial
features.

My framing: it is not enough for a scene to *look* ambiguous — I have to be able
to **prove** each scene is unsolvable from text alone, and score it
automatically. So I built every scene around three requirements: the language
stays vague by construction, the BDDL goal names one specific instance so a
rollout is machine-scorable, and the scene passes an oracle certification that
the ambiguity is real. The disambiguating signal is a hand-style sketch drawn on
the camera image: a circle around the intended object, an arrow to the intended
destination, placed by projecting ground-truth 3D positions to pixels.

## 2. What I built

Three suites covering three LIBERO task families — **114 scenes, 38 each**:

- **Spatial** — tabletop `On(bowl, plate)`, scenes authored from scratch with
  duplicated near-identical bowls and multiple plates.
- **Object** — floor workspace, `In(grocery, basket)`, duplicated lookalike
  groceries and multiple baskets.
- **Goal** — LIBERO's own bespoke scenes with real fixtures (stove, cabinet,
  wine rack). Here I don't author from scratch: I inject duplicate instances
  into the shipped BDDLs, keeping every fixture and region intact and
  retargeting only the goal.

Each suite has four graded tiers: **control** (5, unambiguous baseline),
**referential** (12, *which object*), **directional** (9, *which destination*),
and **both** (12).

Every scene must pass a gate stack before it ships: physics settled; the target
silhouette fully in frame; visibility ≥ 0.35; not already solved; a **positive
oracle** — teleporting the target to the destination satisfies the goal; and the
key one, **negative oracles** — moving the *wrong* object, or moving to the
*wrong* destination, must **fail**, for every wrong candidate. The negatives are
what certify the scene is genuinely impossible from text alone. On top of that,
pixel-separation gates guarantee the circle and arrow are visually resolvable at
128×128.

Everything is normalised to one schema (v1.0), and a read-only audit script
re-parses every BDDL from disk and re-checks the claims independently — current
state: **114/114 clean**.

## 3. How the trained model validates against this set

The set is a held-out benchmark, not training data. The protocol is an A/B
rollout: run the trained policy on each scene **twice — text-only versus
text + sketch**. Because every scene is certified unsolvable from text, a
text-only policy is forced to guess: on a referential scene with 4 bowls its
expected success is ~25%. A policy that genuinely uses the sketch should recover
most of that gap. **The success-rate gap is the headline number** — the measured
value of sketch prompting. Scoring is automatic: each scene ships its BDDL and
success check.

The tiers make it diagnostic, not just a single number: control isolates "did
adding the sketch hurt basic competence"; referential vs directional separates
*which-object* from *which-destination* reasoning; both is the hardest case.
This holds regardless of how the training team conditions on the sketch — the
data ships both the rendered sketch image and the symbolic geometry tokens, so
either input route works.

## 4. Limitations, honestly stated

1. **Goal's hard tiers rest on 2 tasks.** Only 2 of 7 usable Goal tasks have
   duplicable object destinations, so directional/both scenes come from those
   two (27 of 38 scenes). Structural to LIBERO-Goal, disclosed in the
   datasheet; Goal results should be read with the per-task breakdown.
2. **One circle + one arrow expresses one action.** LIBERO-Long/libero_10 tasks
   have 2–3 predicates, so that suite is postponed — the sketch language itself
   would need to grow (numbered strokes or a sketch sequence).
3. **Oracles certify goal semantics, not kinematic feasibility.** The teleport
   oracle proves the goal is satisfiable; a scripted top-down grasp is recorded
   but doesn't gate in Goal — 6 scenes carry `grasp_success=False` (wine bottle,
   plate). A strong policy may still solve them; the scripted probe is crude.
4. **Sketches are synthetic.** Programmatic hand-style wobble, not real human
   drawings — there is a domain gap to actual user sketches, single fixed
   camera, 128×128.
5. **Scale**: 38 scenes per suite is enough for a significant A/B gap, but per
   tier-per-task cells are small.

Future work, in order of value: collect a small human-drawn sketch subset to
measure the synthetic-to-human gap; extend the sketch language to multi-step for
LIBERO-Long; add camera viewpoints and occlusion-heavy tiers; scale scene counts
once the pipeline's throughput matters.

---

*Close: the pipeline is deterministic (every scene reproducible from its seed),
fully audited, and the benchmark is ready for the training team's first
rollout — that final step needs GPU access, which is the one thing not run yet.*
