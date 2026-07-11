# Sketch-Prompted VLA: Circle-and-Arrow Visual Instructions for Disambiguated Manipulation

**Working title options:** *DrawVLA*, *ScribbleVLA*, *Point-and-Push VLA*, *Deictic-VLA*

## 0. One-line thesis

Language under-specifies "which one" and "to where" in cluttered or duplicate-object scenes. A human-drawn **circle** (a deictic reference: *this* object) plus a human-drawn **arrow** (a coarse motion goal: move it *there*) overlaid on the current observation is a low-effort, expressive instruction channel. We fold these two primitives into a latent-action VLA (UniVLA), train it for free from LIBERO using simulator ground truth, and show it resolves referential and directional ambiguity that text-only and goal-image policies cannot.

## 1. Motivation

Three goal-specification modalities dominate manipulation policies, each with a failure mode.

Natural language is fast to provide but ambiguous. "Pick up the apple" is undefined when the table holds five apples. Spatial qualifiers ("the left one", "the one behind the cup") help only when the scene admits a clean verbal description, which cluttered or near-duplicate scenes do not.

Goal images are spatially precise but over-specified and hard to provide on the fly. They force the human to produce or imagine a full desired scene, and they entangle task-relevant change with irrelevant pixels.

Sketches and pointing sit between the two. RT-Sketch and RT-Trajectory established that hand-drawn input is easy to give yet spatially grounded. But RT-Sketch encodes a full *goal scene* and RT-Trajectory encodes a dense *path*, both of which still ask the human for more than they often want to give. Our claim is that two minimal primitives carry most of the disambiguating signal:

- a **circle** answers *which* (referential grounding, resolving identical-distractor ambiguity),
- an **arrow** answers *where / which direction* (a coarse motion or destination goal).

This pairing maps cleanly onto how people already gesture at the world, and it costs the human two strokes.

There is a second, deeper motivation that connects to the latent-action line of work and to depth-augmented latent actions. An arrow drawn on the image plane is, almost literally, a human-specified **coarse latent action**: a direction (and, through its length, a magnitude in image space) for how the scene should change. UniVLA's central object is exactly a task-centric latent action derived from how frames change. So the arrow is not an auxiliary hint bolted onto the policy; it is a human prior over the very latent action UniVLA already learns. Image-plane arrow length is ambiguous about true 3D magnitude (the same RGB-magnitude ambiguity that motivates depth-augmented latent actions), which gives a natural depth extension and ties this project to existing depth-in-latent-action work.

## 2. Creating the dataset from LIBERO (no human labeling for training)

The core feasibility argument: **every training annotation can be generated automatically**, because LIBERO is a simulator with ground-truth object identity, object and end-effector poses, camera matrices, and the demonstration trajectory itself. The human is needed only at test time.

### 2.1 What LIBERO gives us

LIBERO ships HDF5 demonstrations (roughly 50 per task across the Spatial, Object, Goal, Long suites, plus the LIBERO-90 pool), each storing a `states` sequence that can be replayed in robosuite/MuJoCo. From a replay we can recover, per frame: RGB from `agentview` (and other views), the camera intrinsics and extrinsics, every object's 6D pose, the end-effector pose and gripper state, and a rendered **segmentation mask** per object. The task's BDDL goal predicate names the manipulated object and the destination region or container (`On(obj, region)`, `In(obj, container)`), which gives us both endpoints of the intended motion. The LIBERO+ / SlotVLA extension already publishes pixel-level masks, bounding boxes, and instance-level temporal IDs for LIBERO, so the segmentation step can be reused rather than re-derived.

### 2.2 Auto-generating the circle (the "which")

For the target object (from the BDDL goal plus parsed instruction), take its mask or bounding box on the relevant frame. Fit an enclosing circle or ellipse around the centroid with radius proportional to the box diagonal. Then apply **human-imprecision augmentation** so the synthetic strokes resemble real drawings: jitter the center, scale the radius, add eccentricity, wobble the contour with spline or Perlin noise, vary stroke width and color, and sometimes render partial or double circles. This sim-to-human style augmentation is what lets a test-time human draw messily and still be understood.

### 2.3 Auto-generating the arrow (the "where / direction")

Two arrow flavors, both derived from ground truth:

1. **Destination arrow** (pick-and-place, "move X to Y"): project the target object centroid (initial) and the BDDL destination region centroid (final) into the image plane using the known camera matrix, then draw a straight or two-segment arrow between them.
2. **Motion arrow** (direction emphasis): project the end-effector or grasped-object 3D trajectory into the image plane, simplify it to a coarse arrow (grasp point to release point), and keep only direction and rough length rather than the full path. This is the deliberately lighter-weight cousin of an RT-Trajectory sketch.

Apply the same imprecision augmentation (arrowhead style, curvature, length noise) as for circles.

### 2.4 The text channel, and manufacturing ambiguity

The value of the drawing only shows up when text alone is insufficient. So we generate, for a controlled fraction of episodes, a **degraded caption**: replace "pick up the milk on the left" with "pick up the milk", "grab this", or "move this there". In LIBERO-Object and LIBERO-Spatial, where multiple similar or identical objects exist, pair an ambiguous caption with a circle that selects one specific instance. This is the multi-apple scenario, manufactured at scale and for free. We co-train on a mix of (clean text only), (ambiguous text + circle), and (ambiguous text + circle + arrow) so the model keeps its text-only competence (Point-VLA showed co-training does not degrade and can even help text-only performance).

### 2.5 Two annotation representations (so we can ablate)

Support both, sharing the same auto-labels:

- **Rendered overlay:** burn the circle and arrow onto the RGB (or keep them on a separate prompt-image channel). Zero architecture change, the visual encoder simply sees the annotated image.
- **Symbolic tokens:** circle as `(c_x, c_y, r)`, arrow as `(x_0, y_0, x_1, y_1)`, embedded as a few special prompt tokens. Keeps the RGB clean and gives a disentangled signal that is easy to ablate and easy to inject into the latent-action model.

Store everything in an RLDS/HDF5 layout compatible with the UniVLA data loader: per frame, the base RGB, the overlay image and/or symbolic tokens, the (possibly degraded) instruction, and the existing action labels.

### 2.6 Negative and distractor mining

Explicitly construct hard cases: N near-identical objects where only the circled instance is correct, and M candidate destinations where only the arrow-pointed one is correct. Success on these is the headline metric, so they should be over-represented relative to their natural frequency.

## 3. Rollout testing with human effort

Training is human-free; evaluation is where the human draws.

### 3.1 Drawing interface and protocol

Build a lightweight canvas GUI (in the spirit of RT-Trajectory's drawing tool). Per episode: render the initial `agentview` frame, present it to the annotator, let them draw a circle and an arrow and optionally type a short or deliberately vague instruction, then feed the prompt to the policy and roll out in the simulator. Log success by the BDDL predicate (the standard LIBERO criterion of the goal predicate holding for a sustained window).

### 3.2 Conditions to compare

- (a) text only
- (c) text + circle + arrow,


### 3.3 Benchmarks

- **Referential disambiguation suite:** scenes with several identical or near-identical objects; the policy must manipulate the circled instance. Score the fraction of episodes where the *correct* instance was acted on.
- **Directional / destination suite:** several plausible targets; the policy must follow the arrow. Score correct-destination rate.
- Standard LIBERO suites for the no-regression check that text-only ability is preserved.

### 3.4 Human-effort and robustness measures

Time-to-draw and stroke count (to quantify that this is genuinely low-effort), inter-annotator agreement (do different people's circles and arrows yield similar success, i.e. is the channel reliable across users), and robustness to sloppy input (sweep circle center offset, radius error, arrow angle error, and report success degradation). Optionally test **interactive re-drawing**: let the human redraw mid-rollout when the robot drifts, a human-driven analogue of PIVOT's iterative refinement.

### 3.5 Scaling evaluation cheaply

Use the auto-annotator with heavy imprecision as a **synthetic human** to run large-scale evaluation, then validate with a smaller real-human study showing the synthetic proxy correlates with real success. This keeps the expensive human study small while still reporting large-N numbers.

## 4. Method: modifying UniVLA

UniVLA learns a task-centric latent action model in DINO feature space (language-conditioned, trained inverse-dynamics-style on how frames change), uses it to pseudo-label internet-scale video, trains an autoregressive policy to predict latent action tokens, then decodes those latent actions to embodiment-specific robot actions. We inject the circle and arrow at three points, in increasing order of integration and novelty.

### 4.1 Baseline A: input overlay

Burn the rendered circle and arrow into the observation that feeds both the latent action model and the policy. No architecture change. Cheap, and a fair baseline, but the overlay can corrupt DINO features, so control overlay opacity and contrast.

### 4.2 Core method B: visual-prompt tokens in the policy

Feed the symbolic circle `(c_x, c_y, r)` and arrow `(x_0, y_0, x_1, y_1)` as a small set of learned **visual-prompt tokens** appended to the language token stream of the policy VLM. The RGB stays clean; the policy cross-attends to these tokens alongside language. This disentangles the drawing signal from appearance and makes ablation clean.

### 4.3 Core method C: prompt-conditioned latent actions (the novel core)

This is where the project earns a paper rather than a demo, and where it connects to the latent-action thesis.

- **Arrow as a latent-action prior.** The image-plane arrow is a coarse latent action. Add an auxiliary loss that aligns the latent action model's predicted latent action (decoded back to an image-space direction) with the projected arrow direction. The arrow thus shapes the latent action space toward the human-intended motion rather than acting as a downstream hint.
- **Circle as task-centric patch grounding.** UniVLA already tries to keep latent actions *task-centric* by suppressing task-irrelevant dynamics. The circle is an explicit, human-given statement of which region is task-relevant. Use it as an attention bias over DINO patch tokens (upweight patches inside the circle) in the latent action model, or as a gate over object slots if you adopt LIBERO+/SlotVLA slot features. This makes the circle a direct controller of UniVLA's task-centricity rather than a generic mask.

The conceptual story is tight: **arrow conditions the latent action, circle conditions which patches the latent action attends to.** Both are exactly the quantities UniVLA already reasons about.

### 4.4 Analysis variant D: explicit decoupling

Route the circle solely to object identity (slot gating) and the arrow solely to a coarse waypoint that conditions the action decoder, then measure how much of the gain comes from "which" versus "where". Useful for the ablation narrative even if it is not the headline model.

### 4.5 Depth extension (optional, ties to depth-augmented latent actions)

An image-plane arrow is ambiguous about true 3D magnitude, the same RGB-magnitude ambiguity that motivates putting depth into latent actions. Lifting the arrow with the rendered depth map (LIBERO can render depth) recovers a 3D motion direction and magnitude, which should especially help height-sensitive tasks (the kind where RT-Trajectory's 2.5D variant beat its 2D variant). This is a clean bridge to existing depth-augmented latent-action work and a strong ablation: 2D arrow vs depth-lifted arrow.

### 4.6 Training recipe

Keep UniVLA's two-stage recipe. Add the prompt tokens (B), the arrow-alignment and circle-attention losses (C), and co-train on the mix of clean and ambiguous-plus-drawn pairs from Section 2.4 so text-only competence is preserved. Initialize from a released UniVLA LIBERO checkpoint to save compute.

## 5. Related work

**Sketch and trajectory goal specification**
- RT-Sketch: hand-drawn *goal-scene* sketches as the goal modality. https://arxiv.org/abs/2403.02709 (project: https://rt-sketch.github.io/). Closest in spirit, but encodes a full desired scene rather than two deictic primitives on the current observation.
- RT-Trajectory: coarse 2D/2.5D *trajectory sketches* as policy conditioning, specifiable by human drawing. https://arxiv.org/abs/2311.01977 (project: https://rt-trajectory.github.io/). Our arrow is a much lighter version of a trajectory; our circle adds referential grounding it does not have.

**Visual prompting and pointing for VLMs/VLAs**
- PIVOT: iterative visual prompting, annotating images with numbered arrows that a frozen VLM selects among. https://arxiv.org/abs/2402.07872 (project: https://pivot-prompt.github.io/). Generates and selects arrows at inference; we instead train on human-drawn circle+arrow and fold them into latent actions.
- RoboPoint: a VLM for spatial affordance (point) prediction. https://arxiv.org/abs/2406.10721. Related grounding signal, point rather than circle+arrow, and not a latent-action policy.
- CrayonRobo: object-centric, prompt-driven VLA that consumes overlaid prompts. https://arxiv.org/abs/2505.02166. Closest VLA-side neighbor; we differ by the circle+arrow primitive pair, the latent-action backbone, and the free LIBERO auto-labeling.
- Point-VLA: pixel-level visual grounding (e.g. boxes) added to VLA, plug-and-play, with co-training that preserves text-only ability. https://arxiv.org/abs/2512.16001 (verify arXiv id; surfaced Dec 2025). Supports our co-training-preserves-text-only design choice.
- VP-VLA: visual prompting as an interface for VLA, with a grounding loss on key frames. https://visualprompt-vla.github.io/.
- Interleave-VLA: interleaved image-text instructions for manipulation, with an automatic pipeline to build interleaved instructions from existing datasets. https://arxiv.org/abs/2505.02152. Same "build multimodal instructions from existing data" idea, different primitive.
- Visual Attentive Prompting / "Bring My Cup!": training-free visual prompting to personalize frozen VLAs to user-specific objects. https://arxiv.org/abs/2512.20014.
- GraphCoT-VLA: handling *ambiguous* instructions via spatial reasoning. https://www.researchgate.net/publication/394439488 (search "GraphCoT-VLA"). Same ambiguity motivation, language-side solution rather than a drawing channel.

**Latent-action VLA backbone**
- UniVLA: task-centric latent actions, the backbone we modify. https://arxiv.org/abs/2505.06111 (code: https://github.com/OpenDriveLab/UniVLA). Builds on the LAPA/LAPO latent-action lineage.

**Benchmark and object-centric annotations (the enabler)**
- LIBERO: the benchmark and demonstration source (Spatial / Object / Goal / Long suites, plus LIBERO-90). https://arxiv.org/abs/2306.03310 (project: https://libero-project.github.io/).
- LIBERO+ / SlotVLA: object-centric extension publishing pixel-level masks, bounding boxes, and instance-level temporal IDs for LIBERO, which we reuse for circle/arrow auto-labeling. https://arxiv.org/abs/2511.06754.

(A couple of arXiv ids above for very recent Dec 2025 papers should be double-checked against the arXiv listing before they go into the bibliography.)

## 6. Suggested contributions for the paper

1. A circle-and-arrow visual instruction channel for VLAs that decouples referential grounding (which) from coarse motion goals (where), each a single stroke.
2. A fully automatic pipeline that turns LIBERO into a sketch-prompted training set using simulator ground truth, including manufactured ambiguity (degraded captions plus duplicate-object scenes) so the drawing's value is measurable.
3. A UniVLA modification in which the arrow conditions the latent action and the circle conditions task-centric patch grounding, rather than treating the drawing as a generic overlay.
4. A human-in-the-loop evaluation protocol and a referential/directional disambiguation benchmark, with effort and robustness metrics and a synthetic-human proxy validated against a small real-human study.
5. (Optional) A depth-lifted arrow that recovers 3D motion magnitude, bridging to depth-augmented latent actions.

## 7. Open questions and risks

- Does the rendered overlay degrade DINO features enough to matter? The symbolic-token route (B) is the hedge.
- Does sim-style auto-drawing transfer to real human strokes? The imprecision augmentation and the inter-annotator-agreement study are the checks.
- Is the arrow's image-plane magnitude too ambiguous without depth? The depth ablation answers this directly.
- Co-training balance: too much ambiguous data may hurt clean text-only performance. Sweep the mixture ratio.
