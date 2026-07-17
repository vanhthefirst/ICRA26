"""
Phase-0 diagnostic for the DrawVLA validation-set augmentation (run in WSL2).

Purpose: gather everything needed to write a scene generator that AUGMENTS a
LIBERO-Spatial scene with extra bowls / plates / ramekins / cookie boxes and
emits a real BDDL task whose goal names a SPECIFIC instance.

It only reads/introspects. It writes nothing except an optional probe BDDL to
outputs/ (clearly named), so it is safe to run.

Run:
    conda activate <your libero env>
    cd /mnt/c/Users/Admin/sketch_vla
    python scripts/probe_augmentation_wsl.py 2>&1 | tee outputs/probe_augmentation.txt

Then send me outputs/probe_augmentation.txt (or paste the console output).
"""

import inspect, os, sys, traceback, glob

def hr(title):
    print("\n" + "=" * 78)
    print("== " + title)
    print("=" * 78)

def safe(fn, label):
    try:
        fn()
    except Exception as e:
        print(f"  [!] {label} failed: {e}")
        traceback.print_exc()

# ---------------------------------------------------------------------------
hr("0. Python / LIBERO / robosuite versions & locations")
def _versions():
    import libero, robosuite
    print("python      :", sys.version.split()[0])
    print("libero at   :", os.path.dirname(libero.__file__))
    print("robosuite   :", getattr(robosuite, "__version__", "?"))
    try:
        import mujoco; print("mujoco      :", mujoco.__version__)
    except Exception as e:
        print("mujoco      : (import failed)", e)
    # locate the libero package root (bddl_files, assets)
    import libero.libero as ll
    print("libero.libero at:", os.path.dirname(ll.__file__))
safe(_versions, "versions")

# ---------------------------------------------------------------------------
hr("1. Procedural-generation utilities available in THIS libero version")
def _proc_utils():
    import importlib
    for mod in [
        "libero.libero.utils.mu_utils",
        "libero.libero.utils.bddl_generation_utils",
        "libero.libero.utils.task_generation_utils",
    ]:
        try:
            m = importlib.import_module(mod)
            names = [n for n in dir(m) if not n.startswith("_")]
            print(f"\n{mod}:")
            print("   ", names)
        except Exception as e:
            print(f"\n{mod}: IMPORT FAILED -> {e}")
safe(_proc_utils, "proc utils listing")

# ---------------------------------------------------------------------------
hr("2. InitialSceneTemplates signature + a concrete subclass example")
def _ist():
    from libero.libero.utils import mu_utils
    IST = getattr(mu_utils, "InitialSceneTemplates", None)
    print("InitialSceneTemplates:", IST)
    if IST is not None:
        print("  __init__ signature:", inspect.signature(IST.__init__))
        # methods/properties we will need to implement
        for meth in ["define_regions", "init_states", "get_region_dict"]:
            print(f"  has {meth}:", hasattr(IST, meth))
    # registered scenes (mu) so we can pick a base tabletop scene to reuse
    for getter in ["get_scene_dict", "SCENE_DICT", "get_mu_dict"]:
        obj = getattr(mu_utils, getter, None)
        if obj is None:
            continue
        try:
            d = obj() if callable(obj) else obj
            print(f"\n  {getter} keys (scene templates):")
            print("   ", list(d.keys()))
        except Exception as e:
            print(f"  {getter} -> {e}")
safe(_ist, "InitialSceneTemplates")

# ---------------------------------------------------------------------------
hr("3. Object registry — confirm our 4 target categories + how instances are named")
def _objects():
    from libero.libero.utils import mu_utils
    # object dict maps category name -> object class
    for getter in ["get_object_dict", "OBJECTS_DICT", "get_all_objects"]:
        obj = getattr(mu_utils, getter, None)
        if obj is None:
            continue
        try:
            d = obj() if callable(obj) else obj
            keys = sorted(d.keys())
            print(f"\n  {getter}: {len(keys)} categories")
            want = ["bowl", "plate", "ramekin", "cookie"]
            hits = [k for k in keys if any(w in k.lower() for w in want)]
            print("  relevant categories:", hits)
            print("  (full list):", keys)
            return
        except Exception as e:
            print(f"  {getter} -> {e}")
    # fallback: introspect the objects module
    try:
        import libero.libero.envs.objects as O
        print("  objects module dir (filtered):",
              [n for n in dir(O) if any(w in n.lower() for w in ["bowl","plate","ramekin","cookie"])])
    except Exception as e:
        print("  objects module introspection failed:", e)
safe(_objects, "object registry")

# ---------------------------------------------------------------------------
hr("4. Template BDDL we will augment (between_the_plate_and_the_ramekin)")
def _bddl():
    import libero.libero as ll
    root = os.path.dirname(ll.__file__)
    spatial_dir = os.path.join(root, "bddl_files", "libero_spatial")
    print("spatial bddl dir:", spatial_dir, "exists:", os.path.isdir(spatial_dir))
    files = sorted(glob.glob(os.path.join(spatial_dir, "*.bddl")))
    print(f"  {len(files)} bddl files. names:")
    for f in files:
        print("   ", os.path.basename(f))
    # dump the ramekin one in full as our template
    tgt = [f for f in files if "between_the_plate_and_the_ramekin" in f]
    if tgt:
        print("\n----- FULL TEXT:", os.path.basename(tgt[0]), "-----")
        with open(tgt[0]) as fh:
            print(fh.read())
safe(_bddl, "template bddl")

# ---------------------------------------------------------------------------
hr("5. task_generation_utils: how tasks get registered + emitted")
def _taskgen():
    from libero.libero.utils import task_generation_utils as T
    for fn in ["register_task_info", "get_task_info", "generate_bddl_from_task_info"]:
        f = getattr(T, fn, None)
        if f is None:
            print(f"  {fn}: NOT PRESENT")
        else:
            try:
                print(f"  {fn}{inspect.signature(f)}")
            except (TypeError, ValueError):
                print(f"  {fn}: present (no signature)")
    from libero.libero.utils import bddl_generation_utils as B
    f = getattr(B, "generate_bddl_from_task_info", None)
    if f:
        try:
            print("  bddl_generation_utils.generate_bddl_from_task_info",
                  inspect.signature(f))
        except (TypeError, ValueError):
            print("  bddl_generation_utils.generate_bddl_from_task_info present")
safe(_taskgen, "task gen utils")

print("\n\nDONE. Please send outputs/probe_augmentation.txt back.")
