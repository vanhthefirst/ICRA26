"""Register the exported sketch scenes as LIBERO benchmark suites.

Import this once before `benchmark.get_benchmark_dict()`. It reads
task_map.json from the directory this file sits in, so the two stay in step.

Point LIBERO at the exported trees first, either by editing ~/.libero/config.yaml
or by copying bddl_files/<pf> and init_files/<pf> into the paths
`get_libero_path("bddl_files")` and `get_libero_path("init_states")` return.

The BDDL alone does not reproduce a scene. `set_init_state` restores qpos/qvel
only, and some Goal arenas draw material state during reset() from the global
numpy stream, so seed with the task's `reset_seed` BEFORE calling reset():

    np.random.seed(TASK_SEEDS["sketch_goal"]["scene_0000"])
    env.reset()
    env.set_init_state(initial_states[episode_idx])

Skipping that reproduces correct physics and the wrong render (measured at
158/255 max channel diff on goal/scene_0000).

Each task carries one caption, as LIBERO's schema allows. The ambiguous caption
for the same scene is in task_map.json under `language_ambiguous`.

The three imports below are LIBERO's registration API and have moved between
releases. Confirm them against the installed copy before trusting this file:

    python -c "from libero.libero import benchmark; print(benchmark.__file__)"
"""

import json
import os

from libero.libero.benchmark import Benchmark
from libero.libero.benchmark import register_benchmark
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "task_map.json")) as _f:
    SKETCH_TASK_MAP = json.load(_f)

TASK_SEEDS = {
    pf: {t["name"]: t["reset_seed"] for t in tasks}
    for pf, tasks in SKETCH_TASK_MAP.items()
}


def _install():
    for pf, tasks in SKETCH_TASK_MAP.items():
        libero_task_map[pf] = [t["name"] for t in tasks]

        def _make(problem_folder):
            @register_benchmark
            class _SketchSuite(Benchmark):
                name = problem_folder

                def __init__(self, task_order_index=0):
                    super().__init__(task_order_index=task_order_index)

            _SketchSuite.__name__ = problem_folder.upper()
            return _SketchSuite

        _make(pf)


_install()
