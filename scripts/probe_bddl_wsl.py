import os
import h5py

DATA_DIR = "/mnt/c/Users/Admin/sketch_vla/data/libero_spatial"
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".hdf5"))
path = os.path.join(DATA_DIR, files[0])

import libero.libero.envs
import robosuite
import robomimic.utils.file_utils as FileUtils

env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=path)
env_kwargs = dict(env_meta["env_kwargs"])

env_kwargs["bddl_file_name"] = "/root/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
robo_env = robosuite.make(env_meta["env_name"], **env_kwargs)
robo_env.reset()

print("\n--- Exploring parsed problem dict if available ---")
if hasattr(robo_env, "parsed_problem"):
    print("robo_env.parsed_problem:", robo_env.parsed_problem)
elif hasattr(robo_env, "task") and hasattr(robo_env.task, "parsed_problem"):
    print("robo_env.task.parsed_problem:", robo_env.task.parsed_problem)
elif hasattr(robo_env, "problem_info"):
    print("robo_env.problem_info:", robo_env.problem_info)
else:
    print("Could not find parsed_problem directly.")
    print("robo_env dir:", dir(robo_env))
    if hasattr(robo_env, "task"):
        print("robo_env.task dir:", dir(robo_env.task))

print("\n--- Exploring Sites in MuJoCo Model ---")
site_names = []
for i in range(robo_env.sim.model.nsite):
    site_names.append(robo_env.sim.model.site_id2name(i))
print(f"Sites: {site_names}")

