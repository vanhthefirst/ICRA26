import h5py
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\Admin\sketch_vla\data\libero_spatial"
OUTPUT_DIR = r"C:\Users\Admin\sketch_vla\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick the first HDF5 file found
hdf5_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".hdf5")]
if not hdf5_files:
    raise FileNotFoundError(f"No HDF5 files found in {DATA_DIR}")

filepath = os.path.join(DATA_DIR, hdf5_files[0])
print(f"Inspecting: {hdf5_files[0]}\n")


with h5py.File(filepath, "r") as f:
    # Top-level keys
    print("=== Top-level keys ===")
    print(list(f.keys()))

    # Data group keys
    print("\n=== Keys inside 'data' ===")
    print(list(f["data"].keys()))

    # Metadata attributes
    print("\n=== Metadata (data attrs) ===")
    for k, v in f["data"].attrs.items():
        print(f"  {k}: {v}")

    # Look inside the first demonstration
    demo = f["data/demo_0"]
    print("\n=== Keys inside demo_0 ===")
    print(list(demo.keys()))

    # Observation keys
    print("\n=== Keys inside demo_0/obs ===")
    print(list(demo["obs"].keys()))

    # Shapes of everything in obs
    print("\n=== Shapes of all obs fields ===")
    for k in demo["obs"].keys():
        arr = demo["obs"][k]
        print(f"  {k}: {arr.shape} | dtype: {arr.dtype}")

    # Actions
    actions = demo["actions"][:]
    print(f"\n=== Actions ===")
    print(f"  shape: {actions.shape} | dtype: {actions.dtype}")
    print(f"  first frame: {actions[0]}")

    # States
    if "states" in demo:
        states = demo["states"][:]
        print(f"\n=== States ===")
        print(f"  shape: {states.shape} | dtype: {states.dtype}")


    if "agentview_rgb" in demo["obs"]:
        rgb_key = "agentview_rgb"
    elif "agentview_image" in demo["obs"]:
        rgb_key = "agentview_image"
    else:
        rgb_key = None

    if rgb_key:
        frames = demo["obs"][rgb_key][:]   # shape (T, H, W, 3)
        print(f"\n=== RGB frames ===")
        print(f"  key used: {rgb_key}")
        print(f"  shape: {frames.shape}")

        # Save frame 0
        out_path = os.path.join(OUTPUT_DIR, "sample_frame.png")
        plt.imsave(out_path, frames[0])
        print(f"\n  Sample frame saved to: {out_path}")
    else:
        print("\nNo RGB key found — check obs keys above.")

print("\n=== Inspection complete ===")