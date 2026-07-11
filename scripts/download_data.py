from huggingface_hub import snapshot_download

print("Downloading LIBERO-Spatial (~5-8 GB). This may take a while...")

snapshot_download(
    repo_id="yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    local_dir=r"C:\Users\Admin\sketch_vla\data",
    allow_patterns="libero_spatial/*"
)

print("Done. Files saved to C:\\Users\\Admin\\sketch_vla\\data\\libero_spatial\\")