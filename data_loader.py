import gc
import json
from typing import Dict, Generator, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
TRAIN_RATIO = 0.80
CODEBOOK_SIZE = 8


def video_disjoint_split(
    video_ids: list,
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
) -> Tuple[np.ndarray, np.ndarray]:
    unique_vids = np.array(sorted(set(video_ids)))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_vids)
    n_train_vids = int(len(shuffled) * train_ratio)
    train_vids = set(shuffled[:n_train_vids])
    train_mask = np.array([v in train_vids for v in video_ids], dtype=bool)
    return train_mask, ~train_mask


def one_hot_indices(indices: torch.Tensor, codebook_size: int = CODEBOOK_SIZE) -> np.ndarray:
    idx = indices.long()
    min_val = idx.min().item()
    if min_val > 0:
        idx = idx - min_val
    return F.one_hot(idx, codebook_size).float().reshape(len(idx), -1).numpy()


def load_split_info(manifest_path: str) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lightweight first pass: reads only video_id and magnitude from every shard
    to compute the train/test split without holding all feature tensors in RAM.
    Returns (manifest, train_mask, test_mask, y).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_video_ids: list = []
    y_parts: list = []

    for part_meta in manifest["parts"]:
        shard = torch.load(part_meta["path"], map_location="cpu", weights_only=False)
        all_video_ids.extend(shard["video_id"])
        y_parts.append(shard["magnitude"].clone())
        del shard
        gc.collect()

    train_mask, test_mask = video_disjoint_split(all_video_ids)
    y = torch.cat(y_parts).numpy()
    return manifest, train_mask, test_mask, y


def assemble_feature(
    manifest: dict,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    dim: int,
    extract_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shard-by-shard feature assembly.
    extract_fn(shard_dict, shard_train_bool, shard_test_bool) -> (X_tr, X_te)

    Peak RAM = one loaded shard + pre-allocated X_train + pre-allocated X_test.
    The full (N, dim) matrix is never materialised.
    """
    n_train = int(train_mask.sum())
    n_test  = int(test_mask.sum())
    X_train = np.empty((n_train, dim), dtype=np.float32)
    X_test  = np.empty((n_test,  dim), dtype=np.float32)
    tr_ptr = te_ptr = offset = 0

    for part_meta in manifest["parts"]:
        n = part_meta["num_samples"]
        shard_tr = train_mask[offset:offset + n]
        shard_te = test_mask [offset:offset + n]

        shard = torch.load(part_meta["path"], map_location="cpu", weights_only=False)
        X_tr, X_te = extract_fn(shard, shard_tr, shard_te)
        del shard
        gc.collect()

        n_tr, n_te = int(shard_tr.sum()), int(shard_te.sum())
        X_train[tr_ptr:tr_ptr + n_tr] = X_tr
        X_test [te_ptr:te_ptr + n_te] = X_te
        tr_ptr += n_tr
        te_ptr += n_te
        offset += n
        del X_tr, X_te

    return X_train, X_test


# kept for scripts that still use bulk loading (run_umap.py uses assemble_feature directly)
def load_all_parts(manifest_path: str) -> Dict:
    with open(manifest_path) as f:
        manifest = json.load(f)

    ids, video_ids = [], []
    tensors: Dict[str, list] = {}

    for part_meta in manifest["parts"]:
        data = torch.load(part_meta["path"], map_location="cpu", weights_only=False)
        ids.extend(data["id"])
        video_ids.extend(data["video_id"])
        for key, val in data.items():
            if torch.is_tensor(val):
                tensors.setdefault(key, []).append(val)

    return {
        "id": ids,
        "video_id": video_ids,
        **{k: torch.cat(v, dim=0) for k, v in tensors.items()},
    }