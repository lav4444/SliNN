"""
One-time PCA reduction of teacher P3 features (512-dim → 48-dim).

Why: the full 512-channel teacher P3 features take ~55 GB across train/val/test
splits, and projecting tiny students (48-96 channels) up to 512-d via 1x1 conv
is rank-bounded and unstable. PCA reduction:
    1. Collapses storage from ~55 GB to ~5 GB
    2. Gives students a matched-dimension target (no rank cage in FGD loss)
    3. Captures dominant teacher feature directions (top-K eigenvalues)

What this script does:
    1. Scans the teacher soft cache (yolo26l/<split>/soft/*.pt) for "p3_features"
    2. Samples spatial positions across all files to build a fitting matrix
    3. Fits a 48-component PCA via torch.svd_lowrank
    4. Projects each .pt file's p3_features through the basis → 48-d reduced
    5. Overwrites the .pt file: REMOVES "p3_features" (512-d) and ADDS
       "p3_features_pca48" (48-d, fp16)
    6. Saves the PCA basis to yolo26l/pca_p3_basis_dim48.pt for diagnostic/reuse

Idempotent: files already reduced (p3_features missing, pca48 present) are skipped.

Run once after baseline_models/yolo26l/evaluate.py has populated p3_features:
    conda activate dipl
    python reduce_p3_to_pca.py
"""

import time
from pathlib import Path

import torch


# ============================ CONFIG ============================
DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7_part2")
PRED_ROOT = DATASET_ROOT / "yolo26l"
SPLITS = ("train", "val", "test")

PCA_DIM = 48                        # target reduced dimension
SAMPLE_POSITIONS_PER_IMAGE = 25     # 25 × 8371 ≈ 209k samples for PCA fitting (memory-conscious)
SVD_NITER = 8                       # power iterations in torch.svd_lowrank (more = more accurate)
RNG_SEED = 42

BASIS_PATH = PRED_ROOT / f"pca_p3_basis_dim{PCA_DIM}.pt"
# ================================================================


def collect_soft_files() -> list[tuple[str, Path, list[Path]]]:
    """Return [(split, soft_dir, [pt paths to process])] across splits."""
    out = []
    for split in SPLITS:
        soft_dir = PRED_ROOT / split / "soft"
        if not soft_dir.exists():
            print(f"[{split}] soft_dir not present, skipping")
            continue
        pts = sorted(soft_dir.glob("*.pt"))
        if not pts:
            print(f"[{split}] no .pt files, skipping")
            continue
        # Spot-check first file to learn what state the cache is in
        sample = torch.load(pts[0], weights_only=False)
        has_full = "p3_features" in sample
        has_pca = f"p3_features_pca{PCA_DIM}" in sample
        if has_pca and not has_full:
            print(f"[{split}] already reduced (pca{PCA_DIM}); skipping {len(pts)} files")
            continue
        if not has_full:
            print(f"[{split}] no p3_features and no pca; teacher cache not populated?")
            continue
        out.append((split, soft_dir, pts))
        print(f"[{split}] {len(pts)} files to reduce")
    return out


def collect_sample_matrix(collected, sample_per_image: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample spatial positions across all files, return (X_centered, mean).

    Uses a PRE-ALLOCATED buffer (not list+cat) to avoid peak memory doubling.
    Samples stored as fp32 directly into a single [N, 512] tensor.
    Returns:
        X_centered: [N_samples, 512] fp32 on CPU, mean-subtracted (IN-PLACE)
        mean: [512] fp32 on CPU
    """
    import gc

    rng = torch.Generator().manual_seed(RNG_SEED)
    total_files = sum(len(pts) for _, _, pts in collected)
    total_samples = total_files * sample_per_image

    # Pre-allocate (avoids doubling memory during torch.cat)
    print(f"\nPre-allocating sample buffer [{total_samples:,}, 512] fp32 "
          f"= {total_samples * 512 * 4 / 1e9:.2f} GB")
    X = torch.empty(total_samples, 512, dtype=torch.float32)

    print(f"\nCollecting samples for PCA fit ({sample_per_image} positions/file)...")
    n_seen = 0
    i_global = 0
    for split, soft_dir, pts in collected:
        for pt in pts:
            data = torch.load(pt, weights_only=False)
            # Process in a small scope so refs are freed promptly
            p3_fp16 = data["p3_features"]                        # [512, 80, 80] fp16
            C, H, W = p3_fp16.shape
            n_pos = H * W
            idx = torch.randperm(n_pos, generator=rng)[:sample_per_image]
            # Slice fp16 → upcast small subset (50×512=25k floats) instead of upcasting all 3.3M
            flat_fp16 = p3_fp16.permute(1, 2, 0).reshape(n_pos, C)
            X[i_global : i_global + sample_per_image] = flat_fp16[idx].float()
            i_global += sample_per_image
            n_seen += 1
            # Explicitly drop references; ensure GC frees the .pt dict promptly
            del data, p3_fp16, flat_fp16, idx
            if n_seen % 1000 == 0:
                print(f"  collected from {n_seen}/{total_files} files")
                gc.collect()    # periodic explicit collection

    print(f"\nSample matrix: {X.shape}  ({X.numel() * 4 / 1e9:.2f} GB)")
    mean = X.mean(dim=0)                                          # [512]
    X.sub_(mean)                                                  # mean-center IN-PLACE
    return X, mean


def fit_pca(X_centered: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit PCA via torch.svd_lowrank. Returns (V, explained_var_ratio).

    V: [D, dim] columns are the top-dim principal directions
    explained_var_ratio: [dim] fraction of total variance per component
    """
    print(f"\nFitting PCA (q={dim}, niter={SVD_NITER}) on CPU...")
    t0 = time.time()
    # CPU only — SVD of [209k, 512] matrix is fast enough (~5s), and avoids any
    # risk of GPU OOM (yolo26L weights may still be cached on GPU from evaluate.py).
    U, S, V = torch.svd_lowrank(X_centered, q=dim, niter=SVD_NITER)   # V: [D, dim]
    print(f"  SVD done in {time.time() - t0:.1f}s")

    # Explained variance ratio
    total_var = (X_centered ** 2).sum() / (X_centered.shape[0] - 1)
    component_var = (S ** 2) / (X_centered.shape[0] - 1)
    evr = component_var / total_var

    print(f"  Top-{dim} explained variance: {evr.sum().item() * 100:.1f}%")
    print(f"  Top-10 component ratios: {[f'{v:.3f}' for v in evr[:10].tolist()]}")
    return V, evr


def apply_pca_to_file(pt_path: Path, V: torch.Tensor, mean: torch.Tensor) -> bool:
    """Project p3_features in this .pt through V → 48-d, overwrite .pt with pca key only.
    Returns True if reduction was performed."""
    data = torch.load(pt_path, weights_only=False)
    if "p3_features" not in data:
        return False  # already reduced or never had it

    p3 = data["p3_features"].float()                             # [512, 80, 80]
    C, H, W = p3.shape
    flat = p3.permute(1, 2, 0).reshape(H * W, C)                 # [6400, 512]
    reduced = (flat - mean) @ V                                   # [6400, dim]
    pca = reduced.reshape(H, W, -1).permute(2, 0, 1).half()      # [dim, 80, 80] fp16

    new_data = dict(data)
    new_data[f"p3_features_pca{PCA_DIM}"] = pca
    del new_data["p3_features"]                                   # drop 512-d
    torch.save(new_data, pt_path)
    return True


def main():
    t_main = time.time()
    print(f"Reducing teacher P3 features 512 → {PCA_DIM}-dim via PCA")
    print(f"Source: {PRED_ROOT}")

    collected = collect_soft_files()
    if not collected:
        print("\nNothing to do.")
        return

    # ----- Fit (or load) PCA basis -----
    if BASIS_PATH.exists():
        print(f"\nLoading existing PCA basis from {BASIS_PATH}")
        basis = torch.load(BASIS_PATH, weights_only=False)
        V = basis["V"]
        mean = basis["mean"]
        evr = basis.get("explained_variance_ratio")
        if evr is not None:
            print(f"  loaded {V.shape[1]}-component basis, explained var: {evr.sum().item() * 100:.1f}%")
    else:
        X_c, mean = collect_sample_matrix(collected, SAMPLE_POSITIONS_PER_IMAGE)
        V, evr = fit_pca(X_c, PCA_DIM)
        torch.save({
            "V": V,                          # [512, dim]
            "mean": mean,                    # [512]
            "explained_variance_ratio": evr, # [dim]
            "n_features_in": 512,
            "n_components": PCA_DIM,
        }, BASIS_PATH)
        print(f"\nSaved PCA basis to {BASIS_PATH}")
        del X_c   # free RAM

    # ----- Apply to all files -----
    print(f"\nApplying PCA reduction to all .pt files...")
    for split, soft_dir, pts in collected:
        print(f"[{split}] reducing {len(pts)} files...")
        t_split = time.time()
        n_done = 0
        for i, pt in enumerate(pts):
            if apply_pca_to_file(pt, V, mean):
                n_done += 1
            if (i + 1) % 500 == 0:
                print(f"  [{split}] {i + 1}/{len(pts)}  ({(i + 1) / (time.time() - t_split):.0f} files/s)")
        print(f"[{split}] done ({n_done}/{len(pts)} reduced) in {time.time() - t_split:.1f}s")

    print(f"\nTotal time: {time.time() - t_main:.1f}s")
    print(f"Storage: each .pt went from ~6.6 MB → ~640 KB (10× smaller)")
    print(f"\nVerify:")
    print(f"  ls -lh {PRED_ROOT}/train/soft | head -3")
    print(f"  python -c \"import torch; "
          f"d=torch.load('{collected[0][2][0]}', weights_only=False); "
          f"print(list(d.keys()))\"")


if __name__ == "__main__":
    main()
