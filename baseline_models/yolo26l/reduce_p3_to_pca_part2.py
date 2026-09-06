
import time
from pathlib import Path

import torch


DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7_part2")
PRED_ROOT = DATASET_ROOT / "yolo26l"
SPLITS = ("train", "val", "test")

PCA_DIM = 48
SAMPLE_POSITIONS_PER_IMAGE = 25
SVD_NITER = 8
RNG_SEED = 42

BASIS_PATH = PRED_ROOT / f"pca_p3_basis_dim{PCA_DIM}.pt"


def collect_soft_files() -> list[tuple[str, Path, list[Path]]]:
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
    import gc

    rng = torch.Generator().manual_seed(RNG_SEED)
    total_files = sum(len(pts) for _, _, pts in collected)
    total_samples = total_files * sample_per_image

    print(f"\nPre-allocating sample buffer [{total_samples:,}, 512] fp32 "
          f"= {total_samples * 512 * 4 / 1e9:.2f} GB")
    X = torch.empty(total_samples, 512, dtype=torch.float32)

    print(f"\nCollecting samples for PCA fit ({sample_per_image} positions/file)...")
    n_seen = 0
    i_global = 0
    for split, soft_dir, pts in collected:
        for pt in pts:
            data = torch.load(pt, weights_only=False)
            p3_fp16 = data["p3_features"]
            C, H, W = p3_fp16.shape
            n_pos = H * W
            idx = torch.randperm(n_pos, generator=rng)[:sample_per_image]
            flat_fp16 = p3_fp16.permute(1, 2, 0).reshape(n_pos, C)
            X[i_global : i_global + sample_per_image] = flat_fp16[idx].float()
            i_global += sample_per_image
            n_seen += 1
            del data, p3_fp16, flat_fp16, idx
            if n_seen % 1000 == 0:
                print(f"  collected from {n_seen}/{total_files} files")
                gc.collect()

    print(f"\nSample matrix: {X.shape}  ({X.numel() * 4 / 1e9:.2f} GB)")
    mean = X.mean(dim=0)
    X.sub_(mean)
    return X, mean


def fit_pca(X_centered: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"\nFitting PCA (q={dim}, niter={SVD_NITER}) on CPU...")
    t0 = time.time()
    U, S, V = torch.svd_lowrank(X_centered, q=dim, niter=SVD_NITER)
    print(f"  SVD done in {time.time() - t0:.1f}s")

    total_var = (X_centered ** 2).sum() / (X_centered.shape[0] - 1)
    component_var = (S ** 2) / (X_centered.shape[0] - 1)
    evr = component_var / total_var

    print(f"  Top-{dim} explained variance: {evr.sum().item() * 100:.1f}%")
    print(f"  Top-10 component ratios: {[f'{v:.3f}' for v in evr[:10].tolist()]}")
    return V, evr


def apply_pca_to_file(pt_path: Path, V: torch.Tensor, mean: torch.Tensor) -> bool:
    data = torch.load(pt_path, weights_only=False)
    if "p3_features" not in data:
        return False

    p3 = data["p3_features"].float()
    C, H, W = p3.shape
    flat = p3.permute(1, 2, 0).reshape(H * W, C)
    reduced = (flat - mean) @ V
    pca = reduced.reshape(H, W, -1).permute(2, 0, 1).half()

    new_data = dict(data)
    new_data[f"p3_features_pca{PCA_DIM}"] = pca
    del new_data["p3_features"]
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
            "V": V,
            "mean": mean,
            "explained_variance_ratio": evr,
            "n_features_in": 512,
            "n_components": PCA_DIM,
        }, BASIS_PATH)
        print(f"\nSaved PCA basis to {BASIS_PATH}")
        del X_c

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
