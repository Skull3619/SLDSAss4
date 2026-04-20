from __future__ import annotations

import argparse
import math
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial import ConvexHull, QhullError, cKDTree

LABEL_TO_TARGET = {"infeasible": 0, "feasible": 1}


@dataclass
class FeatureConfig:
    occupancy_scales: tuple[int, ...] = (8, 16, 24)
    radial_bins: int = 8
    d2_bins: int = 8
    projection_bins: int = 8
    nn_sample_size: int = 6000
    hull_sample_size: int = 4000
    local_sample_size: int = 1500
    local_k: int = 16
    d2_pair_sample: int = 5000
    random_seed: int = 42
    remove_duplicates: bool = True
    clip_outliers: bool = False
    outlier_quantile: float = 0.995


def _safe_ratio(num: float, den: float) -> float:
    if den == 0 or math.isclose(den, 0.0):
        return 0.0
    return float(num / den)


def read_ply_xyz(path: str | Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"{path} has no vertex element.")
    vertex = ply["vertex"].data
    names = vertex.dtype.names
    if not all(c in names for c in ("x", "y", "z")):
        raise ValueError(f"{path} is missing x/y/z coordinates.")
    pts = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)
    if len(pts) == 0:
        raise ValueError(f"{path} has zero vertices.")
    return pts


def clean_points(points: np.ndarray, remove_duplicates: bool = True, clip_outliers: bool = False, outlier_quantile: float = 0.995) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if remove_duplicates:
        pts = np.unique(pts, axis=0)
    if clip_outliers and len(pts) >= 10:
        centroid = pts.mean(axis=0)
        d = np.linalg.norm(pts - centroid, axis=1)
        cutoff = np.quantile(d, outlier_quantile)
        pts = pts[d <= cutoff]
    return pts


def _sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def _hist_features(values: np.ndarray, bins: int, prefix: str) -> dict[str, float]:
    if len(values) == 0:
        return {f"{prefix}_{i}": 0.0 for i in range(bins)}
    hist, _ = np.histogram(values, bins=bins, density=False)
    hist = hist.astype(float) / max(hist.sum(), 1.0)
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(hist)}


def _occupancy_features(points: np.ndarray, bins: int) -> dict[str, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-12)
    normalized = (points - mins) / spans
    idx = np.floor(normalized * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    flat = idx[:, 0] * bins * bins + idx[:, 1] * bins + idx[:, 2]
    unique, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        f"occ_{bins}_occupied_bins": float(len(unique)),
        f"occ_{bins}_ratio": float(len(unique) / (bins ** 3)),
        f"occ_{bins}_entropy": entropy,
    }


def _nearest_neighbor_features(points: np.ndarray, sample_size: int, rng: np.random.Generator) -> dict[str, float]:
    pts = _sample_points(points, sample_size, rng)
    if len(pts) < 2:
        return {k: 0.0 for k in ["nn_mean", "nn_std", "nn_q10", "nn_q50", "nn_q90"]}
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn = dists[:, 1]
    return {
        "nn_mean": float(nn.mean()),
        "nn_std": float(nn.std(ddof=0)),
        "nn_q10": float(np.quantile(nn, 0.10)),
        "nn_q50": float(np.quantile(nn, 0.50)),
        "nn_q90": float(np.quantile(nn, 0.90)),
    }


def _hull_features(points: np.ndarray, sample_size: int, rng: np.random.Generator) -> dict[str, float]:
    pts = _sample_points(points, sample_size, rng)
    if len(pts) < 4:
        return {"hull_area": 0.0, "hull_volume": 0.0, "compactness": 0.0}
    try:
        hull = ConvexHull(pts)
        area = float(hull.area)
        volume = float(hull.volume)
        compactness = _safe_ratio(volume ** (2.0 / 3.0), area)
        return {"hull_area": area, "hull_volume": volume, "compactness": compactness}
    except QhullError:
        return {"hull_area": 0.0, "hull_volume": 0.0, "compactness": 0.0}


def _d2_features(points: np.ndarray, bins: int, pair_sample: int, rng: np.random.Generator) -> dict[str, float]:
    n = len(points)
    if n < 2:
        return {f"d2_hist_{i}": 0.0 for i in range(bins)}
    idx1 = rng.integers(0, n, size=pair_sample)
    idx2 = rng.integers(0, n, size=pair_sample)
    same = idx1 == idx2
    if same.any():
        idx2[same] = (idx2[same] + 1) % n
    d = np.linalg.norm(points[idx1] - points[idx2], axis=1)
    d = d / max(d.max(), 1e-12)
    return _hist_features(d, bins, "d2_hist")


def _local_geometry_features(points: np.ndarray, sample_size: int, k: int, rng: np.random.Generator) -> dict[str, float]:
    pts = _sample_points(points, sample_size, rng)
    if len(pts) <= k + 1:
        keys = [
            "local_curvature_mean", "local_curvature_std", "local_curvature_q90",
            "local_linearity_mean", "local_linearity_std",
            "local_planarity_mean", "local_planarity_std",
            "local_sphericity_mean", "local_sphericity_std",
            "normal_abs_x_mean", "normal_abs_y_mean", "normal_abs_z_mean",
        ]
        return {k: 0.0 for k in keys}

    tree = cKDTree(points)
    _, idxs = tree.query(pts, k=k + 1)
    local_curv, local_lin, local_plan, local_sph = [], [], [], []
    normal_abs = []

    for nbr_idx in idxs:
        nbrs = points[nbr_idx[1:]]
        centroid = nbrs.mean(axis=0)
        centered = nbrs - centroid
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.sort(np.maximum(eigvals, 1e-12))[::-1]
        e1, e2, e3 = eigvals
        eigsum = e1 + e2 + e3
        local_lin.append(_safe_ratio(e1 - e2, e1))
        local_plan.append(_safe_ratio(e2 - e3, e1))
        local_sph.append(_safe_ratio(e3, e1))
        local_curv.append(_safe_ratio(e3, eigsum))
        normal = eigvecs[:, np.argmin(eigvals)]
        normal_abs.append(np.abs(normal))

    normal_abs = np.asarray(normal_abs)
    return {
        "local_curvature_mean": float(np.mean(local_curv)),
        "local_curvature_std": float(np.std(local_curv)),
        "local_curvature_q90": float(np.quantile(local_curv, 0.90)),
        "local_linearity_mean": float(np.mean(local_lin)),
        "local_linearity_std": float(np.std(local_lin)),
        "local_planarity_mean": float(np.mean(local_plan)),
        "local_planarity_std": float(np.std(local_plan)),
        "local_sphericity_mean": float(np.mean(local_sph)),
        "local_sphericity_std": float(np.std(local_sph)),
        "normal_abs_x_mean": float(np.mean(normal_abs[:, 0])),
        "normal_abs_y_mean": float(np.mean(normal_abs[:, 1])),
        "normal_abs_z_mean": float(np.mean(normal_abs[:, 2])),
    }


def extract_features(points: np.ndarray, file_name: str, file_path: str, label: str, cfg: FeatureConfig) -> dict[str, float | str | int]:
    rng = np.random.default_rng(cfg.random_seed)
    n = len(points)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins
    bbox_volume = float(np.prod(extents))
    bbox_diag = float(np.linalg.norm(extents))
    centroid = points.mean(axis=0)
    centered = points - centroid
    stds = points.std(axis=0, ddof=0)

    q10 = np.quantile(points, 0.10, axis=0)
    q50 = np.quantile(points, 0.50, axis=0)
    q90 = np.quantile(points, 0.90, axis=0)

    radial = np.linalg.norm(centered, axis=1)
    radial_norm = radial / max(radial.max(), 1e-12)

    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    e1, e2, e3 = [float(v) for v in eigvals]
    eigsum = e1 + e2 + e3
    p1, p2, p3 = [(v / eigsum) if eigsum > 0 else 0.0 for v in (e1, e2, e3)]

    hull = _hull_features(points, cfg.hull_sample_size, rng)

    row: dict[str, float | str | int] = {
        "file_name": file_name,
        "file_path": file_path,
        "label": label,
        "target": LABEL_TO_TARGET.get(label, -1),
        "num_points": int(n),
        "min_x": float(mins[0]), "min_y": float(mins[1]), "min_z": float(mins[2]),
        "max_x": float(maxs[0]), "max_y": float(maxs[1]), "max_z": float(maxs[2]),
        "extent_x": float(extents[0]), "extent_y": float(extents[1]), "extent_z": float(extents[2]),
        "bbox_volume": bbox_volume,
        "bbox_diag": bbox_diag,
        "aspect_xy": _safe_ratio(extents[0], extents[1]),
        "aspect_xz": _safe_ratio(extents[0], extents[2]),
        "aspect_yz": _safe_ratio(extents[1], extents[2]),
        "density_bbox": _safe_ratio(n, bbox_volume),
        "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]), "centroid_z": float(centroid[2]),
        "std_x": float(stds[0]), "std_y": float(stds[1]), "std_z": float(stds[2]),
        "x_q10": float(q10[0]), "y_q10": float(q10[1]), "z_q10": float(q10[2]),
        "x_q50": float(q50[0]), "y_q50": float(q50[1]), "z_q50": float(q50[2]),
        "x_q90": float(q90[0]), "y_q90": float(q90[1]), "z_q90": float(q90[2]),
        "radial_mean": float(radial.mean()),
        "radial_std": float(radial.std(ddof=0)),
        "radial_max": float(radial.max()),
        "eig1": e1, "eig2": e2, "eig3": e3,
        "linearity": _safe_ratio(e1 - e2, e1),
        "planarity": _safe_ratio(e2 - e3, e1),
        "sphericity": _safe_ratio(e3, e1),
        "anisotropy": _safe_ratio(e1 - e3, e1),
        "curvature": _safe_ratio(e3, eigsum),
        "eigentropy": float(-sum(p * math.log(p + 1e-12) for p in (p1, p2, p3))),
        "hull_area": float(hull["hull_area"]),
        "hull_volume": float(hull["hull_volume"]),
        "hull_fill_ratio": _safe_ratio(float(hull["hull_volume"]), bbox_volume),
        "density_hull": _safe_ratio(n, float(hull["hull_volume"])),
        "compactness": float(hull["compactness"]),
    }

    row.update(_nearest_neighbor_features(points, cfg.nn_sample_size, rng))
    row.update(_local_geometry_features(points, cfg.local_sample_size, cfg.local_k, rng))
    for bins in cfg.occupancy_scales:
        row.update(_occupancy_features(points, bins))
    row.update(_hist_features(radial_norm, cfg.radial_bins, "radial_hist"))
    row.update(_d2_features(points, cfg.d2_bins, cfg.d2_pair_sample, rng))
    for axis_idx, axis_name in enumerate(["x", "y", "z"]):
        vals = points[:, axis_idx]
        vals = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-12)
        row.update(_hist_features(vals, cfg.projection_bins, f"proj_{axis_name}"))
    return row


def _process_one(task):
    path_str, label, cfg = task
    path = Path(path_str)
    pts = read_ply_xyz(path)
    pts = clean_points(pts, cfg.remove_duplicates, cfg.clip_outliers, cfg.outlier_quantile)
    return extract_features(pts, path.name, str(path), label, cfg)


def collect_tasks(root_dir: Path):
    tasks = []
    for label in ["feasible", "infeasible"]:
        folder = root_dir / label
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.ply")):
            tasks.append((str(path), label))
    if not tasks:
        raise FileNotFoundError(f"No .ply files found under {root_dir}/feasible and {root_dir}/infeasible")
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Offline 3D point-cloud feature extraction for manufacturing feasibility.")
    parser.add_argument("--root_dir", required=True, help="Dataset root containing feasible/ and infeasible/ folders.")
    parser.add_argument("--output", required=True, help="Output path: .csv, .xlsx, or .parquet")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--clip_outliers", action="store_true")
    parser.add_argument("--outlier_quantile", type=float, default=0.995)
    parser.add_argument("--keep_duplicates", action="store_true")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    cfg = FeatureConfig(
        remove_duplicates=not args.keep_duplicates,
        clip_outliers=args.clip_outliers,
        outlier_quantile=args.outlier_quantile,
    )
    tasks = collect_tasks(root_dir)
    rows = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_process_one, (path, label, cfg)): (path, label) for path, label in tasks}
            for fut in as_completed(futs):
                path, _ = futs[fut]
                try:
                    rows.append(fut.result())
                except Exception as e:
                    warnings.warn(f"Skipping {path}: {e}")
    else:
        for path, label in tasks:
            try:
                rows.append(_process_one((path, label, cfg)))
            except Exception as e:
                warnings.warn(f"Skipping {path}: {e}")

    df = pd.DataFrame(rows).sort_values(["label", "file_name"]).reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    elif out.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(out, index=False)
    elif out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    else:
        raise ValueError("Output must end with .csv, .xlsx, or .parquet")
    print(f"Saved {len(df)} rows and {df.shape[1]} columns to {out}")


if __name__ == "__main__":
    main()
