#!/usr/bin/env python3
"""Compare two small/medium ASCII PCD maps.

The tool reports symmetric nearest-neighbor distances, local geometric
thickness, and Euclidean cluster summaries. It also computes an optional
rigid ICP alignment of the test cloud to the reference cloud, but prints raw
metrics first so coordinate-frame problems remain visible.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load_ascii_pcd(path):
    fields = None
    points = None
    data_line = None
    lines = Path(path).read_text(encoding="ascii", errors="strict").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        key = parts[0].upper()
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "POINTS":
            points = int(parts[1])
        elif key == "DATA":
            if parts[1].lower() != "ascii":
                raise ValueError(f"{path}: only ASCII PCD is supported")
            data_line = i + 1
            break
    if fields is None or data_line is None:
        raise ValueError(f"{path}: invalid or unsupported PCD header")
    rows = []
    for line in lines[data_line:]:
        stripped = line.strip()
        if stripped:
            rows.append([float(v) for v in stripped.split()])
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path}: expected at least x y z fields")
    if points is not None and points != data.shape[0]:
        raise ValueError(f"{path}: POINTS={points}, but read {data.shape[0]} rows")
    field_to_index = {name: i for i, name in enumerate(fields)}
    xyz = data[:, [field_to_index["x"], field_to_index["y"], field_to_index["z"]]]
    intensity = None
    for name in ("intensity", "reflectivity"):
        if name in field_to_index:
            intensity = data[:, field_to_index[name]]
            break
    return xyz, intensity, fields


def fmt_mm(value_m):
    return f"{value_m * 1000.0:.1f}mm"


def distance_stats(distances):
    if len(distances) == 0:
        return {}
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p90": float(np.percentile(distances, 90)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "max": float(np.max(distances)),
    }


def print_distance_block(title, distances):
    stats = distance_stats(distances)
    print(title)
    print(
        "  mean={} median={} p90={} p95={} p99={} max={}".format(
            fmt_mm(stats["mean"]),
            fmt_mm(stats["median"]),
            fmt_mm(stats["p90"]),
            fmt_mm(stats["p95"]),
            fmt_mm(stats["p99"]),
            fmt_mm(stats["max"]),
        )
    )
    for threshold in (0.01, 0.02, 0.05, 0.10):
        coverage = np.mean(distances <= threshold) * 100.0
        print(f"  <= {fmt_mm(threshold)}: {coverage:.1f}%")


def nearest_distances(source, target):
    tree = cKDTree(target)
    distances, indices = tree.query(source, k=1, workers=-1)
    return distances, indices


def write_ascii_pcd(path, xyz, intensity=None):
    with Path(path).open("w", encoding="ascii") as output:
        output.write("# .PCD v0.7 - Point Cloud Data file format\n")
        output.write("VERSION 0.7\n")
        if intensity is None:
            output.write("FIELDS x y z\n")
            output.write("SIZE 4 4 4\n")
            output.write("TYPE F F F\n")
            output.write("COUNT 1 1 1\n")
        else:
            output.write("FIELDS x y z intensity\n")
            output.write("SIZE 4 4 4 4\n")
            output.write("TYPE F F F F\n")
            output.write("COUNT 1 1 1 1\n")
        output.write(f"WIDTH {len(xyz)}\n")
        output.write("HEIGHT 1\n")
        output.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        output.write(f"POINTS {len(xyz)}\n")
        output.write("DATA ascii\n")
        if intensity is None:
            for point in xyz:
                output.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
        else:
            for point, value in zip(xyz, intensity):
                output.write(
                    f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {value:.9g}\n"
                )


def bounds_summary(name, xyz, intensity):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    extent = maxs - mins
    centroid = xyz.mean(axis=0)
    print(f"{name}:")
    print(f"  points={len(xyz)}")
    print(f"  centroid=[{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    print(f"  min=[{mins[0]:.3f}, {mins[1]:.3f}, {mins[2]:.3f}]")
    print(f"  max=[{maxs[0]:.3f}, {maxs[1]:.3f}, {maxs[2]:.3f}]")
    print(f"  extent=[{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")
    if intensity is not None and len(intensity):
        print(
            "  intensity min={:.1f} median={:.1f} mean={:.1f} max={:.1f}".format(
                float(np.min(intensity)),
                float(np.median(intensity)),
                float(np.mean(intensity)),
                float(np.max(intensity)),
            )
        )


def local_thickness(xyz, radius, min_neighbors):
    tree = cKDTree(xyz)
    values = []
    for i, point in enumerate(xyz):
        ids = tree.query_ball_point(point, radius)
        if len(ids) < min_neighbors:
            continue
        neighborhood = xyz[ids]
        centered = neighborhood - neighborhood.mean(axis=0)
        cov = centered.T @ centered / max(1, len(ids) - 1)
        eigvals = np.linalg.eigvalsh(cov)
        values.append(math.sqrt(max(0.0, float(eigvals[0]))))
    return np.asarray(values, dtype=np.float64)


def print_thickness(name, xyz, radius, min_neighbors):
    thickness = local_thickness(xyz, radius, min_neighbors)
    print(f"{name} local thickness radius={fmt_mm(radius)} min_neighbors={min_neighbors}:")
    if len(thickness) == 0:
        print("  not enough neighborhoods")
        return
    stats = distance_stats(thickness)
    print(
        "  neighborhoods={} median={} p90={} p95={} p99={} max={}".format(
            len(thickness),
            fmt_mm(stats["median"]),
            fmt_mm(stats["p90"]),
            fmt_mm(stats["p95"]),
            fmt_mm(stats["p99"]),
            fmt_mm(stats["max"]),
        )
    )


def cluster_summary(xyz, radius, min_points):
    tree = cKDTree(xyz)
    visited = np.zeros(len(xyz), dtype=bool)
    sizes = []
    diagonals = []
    for start in range(len(xyz)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        cluster = []
        while stack:
            idx = stack.pop()
            cluster.append(idx)
            for nb in tree.query_ball_point(xyz[idx], radius):
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(cluster) >= min_points:
            pts = xyz[cluster]
            sizes.append(len(cluster))
            diagonals.append(float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))))
    return np.asarray(sizes, dtype=np.float64), np.asarray(diagonals, dtype=np.float64)


def print_cluster_block(name, xyz, radius, min_points):
    sizes, diagonals = cluster_summary(xyz, radius, min_points)
    print(f"{name} clusters radius={fmt_mm(radius)} min_points={min_points}:")
    if len(sizes) == 0:
        print("  no clusters")
        return
    clustered = int(np.sum(sizes))
    print(
        "  clusters={} clustered_points={} ({:.1f}%) size median={:.0f} p95={:.0f} max={:.0f}".format(
            len(sizes),
            clustered,
            clustered / len(xyz) * 100.0,
            float(np.median(sizes)),
            float(np.percentile(sizes, 95)),
            float(np.max(sizes)),
        )
    )
    print(
        "  bbox_diag median={} p95={} max={}".format(
            fmt_mm(float(np.median(diagonals))),
            fmt_mm(float(np.percentile(diagonals, 95))),
            fmt_mm(float(np.max(diagonals))),
        )
    )


def print_vertical_diagnostics(ref_xyz, test_xyz):
    print("\nVertical / Range Diagnostics")
    for name, xyz in (("Reference", ref_xyz), ("Test", test_xyz)):
        z = xyz[:, 2]
        percentiles = [0, 1, 5, 50, 95, 99, 100]
        values = [float(np.percentile(z, p)) for p in percentiles]
        print(
            "  {} z min/p1/p5/median/p95/p99/max = {}".format(
                name,
                "[" + ", ".join(f"{v:.3f}" for v in values) + "]",
            )
        )
        high_abs = np.abs(z) > 2.0
        print(
            "  {} |z| > 2m: {} / {} ({:.2f}%)".format(
                name,
                int(np.count_nonzero(high_abs)),
                len(z),
                float(np.mean(high_abs) * 100.0),
            )
        )
    lo = float(np.min(ref_xyz[:, 2]) - 0.2)
    hi = float(np.max(ref_xyz[:, 2]) + 0.2)
    outside = (test_xyz[:, 2] < lo) | (test_xyz[:, 2] > hi)
    print(
        "  Test outside reference z range +/-0.2m [{:.3f}, {:.3f}]: {} / {} ({:.2f}%)".format(
            lo,
            hi,
            int(np.count_nonzero(outside)),
            len(test_xyz),
            float(np.mean(outside) * 100.0),
        )
    )


def rigid_fit(source, target):
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    src = source - source_centroid
    dst = target - target_centroid
    h = src.T @ dst
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = target_centroid - r @ source_centroid
    return r, t


def run_icp(test_xyz, ref_xyz, iterations, trim_percentile):
    transformed = test_xyz.copy()
    total_r = np.eye(3)
    total_t = ref_xyz.mean(axis=0) - test_xyz.mean(axis=0)
    transformed += total_t
    tree = cKDTree(ref_xyz)
    last_mean = None
    for _ in range(iterations):
        distances, indices = tree.query(transformed, k=1, workers=-1)
        cutoff = np.percentile(distances, trim_percentile)
        keep = distances <= cutoff
        if np.count_nonzero(keep) < 3:
            break
        r, t = rigid_fit(transformed[keep], ref_xyz[indices[keep]])
        transformed = (r @ transformed.T).T + t
        total_r = r @ total_r
        total_t = r @ total_t + t
        mean = float(np.mean(distances[keep]))
        if last_mean is not None and abs(last_mean - mean) < 1e-6:
            break
        last_mean = mean
    return transformed, total_r, total_t


def transform_with_yaw(points, yaw, translation):
    c = math.cos(yaw)
    s = math.sin(yaw)
    r = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (r @ points.T).T + translation, r


def run_yaw_grid_alignment(test_xyz, ref_xyz, step_deg, trim_percentile):
    ref_centroid = ref_xyz.mean(axis=0)
    test_centroid = test_xyz.mean(axis=0)
    centered_test = test_xyz - test_centroid
    tree = cKDTree(ref_xyz)
    best = None
    step = math.radians(step_deg)
    count = max(1, int(round((2.0 * math.pi) / step)))
    for i in range(count):
        yaw = -math.pi + i * step
        transformed, r = transform_with_yaw(centered_test, yaw, ref_centroid)
        distances, _ = tree.query(transformed, k=1, workers=-1)
        cutoff = np.percentile(distances, trim_percentile)
        kept = distances[distances <= cutoff]
        if len(kept) == 0:
            continue
        score = float(np.mean(kept))
        if best is None or score < best["score"]:
            translation = ref_centroid - r @ test_centroid
            best = {
                "score": score,
                "yaw": yaw,
                "translation": translation,
                "transformed": transformed,
            }
    if best is None:
        raise RuntimeError("yaw grid alignment failed")
    return best["transformed"], best["yaw"], best["translation"], best["score"]


def evaluate_pair(ref_xyz, test_xyz, label):
    ref_to_test, _ = nearest_distances(ref_xyz, test_xyz)
    test_to_ref, _ = nearest_distances(test_xyz, ref_xyz)
    print(f"\n{label}")
    print_distance_block("  ref -> test nearest distance:", ref_to_test)
    print_distance_block("  test -> ref nearest distance:", test_to_ref)
    chamfer_mean = float(np.mean(ref_to_test) + np.mean(test_to_ref)) / 2.0
    print(f"  symmetric mean distance: {fmt_mm(chamfer_mean)}")


def main():
    parser = argparse.ArgumentParser(description="Compare two high-reflectivity PCD maps.")
    parser.add_argument("--ref", required=True, help="Reference PCD, e.g. Super-LIO filtered map")
    parser.add_argument("--test", required=True, help="Test PCD, e.g. reflective mapper output")
    parser.add_argument("--thickness-radius", type=float, default=0.05)
    parser.add_argument("--thickness-min-neighbors", type=int, default=5)
    parser.add_argument("--cluster-radius", type=float, default=0.08)
    parser.add_argument("--cluster-min-points", type=int, default=3)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--icp-trim-percentile", type=float, default=80.0)
    parser.add_argument("--yaw-grid-step-deg", type=float, default=2.0)
    parser.add_argument("--yaw-grid-trim-percentile", type=float, default=80.0)
    parser.add_argument(
        "--save-aligned-test",
        default="",
        help="Optional output path for the ICP-aligned test cloud.",
    )
    parser.add_argument("--no-icp", action="store_true")
    args = parser.parse_args()

    ref_xyz, ref_intensity, ref_fields = load_ascii_pcd(args.ref)
    test_xyz, test_intensity, test_fields = load_ascii_pcd(args.test)

    print("Loaded PCDs")
    print(f"  ref={args.ref}")
    print(f"  test={args.test}")
    print(f"  ref_fields={ref_fields}")
    print(f"  test_fields={test_fields}")
    print()
    bounds_summary("Reference", ref_xyz, ref_intensity)
    bounds_summary("Test", test_xyz, test_intensity)

    evaluate_pair(ref_xyz, test_xyz, "Raw Coordinate Comparison")

    if not args.no_icp:
        yaw_aligned, grid_yaw, grid_t, grid_score = run_yaw_grid_alignment(
            test_xyz,
            ref_xyz,
            step_deg=args.yaw_grid_step_deg,
            trim_percentile=args.yaw_grid_trim_percentile,
        )
        print("\nYaw-Grid Initial Alignment test -> ref")
        print(f"  yaw={math.degrees(grid_yaw):.2f}deg")
        print(f"  translation=[{grid_t[0]:.3f}, {grid_t[1]:.3f}, {grid_t[2]:.3f}]")
        print(f"  trimmed_mean={fmt_mm(grid_score)}")
        evaluate_pair(ref_xyz, yaw_aligned, "Yaw-Grid Aligned Geometry Comparison")

        aligned_test, r, t = run_icp(
            yaw_aligned,
            ref_xyz,
            iterations=args.icp_iterations,
            trim_percentile=args.icp_trim_percentile,
        )
        yaw = math.atan2(r[1, 0], r[0, 0])
        print("\nICP transform test -> ref")
        print(f"  translation=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
        print(f"  incremental_yaw={math.degrees(yaw):.2f}deg")
        evaluate_pair(ref_xyz, aligned_test, "ICP-Aligned Geometry Comparison")
        if args.save_aligned_test:
            write_ascii_pcd(args.save_aligned_test, aligned_test, test_intensity)
            print(f"\nSaved ICP-aligned test PCD: {args.save_aligned_test}")

    print("\nLocal Geometry")
    print_thickness(
        "Reference",
        ref_xyz,
        args.thickness_radius,
        args.thickness_min_neighbors,
    )
    print_thickness(
        "Test",
        test_xyz,
        args.thickness_radius,
        args.thickness_min_neighbors,
    )

    print("\nCluster Structure")
    print_cluster_block("Reference", ref_xyz, args.cluster_radius, args.cluster_min_points)
    print_cluster_block("Test", test_xyz, args.cluster_radius, args.cluster_min_points)
    print_vertical_diagnostics(ref_xyz, test_xyz)


if __name__ == "__main__":
    main()
