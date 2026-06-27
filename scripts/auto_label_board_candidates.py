#!/usr/bin/env python3
"""Offline auto-labeler for reflective board candidate regions in rosbags.

The tool discovers candidate ROIs from the first few seconds of each bag,
accumulates ROI clouds over the full bag duration, matches every candidate
against the recorded board templates, and writes a review-friendly dataset:

  Data/
    Image/bag1/metadata.yaml + <board_id>.bmp
    PointCloud/bag1/<board_id>.yaml
"""

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import yaml

YAML_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)

try:
    import rosbag
except ImportError as exc:
    rosbag = None
    ROSBAG_IMPORT_ERROR = exc
else:
    ROSBAG_IMPORT_ERROR = None

try:
    import sensor_msgs.point_cloud2 as pc2
except ImportError:
    pc2 = None


POINT_RE = re.compile(r"\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)(?:\s*,\s*([-+0-9.eE]+))?")


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def sanitize_token(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "unset"


def natural_key(path):
    name = Path(path).name
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part for part in parts]


def bag_output_index(path, fallback):
    stem = Path(path).stem
    return int(stem) if stem.isdigit() else fallback


def board_number(label):
    match = re.search(r"(\d+)$", label)
    if not match:
        return label
    return str(int(match.group(1)))


def load_params(path):
    defaults = {
        "reflectivity_threshold": 100,
        "min_distance_m": 0.1,
        "max_distance_m": 30.0,
        "cluster_tolerance_m": 0.25,
        "min_cluster_points": 3,
        "max_cluster_points": 2000,
        "roi_padding_m": 0.15,
        "min_roi_radius_m": 0.30,
        "max_roi_radius_m": 0.60,
        "template_grid_cols": 48,
        "template_grid_rows": 48,
        "template_mid_grid_cols": 32,
        "template_mid_grid_rows": 32,
        "template_coarse_grid_cols": 24,
        "template_coarse_grid_rows": 24,
        "min_template_points": 20,
        "target_template_match_points": 120,
        "template_plane_tolerance_m": 0.045,
    }
    if not path or not Path(path).exists():
        return defaults
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    task1 = data.get("task1", {}) or {}
    collector = data.get("dataset_collector", {}) or {}
    defaults["reflectivity_threshold"] = int(data.get("reflectivity_threshold", defaults["reflectivity_threshold"]))
    defaults["min_distance_m"] = float(data.get("min_distance_m", defaults["min_distance_m"]))
    defaults["max_distance_m"] = float(data.get("max_distance_m", defaults["max_distance_m"]))
    defaults["cluster_tolerance_m"] = float(
        collector.get("cluster_tolerance_m",
                      task1.get("discovery_cluster_tolerance_m", defaults["cluster_tolerance_m"])))
    defaults["min_cluster_points"] = int(
        collector.get("min_cluster_points",
                      task1.get("min_discovery_cluster_points", defaults["min_cluster_points"])))
    defaults["max_cluster_points"] = int(
        collector.get("max_cluster_points",
                      task1.get("max_discovery_cluster_points", defaults["max_cluster_points"])))
    for key in ("roi_padding_m", "min_roi_radius_m", "max_roi_radius_m"):
        defaults[key] = float(collector.get(key, task1.get(key, defaults[key])))
    for key in (
        "template_grid_cols", "template_grid_rows",
        "template_mid_grid_cols", "template_mid_grid_rows",
        "template_coarse_grid_cols", "template_coarse_grid_rows",
        "min_template_points", "target_template_match_points",
        "template_plane_tolerance_m",
    ):
        if key in task1:
            defaults[key] = task1[key]
    return defaults


def default_match_plane_offset(config):
    return max(0.08, 2.0 * float(config["template_plane_tolerance_m"]))


def uv_bounds(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return {
            "count": 0,
            "min_u": 0.0,
            "max_u": 0.0,
            "min_v": 0.0,
            "max_v": 0.0,
            "span_u": 0.0,
            "span_v": 0.0,
            "long_span": 0.0,
            "short_span": 0.0,
            "area": 0.0,
        }
    min_u = float(points[:, 0].min())
    max_u = float(points[:, 0].max())
    min_v = float(points[:, 1].min())
    max_v = float(points[:, 1].max())
    span_u = max_u - min_u
    span_v = max_v - min_v
    return {
        "count": int(len(points)),
        "min_u": min_u,
        "max_u": max_u,
        "min_v": min_v,
        "max_v": max_v,
        "span_u": float(span_u),
        "span_v": float(span_v),
        "long_span": float(max(span_u, span_v)),
        "short_span": float(min(span_u, span_v)),
        "area": float(span_u * span_v),
    }


def yaml_bounds(bounds):
    return {
        "count": int(bounds.get("count", 0)),
        "min_u_m": float(bounds.get("min_u", 0.0)),
        "max_u_m": float(bounds.get("max_u", 0.0)),
        "min_v_m": float(bounds.get("min_v", 0.0)),
        "max_v_m": float(bounds.get("max_v", 0.0)),
        "span_u_m": float(bounds.get("span_u", 0.0)),
        "span_v_m": float(bounds.get("span_v", 0.0)),
        "long_span_m": float(bounds.get("long_span", 0.0)),
        "short_span_m": float(bounds.get("short_span", 0.0)),
        "area_m2": float(bounds.get("area", 0.0)),
    }


def add_reason(reasons, reason):
    if reason and reason not in reasons:
        reasons.append(reason)


def dump_yaml(data, handle):
    yaml.dump(
        data,
        handle,
        Dumper=YAML_DUMPER,
        allow_unicode=False,
        sort_keys=False,
        default_flow_style=False,
    )


def parse_template_uv(path):
    points = []
    in_points = False
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "points_local_uv:":
                in_points = True
                continue
            if not in_points:
                continue
            if line.startswith("- ["):
                match = POINT_RE.search(line)
                if match:
                    points.append([float(match.group(1)), float(match.group(2))])
            elif line and not line.startswith("-"):
                break
    return np.asarray(points, dtype=np.float64)


def normalize_uv(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / len(points)
    _, vectors = np.linalg.eigh(covariance)
    axes = np.column_stack((vectors[:, 1], vectors[:, 0]))
    q = centered @ axes
    span_x = max(1e-4, float(q[:, 0].max() - q[:, 0].min()))
    span_y = max(1e-4, float(q[:, 1].max() - q[:, 1].min()))
    return q / max(span_x, span_y)


def build_descriptor(points, cols, rows, flip_x=False, flip_y=False):
    grid = np.zeros(cols * rows, dtype=np.float64)
    normalized = normalize_uv(points)
    for point in normalized:
        x = -point[0] if flip_x else point[0]
        y = -point[1] if flip_y else point[1]
        x += 0.5
        y += 0.5
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            continue
        col = min(cols - 1, int(math.floor(x * cols)))
        row = min(rows - 1, int(math.floor(y * rows)))
        grid[row * cols + col] += 1.0
    grid = np.sqrt(grid)
    norm = np.linalg.norm(grid)
    if norm > 1e-9:
        grid /= norm
    return grid


def descriptor_variants(points, cols, rows):
    return [
        build_descriptor(points, cols, rows, False, False),
        build_descriptor(points, cols, rows, True, False),
        build_descriptor(points, cols, rows, False, True),
        build_descriptor(points, cols, rows, True, True),
    ]


def load_templates(template_dir, config):
    templates = []
    for path in sorted(Path(template_dir).glob("board_*.yaml")):
        label = path.stem
        uv = parse_template_uv(path)
        if len(uv) < config["min_template_points"]:
            continue
        templates.append({
            "label": label,
            "board_id": board_number(label),
            "path": str(path),
            "points": uv,
            "fine": descriptor_variants(uv, int(config["template_grid_cols"]), int(config["template_grid_rows"])),
            "mid": descriptor_variants(uv, int(config["template_mid_grid_cols"]), int(config["template_mid_grid_rows"])),
            "coarse": descriptor_variants(uv, int(config["template_coarse_grid_cols"]), int(config["template_coarse_grid_rows"])),
        })
    return templates


def point_from_xyz_reflectivity(x, y, z, reflectivity):
    x = float(x)
    y = float(y)
    z = float(z)
    reflectivity = int(clamp(round(float(reflectivity)), 0, 255))
    distance = math.sqrt(x * x + y * y + z * z)
    return (x, y, z, reflectivity, distance)


def point_from_raw(raw):
    x = float(raw.x)
    y = float(raw.y)
    z = float(raw.z)
    reflectivity = int(raw.reflectivity)
    return point_from_xyz_reflectivity(x, y, z, reflectivity)


def pointcloud2_field_names(msg):
    return [field.name for field in getattr(msg, "fields", [])]


def pointfield_dtype(datatype, is_bigendian):
    endian = ">" if is_bigendian else "<"
    mapping = {
        1: np.int8,
        2: np.uint8,
        3: endian + "i2",
        4: endian + "u2",
        5: endian + "i4",
        6: endian + "u4",
        7: endian + "f4",
        8: endian + "f8",
    }
    return mapping.get(datatype)


def pointcloud2_to_array(msg, config, high_only=False):
    fields = {field.name: field for field in getattr(msg, "fields", [])}
    if not all(name in fields for name in ("x", "y", "z")):
        return np.zeros((0, 5), dtype=np.float64)

    reflectivity_name = "reflectivity" if "reflectivity" in fields else None
    if reflectivity_name is None and "intensity" in fields:
        reflectivity_name = "intensity"

    names = ["x", "y", "z"] + ([reflectivity_name] if reflectivity_name else [])
    dtype_names = []
    dtype_formats = []
    dtype_offsets = []
    for name in names:
        field = fields[name]
        fmt = pointfield_dtype(field.datatype, msg.is_bigendian)
        if fmt is None or field.count != 1:
            return pointcloud2_to_array_slow(msg, config, high_only)
        dtype_names.append(name)
        dtype_formats.append(fmt)
        dtype_offsets.append(field.offset)
    dtype = np.dtype({
        "names": dtype_names,
        "formats": dtype_formats,
        "offsets": dtype_offsets,
        "itemsize": msg.point_step,
    })

    count = int(msg.width) * int(msg.height)
    if count <= 0:
        return np.zeros((0, 5), dtype=np.float64)
    if msg.row_step == msg.point_step * msg.width:
        structured = np.frombuffer(msg.data, dtype=dtype, count=count)
    else:
        rows = []
        for row in range(msg.height):
            start = row * msg.row_step
            end = start + msg.point_step * msg.width
            rows.append(np.frombuffer(msg.data[start:end], dtype=dtype,
                                      count=msg.width))
        structured = np.concatenate(rows) if rows else np.zeros((0,), dtype=dtype)

    x = structured["x"].astype(np.float64, copy=False)
    y = structured["y"].astype(np.float64, copy=False)
    z = structured["z"].astype(np.float64, copy=False)
    if reflectivity_name:
        reflectivity = structured[reflectivity_name].astype(np.float64, copy=False)
    else:
        reflectivity = np.full(x.shape, 255.0, dtype=np.float64)
    reflectivity = np.clip(np.rint(reflectivity), 0, 255)
    distance = np.sqrt(x * x + y * y + z * z)
    mask = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z) &
        (distance >= config["min_distance_m"]) &
        (distance <= config["max_distance_m"])
    )
    if high_only:
        mask &= reflectivity >= config["reflectivity_threshold"]
    if not np.any(mask):
        return np.zeros((0, 5), dtype=np.float64)
    return np.column_stack((x[mask], y[mask], z[mask], reflectivity[mask],
                            distance[mask]))


def pointcloud2_to_array_slow(msg, config, high_only=False):
    if pc2 is None:
        raise RuntimeError("sensor_msgs.point_cloud2 is required for PointCloud2 bags")
    fields = pointcloud2_field_names(msg)
    reflectivity_field = "reflectivity" if "reflectivity" in fields else None
    if reflectivity_field is None and "intensity" in fields:
        reflectivity_field = "intensity"
    field_names = ["x", "y", "z"] + ([reflectivity_field] if reflectivity_field else [])
    points = []
    for values in pc2.read_points(msg, field_names=field_names, skip_nans=True):
        if reflectivity_field:
            x, y, z, reflectivity = values
        else:
            x, y, z = values
            reflectivity = 255
        point = point_from_xyz_reflectivity(x, y, z, reflectivity)
        if is_valid_range(point, config) and (not high_only or point[3] >= config["reflectivity_threshold"]):
            points.append(point)
    return np.asarray(points, dtype=np.float64) if points else np.zeros((0, 5), dtype=np.float64)


def custom_msg_to_array(msg, config, high_only=False):
    points = []
    for raw in msg.points:
        point = point_from_raw(raw)
        if is_valid_range(point, config) and (not high_only or point[3] >= config["reflectivity_threshold"]):
            points.append(point)
    return np.asarray(points, dtype=np.float64) if points else np.zeros((0, 5), dtype=np.float64)


def msg_points_array(msg, config, high_only=False):
    if hasattr(msg, "fields") and hasattr(msg, "data"):
        return pointcloud2_to_array(msg, config, high_only)
    if hasattr(msg, "points"):
        return custom_msg_to_array(msg, config, high_only)
    return np.zeros((0, 5), dtype=np.float64)


def iter_msg_points(msg, config=None, high_only=False):
    if config is not None:
        arr = msg_points_array(msg, config, high_only)
        for row in arr:
            yield (float(row[0]), float(row[1]), float(row[2]),
                   int(row[3]), float(row[4]))
        return

    if hasattr(msg, "fields") and hasattr(msg, "data"):
        if pc2 is None:
            raise RuntimeError("sensor_msgs.point_cloud2 is required for PointCloud2 bags")
        fields = pointcloud2_field_names(msg)
        if not all(name in fields for name in ("x", "y", "z")):
            return
        reflectivity_field = "reflectivity" if "reflectivity" in fields else None
        if reflectivity_field is None and "intensity" in fields:
            reflectivity_field = "intensity"
        field_names = ["x", "y", "z"] + ([reflectivity_field] if reflectivity_field else [])
        for values in pc2.read_points(msg, field_names=field_names, skip_nans=True):
            if reflectivity_field:
                x, y, z, reflectivity = values
            else:
                x, y, z = values
                reflectivity = 255
            yield point_from_xyz_reflectivity(x, y, z, reflectivity)
        return

    if hasattr(msg, "points"):
        for raw in msg.points:
            yield point_from_raw(raw)


def is_valid_range(point, config):
    x, y, z, _, distance = point
    return (
        math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and
        config["min_distance_m"] <= distance <= config["max_distance_m"]
    )


def is_high_reflective(point, config):
    return is_valid_range(point, config) and point[3] >= config["reflectivity_threshold"]


def squared_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def cluster_points(points, tolerance, min_points, max_points):
    if not points:
        return []
    cell_size = max(1e-6, tolerance)
    grid = {}
    for i, point in enumerate(points):
        key = (
            int(math.floor(point[0] / cell_size)),
            int(math.floor(point[1] / cell_size)),
            int(math.floor(point[2] / cell_size)),
        )
        grid.setdefault(key, []).append(i)

    visited = [False] * len(points)
    tolerance_sq = tolerance * tolerance
    clusters = []
    for i in range(len(points)):
        if visited[i]:
            continue
        visited[i] = True
        queue = [i]
        indices = []
        while queue:
            current = queue.pop()
            indices.append(current)
            point = points[current]
            key = (
                int(math.floor(point[0] / cell_size)),
                int(math.floor(point[1] / cell_size)),
                int(math.floor(point[2] / cell_size)),
            )
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for other in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), []):
                            if visited[other]:
                                continue
                            if squared_distance(point, points[other]) <= tolerance_sq:
                                visited[other] = True
                                queue.append(other)
        if len(indices) >= min_points and (max_points <= 0 or len(indices) <= max_points):
            arr = np.asarray([[points[idx][0], points[idx][1], points[idx][2]] for idx in indices], dtype=np.float64)
            center = arr.mean(axis=0)
            clusters.append({
                "indices": indices,
                "center": center,
                "max_reflectivity": max(points[idx][3] for idx in indices),
            })
    clusters.sort(key=lambda c: (len(c["indices"]), c["max_reflectivity"]), reverse=True)
    return clusters


def estimate_plane(points, min_points, match_plane_offset):
    if len(points) < min_points:
        return None
    arr = np.asarray([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
    origin = arr.mean(axis=0)
    centered = arr - origin
    covariance = centered.T @ centered / len(arr)
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    normal /= max(1e-12, np.linalg.norm(normal))
    x_axis = vectors[:, 2]
    x_axis = x_axis - float(np.dot(x_axis, normal)) * normal
    if np.linalg.norm(x_axis) < 1e-4:
        return None
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    if np.linalg.norm(y_axis) < 1e-4:
        return None
    y_axis /= np.linalg.norm(y_axis)
    to_lidar = -origin
    if np.linalg.norm(to_lidar) > 1e-4 and float(np.dot(normal, to_lidar / np.linalg.norm(to_lidar))) < 0.0:
        normal = -normal
        y_axis = -y_axis
    local = []
    for p in points:
        rel = np.asarray(p[:3], dtype=np.float64) - origin
        local.append([float(np.dot(rel, x_axis)), float(np.dot(rel, y_axis)), float(np.dot(rel, normal)), p[3]])
    w = np.asarray([p[2] for p in local], dtype=np.float64)
    match_local = [p for p in local if abs(p[2]) <= match_plane_offset]
    local_uv = np.asarray([[p[0], p[1]] for p in local], dtype=np.float64)
    match_local_uv = np.asarray([[p[0], p[1]] for p in match_local], dtype=np.float64)
    match_point_ratio = len(match_local) / float(len(local)) if local else 0.0
    return {
        "origin": origin,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "normal": normal,
        "local": local,
        "local_uv": local_uv,
        "local_bounds": uv_bounds(local_uv),
        "match_local": match_local,
        "match_local_uv": match_local_uv,
        "match_bounds": uv_bounds(match_local_uv),
        "match_plane_offset_m": float(match_plane_offset),
        "match_point_ratio": float(match_point_ratio),
        "rms_m": float(math.sqrt(float(np.mean(w * w)))) if len(w) else 0.0,
        "max_abs_w_m": float(np.max(np.abs(w))) if len(w) else 0.0,
    }


def template_score(candidate_uv, template, config):
    if len(candidate_uv) < config["min_template_points"]:
        return 0.0
    cols = int(config["template_grid_cols"])
    rows = int(config["template_grid_rows"])
    template_variants = template["fine"]
    if len(candidate_uv) < 0.5 * config["target_template_match_points"]:
        cols = int(config["template_coarse_grid_cols"])
        rows = int(config["template_coarse_grid_rows"])
        template_variants = template["coarse"]
    elif len(candidate_uv) < config["target_template_match_points"]:
        cols = int(config["template_mid_grid_cols"])
        rows = int(config["template_mid_grid_rows"])
        template_variants = template["mid"]

    candidates = descriptor_variants(candidate_uv, cols, rows)
    swapped = descriptor_variants(candidate_uv[:, ::-1], cols, rows)
    best = 0.0
    for cand in candidates + swapped:
        for templ in template_variants:
            best = max(best, float(np.dot(cand, templ)))
    return best


def candidate_descriptor_set(candidate_uv, config):
    if len(candidate_uv) < config["min_template_points"]:
        return None, []
    level = "fine"
    cols = int(config["template_grid_cols"])
    rows = int(config["template_grid_rows"])
    if len(candidate_uv) < 0.5 * config["target_template_match_points"]:
        level = "coarse"
        cols = int(config["template_coarse_grid_cols"])
        rows = int(config["template_coarse_grid_rows"])
    elif len(candidate_uv) < config["target_template_match_points"]:
        level = "mid"
        cols = int(config["template_mid_grid_cols"])
        rows = int(config["template_mid_grid_rows"])
    variants = descriptor_variants(candidate_uv, cols, rows)
    variants.extend(descriptor_variants(candidate_uv[:, ::-1], cols, rows))
    return level, variants


def rank_templates(candidate_uv, templates, config):
    level, candidate_variants = candidate_descriptor_set(candidate_uv, config)
    if not candidate_variants:
        return []
    rows = []
    for template in templates:
        template_variants = template[level]
        score = 0.0
        for candidate in candidate_variants:
            for templ in template_variants:
                score = max(score, float(np.dot(candidate, templ)))
        rows.append({
            "label": template["label"],
            "board_id": template["board_id"],
            "score": score,
        })
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def assign_unique(candidates, templates):
    pairs = []
    for candidate in candidates:
        for rank in candidate["scores"]:
            pairs.append((rank["score"], candidate["candidate_id"], rank["board_id"], rank["label"]))
    pairs.sort(reverse=True, key=lambda item: item[0])
    used_candidates = set()
    used_boards = set()
    assignment = {}
    for score, candidate_id, board_id, label in pairs:
        if candidate_id in used_candidates or board_id in used_boards:
            continue
        assignment[candidate_id] = (board_id, label, score)
        used_candidates.add(candidate_id)
        used_boards.add(board_id)
        if len(used_candidates) == min(len(candidates), len(templates)):
            break
    return assignment


def sample_points(points, limit):
    if limit <= 0 or len(points) <= limit:
        return points
    stride = len(points) / float(limit)
    return [points[min(len(points) - 1, int(math.floor(i * stride)))] for i in range(limit)]


def yaml_float_list(values):
    return [float(v) for v in values]


def write_bmp(path, uv_points, size=640, padding=48, radius=2):
    if len(uv_points) == 0:
        return False
    points = np.asarray(uv_points, dtype=np.float64)
    min_u = float(points[:, 0].min())
    max_u = float(points[:, 0].max())
    min_v = float(points[:, 1].min())
    max_v = float(points[:, 1].max())
    size = max(128, int(size))
    padding = clamp(int(padding), 0, size // 3)
    radius = max(1, int(radius))
    drawable = max(1, size - 2 * padding)
    scale = min(drawable / max(1e-9, max_u - min_u),
                drawable / max(1e-9, max_v - min_v))
    image = bytearray([255] * (size * size * 3))

    def set_black(x, y):
        if 0 <= x < size and 0 <= y < size:
            off = (y * size + x) * 3
            image[off:off + 3] = b"\x00\x00\x00"

    for u, v in points:
        x = int(round((u - min_u) * scale + padding))
        y = int(round((max_v - v) * scale + padding))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    set_black(x + dx, y + dy)

    row_bytes = size * 3
    row_stride = (row_bytes + 3) & ~3
    pixel_bytes = row_stride * size
    file_size = 14 + 40 + pixel_bytes
    with open(path, "wb") as handle:
        handle.write(b"BM")
        handle.write(file_size.to_bytes(4, "little"))
        handle.write((0).to_bytes(2, "little"))
        handle.write((0).to_bytes(2, "little"))
        handle.write((54).to_bytes(4, "little"))
        handle.write((40).to_bytes(4, "little"))
        handle.write(size.to_bytes(4, "little"))
        handle.write(size.to_bytes(4, "little"))
        handle.write((1).to_bytes(2, "little"))
        handle.write((24).to_bytes(2, "little"))
        handle.write((0).to_bytes(4, "little"))
        handle.write(pixel_bytes.to_bytes(4, "little"))
        handle.write((2835).to_bytes(4, "little"))
        handle.write((2835).to_bytes(4, "little"))
        handle.write((0).to_bytes(4, "little"))
        handle.write((0).to_bytes(4, "little"))
        pad = bytes(row_stride - row_bytes)
        for y in range(size - 1, -1, -1):
            row = image[y * size * 3:(y + 1) * size * 3]
            bgr = bytearray()
            for i in range(0, len(row), 3):
                bgr.extend((row[i + 2], row[i + 1], row[i]))
            handle.write(bgr)
            handle.write(pad)
    return True


def point_xyzr(point):
    return [float(point[0]), float(point[1]), float(point[2]), int(point[3])]


def cumulative_frame_points(frames):
    accumulated = []
    for frame_points in frames:
        accumulated.extend(frame_points)
        yield list(accumulated)


def write_frame_points_yaml(path, frame_count, points):
    data = {
        "frame_count": int(frame_count),
        "point_count": len(points),
        "points_xyzr": [point_xyzr(point) for point in points],
    }
    with open(path, "w", encoding="utf-8") as handle:
        dump_yaml(data, handle)


def write_frame_accumulations(pointcloud_dir, base_name, candidate, args):
    frames = candidate.get("high_points_by_frame", [])
    frame_count = min(int(args.frame_accumulation_count), len(frames))
    if frame_count <= 0:
        candidate["frame_accumulation_dir"] = ""
        candidate["frame_accumulation_files"] = []
        return

    frame_dir_name = f"{base_name}_frames"
    frame_dir = pointcloud_dir / frame_dir_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for index, points in enumerate(cumulative_frame_points(frames[:frame_count]), start=1):
        if args.max_frame_accumulation_points > 0:
            saved_points = sample_points(points, args.max_frame_accumulation_points)
        else:
            saved_points = points
        file_name = f"frame_{index:03d}.yaml"
        write_frame_points_yaml(frame_dir / file_name, index, saved_points)
        files.append({
            "frame_count": index,
            "point_count": len(points),
            "saved_point_count": len(saved_points),
            "file": f"{frame_dir_name}/{file_name}",
        })

    candidate["frame_accumulation_dir"] = frame_dir_name
    candidate["frame_accumulation_files"] = files


def write_per_frame_points_yaml(path, bag_info, candidate, frame, points):
    data = {
        "sample_version": 1,
        "collection_mode": "auto_board_label_per_frame",
        "label": f"board_{int(candidate['assigned_board_id']):02d}" if str(candidate["assigned_board_id"]).isdigit() else candidate["assigned_board_id"],
        "board_id": candidate["assigned_board_id"],
        "bag_name": bag_info["bag_name"],
        "bag_index": bag_info["bag_index"],
        "input_topic": bag_info["topic"],
        "frame_id": bag_info.get("frame_id", ""),
        "candidate_index": candidate["candidate_id"],
        "left_right_index": int(candidate.get("left_right_index", candidate["candidate_id"] + 1)),
        "frame_index": int(frame["frame_index"]),
        "stamp_sec": float(frame["stamp_sec"]),
        "elapsed_sec": float(frame["elapsed_sec"]),
        "candidate_center_xyz": yaml_float_list(candidate["center"]),
        "roi_radius_m": float(candidate["roi_radius"]),
        "reflectivity_threshold": int(bag_info["reflectivity_threshold"]),
        "point_count": len(points),
        "points_xyzr": [point_xyzr(point) for point in points],
        "points_xyz": [[float(point[0]), float(point[1]), float(point[2])] for point in points],
    }
    with open(path, "w", encoding="utf-8") as handle:
        dump_yaml(data, handle)


def write_per_frame_points(pointcloud_dir, base_name, bag_info, candidate, args):
    frames = candidate.get("per_frame_points", [])
    if not args.save_per_frame_points or not frames:
        candidate["per_frame_dir"] = ""
        candidate["per_frame_index_file"] = ""
        candidate["per_frame_file_count"] = 0
        candidate["per_frame_nonempty_count"] = 0
        candidate["per_frame_total_point_count"] = 0
        return

    frame_dir_name = f"{base_name}_per_frame"
    frame_dir = pointcloud_dir / frame_dir_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    files = []
    nonempty_count = 0
    total_point_count = 0
    for frame in frames:
        points = frame["points"]
        if args.skip_empty_per_frame_points and not points:
            continue
        saved_points = sample_points(points, args.max_per_frame_points)
        file_name = f"frame_{int(frame['frame_index']):06d}.yaml"
        write_per_frame_points_yaml(frame_dir / file_name, bag_info, candidate,
                                    frame, saved_points)
        point_count = len(points)
        if point_count:
            nonempty_count += 1
        total_point_count += point_count
        files.append({
            "frame_index": int(frame["frame_index"]),
            "stamp_sec": float(frame["stamp_sec"]),
            "elapsed_sec": float(frame["elapsed_sec"]),
            "point_count": point_count,
            "saved_point_count": len(saved_points),
            "file": f"{frame_dir_name}/{file_name}",
        })

    index_data = {
        "collection_mode": "auto_board_label_per_frame_index",
        "label": f"board_{int(candidate['assigned_board_id']):02d}" if str(candidate["assigned_board_id"]).isdigit() else candidate["assigned_board_id"],
        "board_id": candidate["assigned_board_id"],
        "bag_name": bag_info["bag_name"],
        "bag_index": bag_info["bag_index"],
        "input_topic": bag_info["topic"],
        "frame_id": bag_info.get("frame_id", ""),
        "candidate_index": candidate["candidate_id"],
        "left_right_index": int(candidate.get("left_right_index", candidate["candidate_id"] + 1)),
        "frame_count": len(frames),
        "file_count": len(files),
        "nonempty_frame_count": nonempty_count,
        "total_point_count": total_point_count,
        "skip_empty_per_frame_points": bool(args.skip_empty_per_frame_points),
        "max_per_frame_points": int(args.max_per_frame_points),
        "files": files,
    }
    index_file = frame_dir / "index.yaml"
    with open(index_file, "w", encoding="utf-8") as handle:
        dump_yaml(index_data, handle)

    candidate["per_frame_dir"] = frame_dir_name
    candidate["per_frame_index_file"] = f"{frame_dir_name}/index.yaml"
    candidate["per_frame_file_count"] = len(files)
    candidate["per_frame_nonempty_count"] = nonempty_count
    candidate["per_frame_total_point_count"] = total_point_count


def write_pointcloud_yaml(path, bag_info, candidate, config, max_roi_points, max_high_points):
    high_points = sample_points(candidate["high_points"], max_high_points)
    roi_points = sample_points(candidate["roi_points"], max_roi_points)
    plane = candidate.get("plane")
    local = plane.get("local", []) if plane else []
    match_local = plane.get("match_local", []) if plane else []
    local_sample = sample_points(local, max_high_points)
    match_local_sample = sample_points(match_local, max_high_points)
    data = {
        "sample_version": 2,
        "collection_mode": "auto_board_label",
        "label": f"board_{int(candidate['assigned_board_id']):02d}" if str(candidate["assigned_board_id"]).isdigit() else candidate["assigned_board_id"],
        "board_id": candidate["assigned_board_id"],
        "bag_name": bag_info["bag_name"],
        "bag_index": bag_info["bag_index"],
        "input_topic": bag_info["topic"],
        "frame_id": bag_info.get("frame_id", ""),
        "candidate_index": candidate["candidate_id"],
        "left_right_index": int(candidate.get("left_right_index", candidate["candidate_id"] + 1)),
        "known_order_board_id": candidate.get("known_order_board_id", ""),
        "assignment_mode": bag_info.get("assignment_mode", ""),
        "known_order_cycle": bool(bag_info.get("known_order_cycle", False)),
        "known_order_ids": list(bag_info.get("known_order_ids", [])),
        "left_right_axis": bag_info.get("left_right_axis", ""),
        "left_right_descending": bool(bag_info.get("left_right_descending", False)),
        "candidate_center_xyz": yaml_float_list(candidate["center"]),
        "roi_radius_m": float(candidate["roi_radius"]),
        "score": float(candidate["assigned_score"]),
        "top1_board_id": candidate["top1"]["board_id"] if candidate["top1"] else "",
        "top1_score": float(candidate["top1"]["score"]) if candidate["top1"] else 0.0,
        "second_score": float(candidate["second_score"]),
        "margin": float(candidate["margin"]),
        "needs_review": bool(candidate["needs_review"]),
        "reason": candidate["reason"],
        "reflectivity_threshold": int(config["reflectivity_threshold"]),
        "cluster_tolerance_m": float(config["cluster_tolerance_m"]),
        "discovery_sec": float(bag_info["discovery_sec"]),
        "accumulation_sec": float(bag_info["accumulation_sec"]),
        "accumulate_high_only": bool(bag_info.get("accumulate_high_only", True)),
        "min_hit_frames": int(bag_info.get("min_hit_frames", 0)),
        "max_accumulated_high_points": int(bag_info.get("max_accumulated_high_points", 0)),
        "drop_bad_geometry": bool(bag_info.get("drop_bad_geometry", False)),
        "accept_score_threshold": float(bag_info.get("accept_score_threshold", 0.0)),
        "accept_margin_threshold": float(bag_info.get("accept_margin_threshold", 0.0)),
        "accept_plane_rms_m": float(bag_info.get("accept_plane_rms_m", 0.0)),
        "accept_geometry_suspect": bool(bag_info.get("accept_geometry_suspect", False)),
        "geometry_priors": bag_info.get("geometry_priors", {}),
        "roi_point_count": len(candidate["roi_points"]),
        "high_point_count": len(candidate["high_points"]),
        "roi_hit_frame_count": int(candidate.get("roi_hit_frame_count", 0)),
        "high_hit_frame_count": int(candidate.get("high_hit_frame_count", 0)),
        "max_frame_high_points": int(candidate.get("max_frame_high_points", 0)),
        "saved_roi_point_count": len(roi_points),
        "saved_high_point_count": len(high_points),
        "accumulation_filter_reasons": list(candidate.get("accumulation_filter_reasons", [])),
        "geometry_reasons": list(candidate.get("geometry_reasons", [])),
        "frame_accumulation_dir": candidate.get("frame_accumulation_dir", ""),
        "frame_accumulation_files": candidate.get("frame_accumulation_files", []),
        "per_frame_dir": candidate.get("per_frame_dir", ""),
        "per_frame_index_file": candidate.get("per_frame_index_file", ""),
        "per_frame_file_count": int(candidate.get("per_frame_file_count", 0)),
        "per_frame_nonempty_count": int(candidate.get("per_frame_nonempty_count", 0)),
        "per_frame_total_point_count": int(candidate.get("per_frame_total_point_count", 0)),
        "plane": {
            "origin_xyz": yaml_float_list(plane["origin"]) if plane else [],
            "x_axis_xyz": yaml_float_list(plane["x_axis"]) if plane else [],
            "y_axis_xyz": yaml_float_list(plane["y_axis"]) if plane else [],
            "normal_xyz": yaml_float_list(plane["normal"]) if plane else [],
            "rms_m": float(plane["rms_m"]) if plane else 0.0,
            "max_abs_w_m": float(plane["max_abs_w_m"]) if plane else 0.0,
            "match_plane_offset_m": float(plane["match_plane_offset_m"]) if plane else 0.0,
            "match_point_ratio": float(plane["match_point_ratio"]) if plane else 0.0,
            "match_point_count": len(match_local),
            "local_bounds": yaml_bounds(plane["local_bounds"]) if plane else {},
            "match_bounds": yaml_bounds(plane["match_bounds"]) if plane else {},
        },
        "rank_scores": [
            {"board_id": row["board_id"], "label": row["label"], "score": float(row["score"])}
            for row in candidate["scores"]
        ],
        "points_xyz": [[float(p[0]), float(p[1]), float(p[2])] for p in roi_points],
        "high_points_xyz": [[float(p[0]), float(p[1]), float(p[2])] for p in high_points],
        "points_local_uvw": [[float(p[0]), float(p[1]), float(p[2]), int(p[3])] for p in local_sample],
        "points_local_uv": [[float(p[0]), float(p[1]), int(p[3])] for p in local_sample],
        "match_points_local_uvw": [[float(p[0]), float(p[1]), float(p[2]), int(p[3])] for p in match_local_sample],
        "match_points_local_uv": [[float(p[0]), float(p[1]), int(p[3])] for p in match_local_sample],
    }
    with open(path, "w", encoding="utf-8") as handle:
        dump_yaml(data, handle)


def discover_candidates(bag_path, topic, config, discovery_sec):
    high_points = []
    frame_id = ""
    start_time = None
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            stamp = msg.header.stamp.to_sec() if getattr(msg, "header", None) else t.to_sec()
            if start_time is None:
                start_time = stamp
            if stamp - start_time > discovery_sec:
                break
            frame_id = getattr(msg.header, "frame_id", frame_id) if getattr(msg, "header", None) else frame_id
            points = msg_points_array(msg, config, high_only=True)
            if len(points):
                high_points.extend(
                    (float(row[0]), float(row[1]), float(row[2]),
                     int(row[3]), float(row[4]))
                    for row in points
                )
    clusters = cluster_points(
        high_points,
        float(config["cluster_tolerance_m"]),
        int(config["min_cluster_points"]),
        int(config["max_cluster_points"]),
    )
    return high_points, clusters, frame_id, start_time


def build_rois(discovery_points, clusters, config, max_candidates):
    rois = []
    selected_clusters = clusters if max_candidates <= 0 else clusters[:max_candidates]
    for candidate_id, cluster in enumerate(selected_clusters):
        center = cluster["center"]
        radius = float(config["min_roi_radius_m"])
        for idx in cluster["indices"]:
            p = discovery_points[idx]
            radius = max(radius, math.sqrt(float(np.sum((np.asarray(p[:3]) - center) ** 2))) + float(config["roi_padding_m"]))
        radius = clamp(radius, float(config["min_roi_radius_m"]), float(config["max_roi_radius_m"]))
        rois.append({
            "candidate_id": candidate_id,
            "center": center,
            "center_tuple": (float(center[0]), float(center[1]), float(center[2])),
            "roi_radius": radius,
            "roi_radius_sq": radius * radius,
            "roi_points": [],
            "high_points": [],
            "high_points_by_frame": [],
            "per_frame_points": [],
            "roi_hit_frame_count": 0,
            "high_hit_frame_count": 0,
            "max_frame_high_points": 0,
            "accumulation_filter_reasons": [],
            "geometry_reasons": [],
        })
    return rois


def left_right_axis_index(axis):
    return {"x": 0, "y": 1, "z": 2}[axis]


def arrange_rois_left_to_right(rois, axis, descending):
    axis_index = left_right_axis_index(axis)
    ordered = sorted(
        rois,
        key=lambda roi: (float(roi["center"][axis_index]), -int(roi["candidate_id"])),
        reverse=descending,
    )
    for index, roi in enumerate(ordered):
        roi["candidate_id"] = index
        roi["left_right_index"] = index + 1
    return ordered


def known_order_for_bag(bag_index, args):
    ids = list(args.known_order_ids or [])
    if not ids:
        count = max(1, int(args.known_order_count))
        ids = list(range(1, count + 1))
    start = int(args.known_order_start)
    if start <= 0:
        if int(bag_index) in ids:
            start = int(bag_index)
        else:
            start = ids[(int(bag_index) - 1) % len(ids)]
    if start in ids:
        start_index = ids.index(start)
    else:
        start_index = 0
    return ids[start_index:] + ids[:start_index]


def attach_known_order_prior(rois, bag_index, args):
    if not args.known_order_cycle:
        return []
    ordered_ids = known_order_for_bag(bag_index, args)
    for index, roi in enumerate(rois):
        if index >= len(ordered_ids):
            roi["known_order_board_id"] = ""
            roi["known_order_label"] = ""
            continue
        board_id = str(ordered_ids[index])
        roi["known_order_board_id"] = board_id
        roi["known_order_label"] = f"board_{int(board_id):02d}"
    return ordered_ids


def accumulate_rois(bag_path, topic, rois, config, accumulation_sec, start_time,
                    high_only=True, frame_accumulation_count=0,
                    save_per_frame_points=False, per_frame_count=0):
    if start_time is None or not rois:
        return
    last_log_sec = -1
    frames_seen = 0
    valid_points_seen = 0
    roi_hits = 0
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            stamp = msg.header.stamp.to_sec() if getattr(msg, "header", None) else t.to_sec()
            elapsed = stamp - start_time
            if elapsed > accumulation_sec:
                break
            frames_seen += 1
            keep_frame_points = (
                frame_accumulation_count > 0 and
                frames_seen <= frame_accumulation_count
            )
            keep_per_frame_points = (
                save_per_frame_points and
                (per_frame_count <= 0 or frames_seen <= per_frame_count)
            )
            if keep_frame_points or keep_per_frame_points:
                for roi in rois:
                    if keep_frame_points:
                        roi["high_points_by_frame"].append([])
                    if keep_per_frame_points:
                        roi["per_frame_points"].append({
                            "frame_index": frames_seen,
                            "stamp_sec": float(stamp),
                            "elapsed_sec": float(elapsed),
                            "points": [],
                        })
            points = msg_points_array(msg, config, high_only=high_only)
            if len(points):
                valid_points_seen += len(points)
                xyz = points[:, :3]
                for roi in rois:
                    center = roi["center"]
                    radius = roi["roi_radius"]
                    diff = xyz - center
                    box_mask = (
                        (diff[:, 0] >= -radius) & (diff[:, 0] <= radius) &
                        (diff[:, 1] >= -radius) & (diff[:, 1] <= radius) &
                        (diff[:, 2] >= -radius) & (diff[:, 2] <= radius)
                    )
                    if not np.any(box_mask):
                        continue
                    boxed = diff[box_mask]
                    local_mask = np.sum(boxed * boxed, axis=1) <= roi["roi_radius_sq"]
                    if not np.any(local_mask):
                        continue
                    selected = points[box_mask][local_mask]
                    selected_points = [
                        (float(row[0]), float(row[1]), float(row[2]),
                         int(row[3]), float(row[4]))
                        for row in selected
                    ]
                    roi["roi_points"].extend(selected_points)
                    roi_hits += len(selected_points)
                    if selected_points:
                        roi["roi_hit_frame_count"] += 1
                    if high_only:
                        roi["high_points"].extend(selected_points)
                        high_selected_points = selected_points
                    else:
                        high_selected_points = [
                            point for point in selected_points
                            if point[3] >= config["reflectivity_threshold"]
                        ]
                        roi["high_points"].extend(high_selected_points)
                    if high_selected_points:
                        roi["high_hit_frame_count"] += 1
                        roi["max_frame_high_points"] = max(
                            int(roi["max_frame_high_points"]),
                            len(high_selected_points))
                    if keep_frame_points:
                        roi["high_points_by_frame"][-1].extend(high_selected_points)
                    if keep_per_frame_points:
                        roi["per_frame_points"][-1]["points"].extend(high_selected_points)
            log_sec = int(elapsed)
            if log_sec >= 0 and log_sec % 2 == 0 and log_sec != last_log_sec:
                last_log_sec = log_sec
                high_hits = sum(len(roi["high_points"]) for roi in rois)
                print(
                    f"  accumulate {elapsed:.1f}/{accumulation_sec:.1f}s "
                    f"frames={frames_seen} valid_points={valid_points_seen} "
                    f"roi_hits={roi_hits} high_hits={high_hits}",
                    flush=True,
                )


def geometry_review_reasons(roi, args):
    reasons = []
    plane = roi.get("plane")
    if plane is None:
        add_reason(reasons, "plane_failed")
        return reasons

    match_bounds = plane.get("match_bounds", {})
    local_bounds = plane.get("local_bounds", {})
    match_count = int(match_bounds.get("count", 0))
    match_ratio = float(plane.get("match_point_ratio", 0.0))
    local_long = float(local_bounds.get("long_span", 0.0))
    local_short = float(local_bounds.get("short_span", 0.0))
    match_long = float(match_bounds.get("long_span", 0.0))
    match_short = float(match_bounds.get("short_span", 0.0))

    if args.min_geometry_points > 0 and match_count < args.min_geometry_points:
        add_reason(reasons, "few_plane_inliers")
    if args.min_plane_inlier_ratio > 0.0 and match_ratio < args.min_plane_inlier_ratio:
        add_reason(reasons, "low_plane_inlier_ratio")

    if args.max_board_span_m > 0.0 and local_long > args.max_board_span_m:
        add_reason(reasons, "board_span_too_large")
    if args.min_board_span_m > 0.0 and local_long < args.min_board_span_m:
        add_reason(reasons, "board_span_too_small")
    if args.max_board_short_span_m > 0.0 and local_short > args.max_board_short_span_m:
        add_reason(reasons, "board_short_span_too_large")

    if args.reflector_min_long_m > 0.0 and match_long < args.reflector_min_long_m:
        add_reason(reasons, "reflector_long_span_too_small")
    if args.reflector_max_long_m > 0.0 and match_long > args.reflector_max_long_m:
        add_reason(reasons, "reflector_long_span_too_large")
    if args.reflector_min_short_m > 0.0 and match_short < args.reflector_min_short_m:
        add_reason(reasons, "reflector_short_span_too_small")
    if args.reflector_max_short_m > 0.0 and match_short > args.reflector_max_short_m:
        add_reason(reasons, "reflector_short_span_too_large")
    return reasons


def candidate_review_reasons(roi, args):
    reasons = []
    if roi.get("assigned_score", 0.0) < args.review_score_threshold:
        add_reason(reasons, "low_score")
    if roi.get("margin", 0.0) < args.review_margin_threshold:
        add_reason(reasons, "low_margin")
    plane = roi.get("plane")
    if plane and plane.get("rms_m", 0.0) > args.review_plane_rms_m:
        add_reason(reasons, "high_plane_rms")
    for reason in roi.get("geometry_reasons", []):
        add_reason(reasons, reason)
    return reasons


def update_review_status(roi, args):
    reasons = candidate_review_reasons(roi, args)
    roi["needs_review"] = bool(reasons)
    roi["reason"] = ",".join(reasons) if reasons else "ok"


def candidate_accept_reasons(roi, args):
    reasons = []
    if not roi.get("top1"):
        add_reason(reasons, "no_template_scores")
        return reasons
    top1_score = float(roi["top1"].get("score", 0.0))
    if top1_score < args.accept_score_threshold:
        add_reason(reasons, "reject_low_score")
    if float(roi.get("margin", 0.0)) < args.accept_margin_threshold:
        add_reason(reasons, "reject_low_margin")
    plane = roi.get("plane")
    if plane and plane.get("rms_m", 0.0) > args.accept_plane_rms_m:
        add_reason(reasons, "reject_high_plane_rms")
    if not args.accept_geometry_suspect:
        for reason in roi.get("geometry_reasons", []):
            add_reason(reasons, "reject_" + reason)
    return reasons


def assign_unconfirmed_label(roi, reasons):
    roi["assigned_board_id"] = f"unassigned_c{roi['candidate_id']:02d}"
    roi["assigned_label"] = "unassigned"
    roi["assigned_score"] = float(roi["top1"]["score"]) if roi.get("top1") else 0.0
    roi["needs_review"] = True
    roi["reason"] = ",".join(reasons) if reasons else "not_auto_accepted"


def score_candidates(rois, templates, config, args):
    print(f"  scoring {len(rois)} candidate(s) against {len(templates)} template(s)", flush=True)
    match_plane_offset = (
        float(args.match_plane_offset_m)
        if args.match_plane_offset_m is not None
        else default_match_plane_offset(config)
    )
    for index, roi in enumerate(rois, start=1):
        if index == 1 or index == len(rois) or index % 5 == 0:
            print(
                f"    scoring candidate {index}/{len(rois)} "
                f"high_points={len(roi['high_points'])}",
                flush=True,
            )
        plane = estimate_plane(
            roi["high_points"], int(config["min_template_points"]),
            match_plane_offset)
        roi["plane"] = plane
        if plane is None:
            roi["scores"] = []
            roi["top1"] = None
            roi["second_score"] = 0.0
            roi["margin"] = 0.0
            roi["geometry_reasons"] = ["plane_failed"]
            roi["reason"] = "insufficient_points_or_plane_failed"
            continue
        roi["geometry_reasons"] = geometry_review_reasons(roi, args)
        scores = rank_templates(plane["match_local_uv"], templates, config)
        roi["scores"] = scores
        roi["top1"] = scores[0] if scores else None
        roi["second_score"] = scores[1]["score"] if len(scores) > 1 else 0.0
        roi["margin"] = (roi["top1"]["score"] - roi["second_score"]) if roi["top1"] else 0.0
        roi["reason"] = "ok" if roi["top1"] else "insufficient_plane_filtered_points"


def assign_candidate_labels(rois, templates, args):
    valid = [roi for roi in rois if roi.get("scores")]
    assignment = assign_unique(valid, templates) if args.unique_assignment else {}
    for roi in rois:
        if args.unique_assignment and roi["candidate_id"] in assignment:
            board_id, label, score = assignment[roi["candidate_id"]]
            roi["assigned_board_id"] = board_id
            roi["assigned_label"] = label
            roi["assigned_score"] = score
            reject_reasons = candidate_accept_reasons(roi, args)
            if reject_reasons:
                assign_unconfirmed_label(roi, reject_reasons)
            else:
                update_review_status(roi, args)
        elif args.unique_assignment and roi.get("top1"):
            roi["assigned_board_id"] = roi["top1"]["board_id"]
            roi["assigned_label"] = roi["top1"]["label"]
            roi["assigned_score"] = roi["top1"]["score"]
            assign_unconfirmed_label(roi, ["not_unique_assigned"])
        elif roi.get("top1"):
            roi["assigned_board_id"] = roi["top1"]["board_id"]
            roi["assigned_label"] = roi["top1"]["label"]
            roi["assigned_score"] = roi["top1"]["score"]
            reject_reasons = candidate_accept_reasons(roi, args)
            if reject_reasons:
                assign_unconfirmed_label(roi, reject_reasons)
            else:
                update_review_status(roi, args)
        else:
            assign_unconfirmed_label(roi, ["no_template_scores"])


def assign_known_order_labels(rois):
    for roi in rois:
        board_id = roi.get("known_order_board_id", "")
        if not board_id:
            assign_unconfirmed_label(roi, ["outside_known_order_count"])
            continue
        roi["assigned_board_id"] = str(board_id)
        roi["assigned_label"] = roi.get("known_order_label", f"board_{int(board_id):02d}")
        roi["assigned_score"] = 1.0
        roi["needs_review"] = bool(roi.get("geometry_reasons", []))
        reasons = ["known_order_prior"]
        if roi.get("top1"):
            top1_board_id = str(roi["top1"]["board_id"])
            if top1_board_id == str(board_id):
                add_reason(reasons, "template_top1_agrees")
            else:
                add_reason(reasons, f"template_top1_is_{top1_board_id}")
                roi["needs_review"] = True
        else:
            add_reason(reasons, "no_template_scores")
            roi["needs_review"] = True
        if roi.get("geometry_reasons"):
            add_reason(reasons, "geometry_review")
        roi["reason"] = ",".join(reasons)


def assignment_mode(args):
    if args.known_order_cycle:
        return "known_order_cycle"
    return "unique" if args.unique_assignment else "top1_per_candidate"


def output_candidate(image_dir, pointcloud_dir, bag_info, candidate, config, args, used_names):
    raw_name = str(candidate["assigned_board_id"])
    if raw_name.isdigit():
        base = str(int(raw_name))
    else:
        base = sanitize_token(raw_name)
    name = base
    if name in used_names:
        name = f"{base}_c{candidate['candidate_id']:02d}"
        candidate["needs_review"] = True
    used_names.add(name)

    pointcloud_file = f"{name}.yaml"
    candidate["pointcloud_file"] = pointcloud_file
    if candidate.get("plane") is not None:
        image_file = f"{name}.bmp"
        candidate["image_file"] = image_file
        uv_points = candidate["plane"].get(
            "match_local_uv", candidate["plane"]["local_uv"])
        uv_points = sample_points(uv_points, args.max_image_points)
        write_bmp(image_dir / image_file,
                  uv_points,
                  args.image_size,
                  args.image_padding,
                  args.point_radius)
    else:
        candidate["image_file"] = ""
    write_frame_accumulations(pointcloud_dir, name, candidate, args)
    write_per_frame_points(pointcloud_dir, name, bag_info, candidate, args)
    write_pointcloud_yaml(pointcloud_dir / pointcloud_file,
                          bag_info,
                          candidate,
                          config,
                          args.max_saved_roi_points,
                          args.max_saved_high_points)


def reason_counts(rois, key):
    counts = {}
    for roi in rois:
        for reason in roi.get(key, []):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def accumulation_filter_reasons(roi, args):
    reasons = []
    high_points = len(roi["high_points"])
    high_hit_frames = int(roi.get("high_hit_frame_count", 0))
    if high_points < args.min_accumulated_high_points:
        add_reason(reasons, "few_accumulated_high_points")
    if args.min_hit_frames > 0 and high_hit_frames < args.min_hit_frames:
        add_reason(reasons, "few_hit_frames")
    if args.max_accumulated_high_points > 0 and high_points > args.max_accumulated_high_points:
        add_reason(reasons, "too_many_accumulated_high_points")
    return reasons


def filter_accumulated_rois(rois, args, bag_name):
    kept = []
    dropped = []
    for roi in rois:
        roi["accumulation_filter_reasons"] = accumulation_filter_reasons(roi, args)
        if roi["accumulation_filter_reasons"]:
            dropped.append(roi)
        else:
            kept.append(roi)
    if dropped:
        print(
            f"[{bag_name}] filtered candidates by accumulation priors: "
            f"{len(rois)} -> {len(kept)} reasons={reason_counts(dropped, 'accumulation_filter_reasons')}",
            flush=True,
        )
    return kept


def filter_bad_geometry_rois(rois, args, bag_name):
    if not args.drop_bad_geometry:
        return rois
    kept = []
    dropped = []
    for roi in rois:
        if roi.get("geometry_reasons"):
            dropped.append(roi)
        else:
            kept.append(roi)
    if dropped:
        print(
            f"[{bag_name}] dropped geometry-suspect candidates: "
            f"{len(rois)} -> {len(kept)} reasons={reason_counts(dropped, 'geometry_reasons')}",
            flush=True,
        )
    return kept


def geometry_priors_dict(args):
    return {
        "min_geometry_points": int(args.min_geometry_points),
        "min_plane_inlier_ratio": float(args.min_plane_inlier_ratio),
        "min_board_span_m": float(args.min_board_span_m),
        "max_board_span_m": float(args.max_board_span_m),
        "max_board_short_span_m": float(args.max_board_short_span_m),
        "reflector_min_long_m": float(args.reflector_min_long_m),
        "reflector_max_long_m": float(args.reflector_max_long_m),
        "reflector_min_short_m": float(args.reflector_min_short_m),
        "reflector_max_short_m": float(args.reflector_max_short_m),
    }


def process_bag(bag_path, bag_index, templates, config, args):
    bag_name = f"bag{bag_index}"
    print(f"[{bag_name}] processing {bag_path}")
    image_dir = Path(args.output_root) / "Image" / bag_name
    pointcloud_dir = Path(args.output_root) / "PointCloud" / bag_name
    image_dir.mkdir(parents=True, exist_ok=True)
    pointcloud_dir.mkdir(parents=True, exist_ok=True)

    discovery_points, clusters, frame_id, start_time = discover_candidates(
        bag_path, args.topic, config, args.discovery_sec)
    rois = build_rois(discovery_points, clusters, config, args.max_candidates)
    known_order_ids = []
    if args.known_order_cycle:
        rois = arrange_rois_left_to_right(
            rois, args.left_right_axis, args.left_right_descending)
        known_order_ids = attach_known_order_prior(rois, bag_index, args)
    top_cluster_sizes = [len(cluster["indices"]) for cluster in clusters[:10]]
    print(
        f"[{bag_name}] discovery_high_points={len(discovery_points)} "
        f"clusters={len(clusters)} rois={len(rois)} "
        f"top_cluster_sizes={top_cluster_sizes}",
        flush=True,
    )
    accumulate_rois(
        bag_path, args.topic, rois, config, args.accumulation_sec,
        start_time, high_only=not args.accumulate_all_roi_points,
        frame_accumulation_count=args.frame_accumulation_count,
        save_per_frame_points=args.save_per_frame_points,
        per_frame_count=args.per_frame_count)
    print(
        f"[{bag_name}] accumulation done: "
        f"roi_points={sum(len(roi['roi_points']) for roi in rois)} "
        f"high_points={sum(len(roi['high_points']) for roi in rois)} "
        f"hit_frames={sum(int(roi.get('high_hit_frame_count', 0)) for roi in rois)}",
        flush=True,
    )
    rois = filter_accumulated_rois(rois, args, bag_name)
    score_candidates(rois, templates, config, args)
    rois = filter_bad_geometry_rois(rois, args, bag_name)
    if args.known_order_cycle:
        assign_known_order_labels(rois)
    else:
        assign_candidate_labels(rois, templates, args)

    bag_info = {
        "bag_name": Path(bag_path).name,
        "bag_path": str(bag_path),
        "bag_index": bag_index,
        "topic": args.topic,
        "frame_id": frame_id,
        "discovery_sec": args.discovery_sec,
        "accumulation_sec": args.accumulation_sec,
        "accumulate_high_only": not args.accumulate_all_roi_points,
        "assignment_mode": assignment_mode(args),
        "known_order_cycle": bool(args.known_order_cycle),
        "known_order_count": int(args.known_order_count),
        "known_order_start": int(args.known_order_start),
        "known_order_ids": [int(value) for value in known_order_ids],
        "left_right_axis": args.left_right_axis,
        "left_right_descending": bool(args.left_right_descending),
        "review_score_threshold": float(args.review_score_threshold),
        "review_margin_threshold": float(args.review_margin_threshold),
        "review_plane_rms_m": float(args.review_plane_rms_m),
        "accept_score_threshold": float(args.accept_score_threshold),
        "accept_margin_threshold": float(args.accept_margin_threshold),
        "accept_plane_rms_m": float(args.accept_plane_rms_m),
        "accept_geometry_suspect": bool(args.accept_geometry_suspect),
        "min_hit_frames": int(args.min_hit_frames),
        "max_accumulated_high_points": int(args.max_accumulated_high_points),
        "drop_bad_geometry": bool(args.drop_bad_geometry),
        "geometry_priors": geometry_priors_dict(args),
        "frame_accumulation_count": int(args.frame_accumulation_count),
        "max_frame_accumulation_points": int(args.max_frame_accumulation_points),
        "save_per_frame_points": bool(args.save_per_frame_points),
        "per_frame_count": int(args.per_frame_count),
        "max_per_frame_points": int(args.max_per_frame_points),
        "skip_empty_per_frame_points": bool(args.skip_empty_per_frame_points),
        "reflectivity_threshold": int(config["reflectivity_threshold"]),
        "cluster_tolerance_m": float(config["cluster_tolerance_m"]),
    }
    used_names = set()
    for index, candidate in enumerate(rois, start=1):
        if index == 1 or index == len(rois) or index % 5 == 0:
            print(f"  writing candidate {index}/{len(rois)}", flush=True)
        output_candidate(image_dir, pointcloud_dir, bag_info, candidate, config, args, used_names)
    print(f"[{bag_name}] wrote candidate files", flush=True)

    metadata = {
        "metadata_version": 1,
        "bag_name": Path(bag_path).name,
        "bag_path": str(bag_path),
        "bag_index": bag_index,
        "input_topic": args.topic,
        "frame_id": frame_id,
        "discovery_sec": float(args.discovery_sec),
        "accumulation_sec": float(args.accumulation_sec),
        "accumulate_high_only": not args.accumulate_all_roi_points,
        "assignment_mode": assignment_mode(args),
        "known_order_cycle": bool(args.known_order_cycle),
        "known_order_count": int(args.known_order_count),
        "known_order_start": int(args.known_order_start),
        "known_order_ids": [int(value) for value in known_order_ids],
        "left_right_axis": args.left_right_axis,
        "left_right_descending": bool(args.left_right_descending),
        "review_score_threshold": float(args.review_score_threshold),
        "review_margin_threshold": float(args.review_margin_threshold),
        "review_plane_rms_m": float(args.review_plane_rms_m),
        "accept_score_threshold": float(args.accept_score_threshold),
        "accept_margin_threshold": float(args.accept_margin_threshold),
        "accept_plane_rms_m": float(args.accept_plane_rms_m),
        "accept_geometry_suspect": bool(args.accept_geometry_suspect),
        "min_hit_frames": int(args.min_hit_frames),
        "max_accumulated_high_points": int(args.max_accumulated_high_points),
        "drop_bad_geometry": bool(args.drop_bad_geometry),
        "geometry_priors": geometry_priors_dict(args),
        "frame_accumulation_count": int(args.frame_accumulation_count),
        "max_frame_accumulation_points": int(args.max_frame_accumulation_points),
        "save_per_frame_points": bool(args.save_per_frame_points),
        "per_frame_count": int(args.per_frame_count),
        "max_per_frame_points": int(args.max_per_frame_points),
        "skip_empty_per_frame_points": bool(args.skip_empty_per_frame_points),
        "reflectivity_threshold": int(config["reflectivity_threshold"]),
        "cluster_tolerance_m": float(config["cluster_tolerance_m"]),
        "candidate_count": len(rois),
        "discovery_high_point_count": len(discovery_points),
        "cluster_count": len(clusters),
        "template_count": len(templates),
        "templates": [{"board_id": t["board_id"], "label": t["label"], "path": t["path"]} for t in templates],
        "candidates": [],
    }
    for candidate in rois:
        metadata["candidates"].append({
            "candidate_id": candidate["candidate_id"],
            "predicted_board_id": candidate["assigned_board_id"],
            "predicted_label": candidate["assigned_label"],
            "score": float(candidate["assigned_score"]),
            "top1_board_id": candidate["top1"]["board_id"] if candidate["top1"] else "",
            "top1_score": float(candidate["top1"]["score"]) if candidate["top1"] else 0.0,
            "second_score": float(candidate["second_score"]),
            "margin": float(candidate["margin"]),
            "needs_review": bool(candidate["needs_review"]),
            "reason": candidate["reason"],
            "known_order_board_id": candidate.get("known_order_board_id", ""),
            "left_right_index": int(candidate.get("left_right_index", candidate["candidate_id"] + 1)),
            "center_xyz": yaml_float_list(candidate["center"]),
            "roi_radius_m": float(candidate["roi_radius"]),
            "roi_point_count": len(candidate["roi_points"]),
            "high_point_count": len(candidate["high_points"]),
            "roi_hit_frame_count": int(candidate.get("roi_hit_frame_count", 0)),
            "high_hit_frame_count": int(candidate.get("high_hit_frame_count", 0)),
            "max_frame_high_points": int(candidate.get("max_frame_high_points", 0)),
            "match_point_count": len(candidate["plane"]["match_local"]) if candidate.get("plane") else 0,
            "match_plane_offset_m": float(candidate["plane"]["match_plane_offset_m"]) if candidate.get("plane") else 0.0,
            "match_point_ratio": float(candidate["plane"]["match_point_ratio"]) if candidate.get("plane") else 0.0,
            "plane_rms_m": float(candidate["plane"]["rms_m"]) if candidate.get("plane") else 0.0,
            "local_bounds": yaml_bounds(candidate["plane"]["local_bounds"]) if candidate.get("plane") else {},
            "match_bounds": yaml_bounds(candidate["plane"]["match_bounds"]) if candidate.get("plane") else {},
            "geometry_reasons": list(candidate.get("geometry_reasons", [])),
            "accumulation_filter_reasons": list(candidate.get("accumulation_filter_reasons", [])),
            "image_file": candidate.get("image_file", ""),
            "pointcloud_file": candidate.get("pointcloud_file", ""),
            "frame_accumulation_dir": candidate.get("frame_accumulation_dir", ""),
            "frame_accumulation_files": candidate.get("frame_accumulation_files", []),
            "per_frame_dir": candidate.get("per_frame_dir", ""),
            "per_frame_index_file": candidate.get("per_frame_index_file", ""),
            "per_frame_file_count": int(candidate.get("per_frame_file_count", 0)),
            "per_frame_nonempty_count": int(candidate.get("per_frame_nonempty_count", 0)),
            "per_frame_total_point_count": int(candidate.get("per_frame_total_point_count", 0)),
            "rank_scores": [
                {"board_id": row["board_id"], "score": float(row["score"])}
                for row in candidate.get("scores", [])
            ],
        })
    with open(image_dir / "metadata.yaml", "w", encoding="utf-8") as handle:
        dump_yaml(metadata, handle)
    print(f"[{bag_name}] candidates={len(rois)} clusters={len(clusters)} output={image_dir.parent.parent}")


def collect_bags(args):
    bags = []
    if args.bag_number is not None:
        if not args.bag_dir:
            raise RuntimeError("--bag-number requires --bag-dir")
        bag = Path(args.bag_dir) / f"{args.bag_number}.bag"
        if not bag.exists():
            raise RuntimeError(f"bag file not found: {bag}")
        return [bag]
    if args.bag_range:
        if not args.bag_dir:
            raise RuntimeError("--bag-range requires --bag-dir")
        start, end = args.bag_range
        if start > end:
            raise RuntimeError("--bag-range START must be <= END")
        for index in range(start, end + 1):
            bag = Path(args.bag_dir) / f"{index}.bag"
            if not bag.exists():
                raise RuntimeError(f"bag file not found: {bag}")
            bags.append(bag)
        return bags
    if args.bags:
        bags.extend(Path(path) for path in args.bags)
    if args.bag_dir:
        bags.extend(sorted(Path(args.bag_dir).glob("*.bag"), key=natural_key))
    seen = set()
    unique = []
    for bag in bags:
        key = str(bag.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(bag)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label 15 reflective-board candidates from 20s rosbags.")
    parser.add_argument("--bag-dir", default="", help="Directory containing .bag files.")
    parser.add_argument("--bags", nargs="*", help="Explicit bag files, processed in the given order.")
    parser.add_argument("--bag-number", type=int,
                        help="Process BAG_DIR/<N>.bag and write output to bag<N>.")
    parser.add_argument("--bag-range", nargs=2, type=int, metavar=("START", "END"),
                        help="Process BAG_DIR/START.bag through BAG_DIR/END.bag.")
    parser.add_argument("--topic", default="/livox/lidar")
    parser.add_argument("--template-dir",
                        default="src/livox_reflective_marker/config/board_templates")
    parser.add_argument("--params",
                        default="",
                        help="Optional YAML override for offline labeling parameters.")
    parser.add_argument("--output-root", default="Data")
    parser.add_argument("--discovery-sec", type=float, default=3.0)
    parser.add_argument("--accumulation-sec", type=float, default=20.0)
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="Maximum discovered clusters to keep; 0 means keep all.")
    parser.add_argument("--max-saved-roi-points", type=int, default=3000)
    parser.add_argument("--max-saved-high-points", type=int, default=3000)
    parser.add_argument("--frame-accumulation-count", type=int, default=30,
                        help="Write cumulative high-point clouds for the first N frames; 0 disables this output.")
    parser.add_argument("--max-frame-accumulation-points", type=int, default=0,
                        help="Maximum points saved in each cumulative frame file; 0 saves all points.")
    parser.add_argument("--save-per-frame-points", action="store_true",
                        help="Write each candidate's high-reflectivity ROI points for every captured frame.")
    parser.add_argument("--per-frame-count", type=int, default=0,
                        help="Number of frames to write in per-frame mode; 0 means all frames within accumulation-sec.")
    parser.add_argument("--max-per-frame-points", type=int, default=0,
                        help="Maximum points saved in each per-frame file; 0 saves all points.")
    parser.add_argument("--skip-empty-per-frame-points", action="store_true",
                        help="Do not write per-frame files for frames with zero points in that candidate ROI.")
    parser.add_argument("--min-accumulated-high-points", type=int, default=20,
                        help="Drop candidates with fewer accumulated high-reflectivity points before scoring/output.")
    parser.add_argument("--max-accumulated-high-points", type=int, default=0,
                        help="Drop candidates with more accumulated high-reflectivity points; 0 disables this guard.")
    parser.add_argument("--min-hit-frames", type=int, default=5,
                        help="Drop candidates hit by high-reflectivity ROI points in fewer than this many frames.")
    parser.add_argument("--accumulate-all-roi-points", action="store_true",
                        help="Save all ROI points. Default saves/checks only high-reflectivity points for speed.")
    parser.add_argument("--unique-assignment", action="store_true",
                        help="Force one candidate per board id. Default keeps each candidate's own top-1 match.")
    parser.add_argument("--known-order-cycle", action="store_true",
                        help="Assign labels from a cyclic left-to-right board order instead of template top-1.")
    parser.add_argument("--known-order-count", type=int, default=5,
                        help="Number of board ids in the known cyclic order.")
    parser.add_argument("--known-order-ids", nargs="+", type=int,
                        help="Explicit cyclic board id list, for example: 6 7 8 9 10 11 12 13 14 15.")
    parser.add_argument("--known-order-start", type=int, default=0,
                        help="First leftmost board id; 0 uses the numeric bag index, wrapped by known-order-count.")
    parser.add_argument("--left-right-axis", choices=("x", "y", "z"), default="y",
                        help="Coordinate axis used to sort candidates from left to right in known-order mode.")
    parser.add_argument("--left-right-descending", dest="left_right_descending",
                        action="store_true", default=True,
                        help="Sort larger coordinate values first when applying the left-to-right prior.")
    parser.add_argument("--left-right-ascending", dest="left_right_descending",
                        action="store_false",
                        help="Sort smaller coordinate values first when applying the left-to-right prior.")
    parser.add_argument("--match-plane-offset-m", type=float,
                        help="Override plane-distance filter used before template matching.")
    parser.add_argument("--drop-bad-geometry", action="store_true",
                        help="Drop candidates that violate board/reflector geometry priors after plane fitting.")
    parser.add_argument("--min-geometry-points", type=int, default=20,
                        help="Mark candidates with fewer plane-filtered match points as needing review.")
    parser.add_argument("--min-plane-inlier-ratio", type=float, default=0.50,
                        help="Mark candidates whose plane-filtered point ratio is below this value.")
    parser.add_argument("--min-board-span-m", type=float, default=0.08,
                        help="Mark candidates whose fitted high-reflectivity span is smaller than this value.")
    parser.add_argument("--max-board-span-m", type=float, default=0.45,
                        help="Mark candidates whose fitted high-reflectivity span exceeds the 40cm board prior.")
    parser.add_argument("--max-board-short-span-m", type=float, default=0.45,
                        help="Mark candidates whose shorter fitted high-reflectivity span exceeds this value.")
    parser.add_argument("--reflector-min-long-m", type=float, default=0.10,
                        help="Mark candidates whose in-plane reflector long span is below this value.")
    parser.add_argument("--reflector-max-long-m", type=float, default=0.36,
                        help="Mark candidates whose in-plane reflector long span is above this value.")
    parser.add_argument("--reflector-min-short-m", type=float, default=0.06,
                        help="Mark candidates whose in-plane reflector short span is below this value.")
    parser.add_argument("--reflector-max-short-m", type=float, default=0.32,
                        help="Mark candidates whose in-plane reflector short span is above this value.")
    parser.add_argument("--review-score-threshold", type=float, default=0.45,
                        help="Mark candidates below this top-1 score as needing review.")
    parser.add_argument("--review-margin-threshold", type=float, default=0.08,
                        help="Mark candidates below this top1-second margin as needing review.")
    parser.add_argument("--review-plane-rms-m", type=float, default=0.02,
                        help="Mark candidates above this fitted-plane RMS as needing review.")
    parser.add_argument("--accept-score-threshold", type=float, default=0.55,
                        help="Only write a board id label when top-1 score is at least this value.")
    parser.add_argument("--accept-margin-threshold", type=float, default=0.10,
                        help="Only write a board id label when top1-second margin is at least this value.")
    parser.add_argument("--accept-plane-rms-m", type=float, default=0.02,
                        help="Only write a board id label when fitted-plane RMS is no larger than this value.")
    parser.add_argument("--accept-geometry-suspect", action="store_true",
                        help="Allow geometry-suspect candidates to keep their top-1 board id label.")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--image-padding", type=int, default=48)
    parser.add_argument("--max-image-points", type=int, default=3000,
                        help="Maximum UV points rendered into each BMP preview; 0 uses all points.")
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--reflectivity-threshold", type=int,
                        help="Override the labeling config's reflectivity_threshold.")
    parser.add_argument("--cluster-tolerance-m", type=float,
                        help="Override discovery cluster tolerance.")
    parser.add_argument("--min-cluster-points", type=int,
                        help="Override minimum high-reflectivity points per discovery cluster.")
    args = parser.parse_args()

    if rosbag is None:
        raise RuntimeError(
            "failed to import rosbag; please run inside the sourced ROS environment"
        ) from ROSBAG_IMPORT_ERROR

    config = load_params(args.params)
    if args.reflectivity_threshold is not None:
        config["reflectivity_threshold"] = int(args.reflectivity_threshold)
    if args.cluster_tolerance_m is not None:
        config["cluster_tolerance_m"] = float(args.cluster_tolerance_m)
    if args.min_cluster_points is not None:
        config["min_cluster_points"] = int(args.min_cluster_points)
    templates = load_templates(args.template_dir, config)
    if not templates:
        raise RuntimeError(f"no usable board templates found in {args.template_dir}")
    bags = collect_bags(args)
    if not bags:
        raise RuntimeError("no bag files provided; use --bag-dir or --bags")

    Path(args.output_root, "Image").mkdir(parents=True, exist_ok=True)
    Path(args.output_root, "PointCloud").mkdir(parents=True, exist_ok=True)
    print(f"templates={len(templates)} bags={len(bags)} output_root={args.output_root}")
    for fallback_index, bag in enumerate(bags, start=1):
        process_bag(bag, bag_output_index(bag, fallback_index), templates, config, args)


if __name__ == "__main__":
    main()
