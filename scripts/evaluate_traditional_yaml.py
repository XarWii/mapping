#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from collections import defaultdict

import numpy as np
import yaml


POINT_RE = re.compile(r"\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)")


def clamp01(value):
    return max(0.0, min(1.0, value))


def parse_vector3(text):
    match = POINT_RE.search(text)
    if not match:
        return None
    return np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))],
                    dtype=np.float64)


def parse_sample(path):
    data = {
        "path": path,
        "label": None,
        "run_id": None,
        "window_frames": None,
        "candidate_index": None,
        "candidate_center": None,
        "roi_radius_m": 0.0,
        "confirmed_target_available": False,
        "confirmed_target_center": None,
        "confirmed_target_radius": 0.0,
        "roi_point_count": 0,
        "high_point_count": 0,
        "high_points": [],
    }
    in_high_points = False
    in_confirmed_target = False
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("label:"):
                data["label"] = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("run_id:"):
                data["run_id"] = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("window_frames:"):
                data["window_frames"] = int(line.split(":", 1)[1])
            elif line.startswith("candidate_index:"):
                data["candidate_index"] = int(line.split(":", 1)[1])
            elif line.startswith("candidate_center_xyz:"):
                data["candidate_center"] = parse_vector3(line)
            elif line.startswith("roi_radius_m:"):
                data["roi_radius_m"] = float(line.split(":", 1)[1])
            elif line.startswith("confirmed_target_available:"):
                data["confirmed_target_available"] = (
                    line.split(":", 1)[1].strip().lower() == "true"
                )
            elif line.startswith("confirmed_target:"):
                in_confirmed_target = True
                continue
            elif in_confirmed_target and line.startswith("  center_xyz:"):
                data["confirmed_target_center"] = parse_vector3(line)
            elif in_confirmed_target and line.startswith("  radius_m:"):
                data["confirmed_target_radius"] = float(line.split(":", 1)[1])
            elif in_confirmed_target and line and not line.startswith(" "):
                in_confirmed_target = False
            elif line.startswith("roi_point_count:"):
                data["roi_point_count"] = int(line.split(":", 1)[1])
            elif line.startswith("high_point_count:"):
                data["high_point_count"] = int(line.split(":", 1)[1])
            elif line.startswith("high_points_xyz:"):
                in_high_points = True
                continue
            elif in_high_points:
                if line.startswith("  - ["):
                    point = parse_vector3(line)
                    if point is not None:
                        data["high_points"].append(point)
                elif line and not line.startswith(" "):
                    in_high_points = False
    if data["high_point_count"] == 0:
        data["high_point_count"] = len(data["high_points"])
    return data


def normalize_uv(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    axes = np.column_stack((vectors[:, 1], vectors[:, 0]))
    q = centered @ axes
    span_x = max(1e-4, float(q[:, 0].max() - q[:, 0].min()))
    span_y = max(1e-4, float(q[:, 1].max() - q[:, 1].min()))
    return q / max(span_x, span_y)


def build_uv_descriptor(points, cols, rows, flip_x=False, flip_y=False):
    grid = np.zeros(cols * rows, dtype=np.float64)
    if len(points) == 0:
        return grid
    for point in points:
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
    if norm > 1e-6:
        grid /= norm
    return grid


def build_descriptor_variants(points, cols, rows):
    normalized = normalize_uv(points)
    return [
        build_uv_descriptor(normalized, cols, rows, False, False),
        build_uv_descriptor(normalized, cols, rows, True, False),
        build_uv_descriptor(normalized, cols, rows, False, True),
        build_uv_descriptor(normalized, cols, rows, True, True),
    ]


def cosine_similarity(a, b):
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-6 or bn < 1e-6:
        return 0.0
    return clamp01(float(np.dot(a, b) / (an * bn)))


def load_template(path, config):
    points = []
    in_points = False
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "points_local_uv:" in line:
                in_points = True
                continue
            if not in_points:
                continue
            match = POINT_RE.search(line)
            if match:
                points.append([float(match.group(1)), float(match.group(2))])
    template = {
        "points": np.asarray(points, dtype=np.float64),
        "fine": build_descriptor_variants(points, config["template_grid_cols"],
                                          config["template_grid_rows"]),
        "mid": build_descriptor_variants(points, config["template_mid_grid_cols"],
                                         config["template_mid_grid_rows"]),
        "coarse": build_descriptor_variants(points, config["template_coarse_grid_cols"],
                                            config["template_coarse_grid_rows"]),
    }
    return template


def estimate_pose(high_points, min_points):
    points = np.asarray(high_points, dtype=np.float64)
    if len(points) < min_points:
        return None
    origin = points.mean(axis=0)
    centered = points - origin
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    z_axis = vectors[:, 0]
    z_axis /= max(1e-12, np.linalg.norm(z_axis))
    x_axis = vectors[:, 2]
    x_axis = x_axis - float(np.dot(x_axis, z_axis)) * z_axis
    if np.linalg.norm(x_axis) < 1e-4:
        return None
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-4:
        return None
    y_axis /= np.linalg.norm(y_axis)
    to_lidar = -origin
    if np.linalg.norm(to_lidar) > 1e-4 and float(np.dot(z_axis, to_lidar / np.linalg.norm(to_lidar))) < 0.0:
        z_axis = -z_axis
        y_axis = -y_axis
    return origin, x_axis, y_axis, z_axis


def template_score(high_points, pose, template, config):
    origin, x_axis, y_axis, z_axis = pose
    max_plane_offset = max(0.08, 2.0 * config["template_plane_tolerance_m"])
    uv = []
    for point in high_points:
        rel = point - origin
        w = float(np.dot(rel, z_axis))
        if abs(w) <= max_plane_offset:
            uv.append([float(np.dot(rel, x_axis)), float(np.dot(rel, y_axis))])
    if len(uv) < config["min_template_points"]:
        return 0.0, len(uv)
    cols = config["template_grid_cols"]
    rows = config["template_grid_rows"]
    variants = template["fine"]
    if len(uv) < 0.5 * config["target_template_match_points"]:
        cols = config["template_coarse_grid_cols"]
        rows = config["template_coarse_grid_rows"]
        variants = template["coarse"]
    elif len(uv) < config["target_template_match_points"]:
        cols = config["template_mid_grid_cols"]
        rows = config["template_mid_grid_rows"]
        variants = template["mid"]
    candidate_variants = build_descriptor_variants(uv, cols, rows)
    best = 0.0
    for candidate in candidate_variants:
        for templ in variants:
            best = max(best, cosine_similarity(candidate, templ))
    return best, len(uv)


def required_template_points(distance_m, config):
    if distance_m >= config["far_template_distance_m"]:
        return config["far_template_match_points"]
    t = clamp01(distance_m / max(1e-3, config["far_template_distance_m"]))
    value = ((1.0 - t) * config["target_template_match_points"] +
             t * config["far_template_match_points"])
    return int(math.floor(value + 0.5))


def required_template_score(distance_m, config):
    if distance_m >= config["far_template_distance_m"]:
        return config["far_min_template_score"]
    t = clamp01(distance_m / max(1e-3, config["far_template_distance_m"]))
    return ((1.0 - t) * config["min_template_score"] +
            t * config["far_min_template_score"])


def calibrated_score(template_score_value, min_template_score, config):
    if template_score_value < min_template_score:
        return 0.0
    normalized = ((template_score_value - min_template_score) /
                  max(1e-6, config["good_template_score"] - min_template_score))
    confidence = clamp01(normalized)
    return min(1.0, config["confirm_score_threshold"] +
               (1.0 - config["confirm_score_threshold"]) * confidence)


def score_sample(sample, template, config, respect_required_points):
    high_points = sample["high_points"]
    center = sample["candidate_center"]
    distance = float(np.linalg.norm(center)) if center is not None else 0.0
    req_points = required_template_points(distance, config)
    req_score = required_template_score(distance, config)
    result = {
        **sample,
        "distance_m": distance,
        "required_points": req_points,
        "required_template_score": req_score,
        "plane_points": 0,
        "template_score": 0.0,
        "score": 0.0,
        "valid": False,
        "reason": "not scored",
    }
    if len(high_points) < config["min_template_points"]:
        result["reason"] = "too few high-reflectivity points"
        return result
    if respect_required_points and len(high_points) < req_points:
        result["reason"] = f"waiting for template points: {len(high_points)}/{req_points}"
        return result
    pose = estimate_pose(high_points, config["min_template_points"])
    if pose is None:
        result["reason"] = "could not estimate pose"
        return result
    score, plane_points = template_score(high_points, pose, template, config)
    result["template_score"] = score
    result["plane_points"] = plane_points
    if score < req_score:
        result["reason"] = f"template score too low: {score:.6f}/{req_score:.6f}"
        return result
    result["valid"] = True
    result["score"] = calibrated_score(score, req_score, config)
    result["reason"] = "ok(template)"
    return result


def load_config(params_path):
    with open(params_path, "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    task = params["task1"]
    keys = [
        "template_grid_cols", "template_grid_rows",
        "template_mid_grid_cols", "template_mid_grid_rows",
        "template_coarse_grid_cols", "template_coarse_grid_rows",
        "min_template_points", "target_template_match_points",
        "far_template_match_points", "far_template_distance_m",
        "min_template_score", "far_min_template_score",
        "good_template_score", "confirm_score_threshold",
        "confirm_margin_threshold", "template_plane_tolerance_m",
    ]
    return {key: task[key] for key in keys}


def load_target_center_tolerance(params_path):
    with open(params_path, "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    return float(params["dataset_collector"].get("target_center_tolerance_m", 0.20))


def collect_samples(dataset_root):
    samples = []
    for label_dir in ("target", "interference"):
        root = os.path.join(dataset_root, label_dir)
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.startswith("sample_") and name.endswith(".yaml"):
                    samples.append(parse_sample(os.path.join(dirpath, name)))
    return samples


def evaluate_from_confirmed_target(samples, target_center_tolerance, output):
    by_run_window = defaultdict(list)
    for sample in samples:
        by_run_window[(sample["run_id"], sample["window_frames"])].append(sample)

    rows = []
    for (run_id, window), candidates in sorted(by_run_window.items()):
        candidates = [c for c in candidates if c["candidate_center"] is not None]
        if not candidates:
            continue
        confirmed = next(
            (c for c in candidates
             if c["confirmed_target_available"] and c["confirmed_target_center"] is not None),
            None,
        )
        if confirmed is None:
            rows.append({
                "run_id": run_id,
                "window_frames": window,
                "candidate_count": len(candidates),
                "decision": 0,
                "correct": 0,
                "predicted_label": "none",
                "predicted_candidate_index": "",
                "predicted_distance_m": "",
                "match_threshold_m": "",
                "target_distance_m": "",
                "target_candidate_index": "",
                "reason": "confirmed_target_unavailable",
            })
            continue

        center = confirmed["confirmed_target_center"]
        scored = []
        for candidate in candidates:
            distance = float(np.linalg.norm(candidate["candidate_center"] - center))
            threshold = (max(candidate["roi_radius_m"], confirmed["confirmed_target_radius"]) +
                         target_center_tolerance)
            scored.append((distance, threshold, candidate))
        scored.sort(key=lambda item: item[0])
        best_distance, best_threshold, best = scored[0]
        decision = best_distance <= best_threshold
        predicted_label = best["label"] if decision else "none"
        correct = bool(decision and predicted_label == "target")

        target_candidates = [item for item in scored if item[2]["label"] == "target"]
        target_distance = target_candidates[0][0] if target_candidates else ""
        target_index = target_candidates[0][2]["candidate_index"] if target_candidates else ""
        rows.append({
            "run_id": run_id,
            "window_frames": window,
            "candidate_count": len(candidates),
            "decision": int(decision),
            "correct": int(correct),
            "predicted_label": predicted_label,
            "predicted_candidate_index": best["candidate_index"] if decision else "",
            "predicted_distance_m": f"{best_distance:.6f}",
            "match_threshold_m": f"{best_threshold:.6f}",
            "target_distance_m": f"{target_distance:.6f}" if target_distance != "" else "",
            "target_candidate_index": target_index,
            "reason": "ok(external_confirmed_target)" if decision else "nearest_candidate_outside_threshold",
        })

    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    earliest = {}
    for run_id, run_rows in by_run.items():
        ordered = sorted(run_rows, key=lambda r: r["window_frames"])
        first_decision = next((r for r in ordered if r["decision"] == 1), None)
        first_correct = next((r for r in ordered if r["correct"] == 1), None)
        earliest[run_id] = {
            "first_decision_window": first_decision["window_frames"] if first_decision else "",
            "first_decision_correct": first_decision["correct"] if first_decision else "",
            "earliest_correct_window": first_correct["window_frames"] if first_correct else "",
            "correct_windows": sum(r["correct"] == 1 for r in ordered),
            "decision_windows": sum(r["decision"] == 1 for r in ordered),
            "tested_windows": len(ordered),
            "run_correct": int(any(r["correct"] == 1 for r in ordered)),
        }
    for row in rows:
        row.update(earliest[row["run_id"]])

    fieldnames = [
        "run_id", "window_frames", "candidate_count",
        "decision", "correct", "predicted_label", "predicted_candidate_index",
        "predicted_distance_m", "match_threshold_m",
        "target_distance_m", "target_candidate_index", "reason",
        "first_decision_window", "first_decision_correct",
        "earliest_correct_window", "correct_windows", "decision_windows",
        "tested_windows", "run_correct",
    ]
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    print(f"samples={len(samples)} run_windows={len(rows)} runs={len(by_run)}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the traditional reflective-board template matcher on saved YAML samples.")
    parser.add_argument("--dataset-root", default="test-2")
    parser.add_argument("--params", default="src/livox_reflective_marker/config/params.yaml")
    parser.add_argument("--template", default="src/livox_reflective_marker/config/target_board_template.yaml")
    parser.add_argument("--output", default="test-2/traditional_yaml_eval.csv")
    parser.add_argument("--mode", choices=["template", "confirmed-target"],
                        default="template",
                        help="template recomputes template matching; confirmed-target evaluates the traditional result recorded in YAML.")
    parser.add_argument("--ignore-required-points", action="store_true",
                        help="Score once min_template_points is reached, even before required_template_points.")
    args = parser.parse_args()

    samples = collect_samples(args.dataset_root)
    if args.mode == "confirmed-target":
        tolerance = load_target_center_tolerance(args.params)
        evaluate_from_confirmed_target(samples, tolerance, args.output)
        return

    config = load_config(args.params)
    template = load_template(args.template, config)
    respect_required_points = not args.ignore_required_points

    scored = [score_sample(sample, template, config, respect_required_points)
              for sample in samples]
    by_run_window = defaultdict(list)
    for row in scored:
        by_run_window[(row["run_id"], row["window_frames"])].append(row)

    rows = []
    for (run_id, window), candidates in sorted(by_run_window.items()):
        candidates = sorted(candidates, key=lambda r: r["score"], reverse=True)
        best = candidates[0]
        second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = best["score"] - second_score
        decision = bool(best["valid"] and
                        best["score"] >= config["confirm_score_threshold"] and
                        margin >= config["confirm_margin_threshold"])
        predicted_label = best["label"] if decision else "none"
        correct = bool(decision and predicted_label == "target")
        targets = [c for c in candidates if c["label"] == "target"]
        target = max(targets, key=lambda r: r["score"]) if targets else None
        target_rank = ""
        if target is not None:
            target_rank = 1 + candidates.index(target)
        rows.append({
            "run_id": run_id,
            "window_frames": window,
            "candidate_count": len(candidates),
            "decision": int(decision),
            "correct": int(correct),
            "predicted_label": predicted_label,
            "predicted_candidate_index": best["candidate_index"] if decision else "",
            "best_actual_label": best["label"],
            "best_score": f"{best['score']:.6f}",
            "second_score": f"{second_score:.6f}",
            "margin": f"{margin:.6f}",
            "best_template_score": f"{best['template_score']:.6f}",
            "best_high_points": len(best["high_points"]),
            "best_required_points": best["required_points"],
            "best_plane_points": best["plane_points"],
            "best_reason": best["reason"],
            "target_rank": target_rank,
            "target_score": f"{target['score']:.6f}" if target else "",
            "target_template_score": f"{target['template_score']:.6f}" if target else "",
            "target_high_points": len(target["high_points"]) if target else "",
            "target_required_points": target["required_points"] if target else "",
            "target_reason": target["reason"] if target else "",
            "mode": "strict_required_points" if respect_required_points else "ignore_required_points",
        })

    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    earliest = {}
    for run_id, run_rows in by_run.items():
        sorted_rows = sorted(run_rows, key=lambda r: r["window_frames"])
        first_decision = next((r for r in sorted_rows if r["decision"] == 1), None)
        first_correct = next((r for r in sorted_rows if r["correct"] == 1), None)
        earliest[run_id] = {
            "first_decision_window": first_decision["window_frames"] if first_decision else "",
            "first_decision_correct": first_decision["correct"] if first_decision else "",
            "earliest_correct_window": first_correct["window_frames"] if first_correct else "",
            "correct_windows": sum(r["correct"] == 1 for r in sorted_rows),
            "decision_windows": sum(r["decision"] == 1 for r in sorted_rows),
            "tested_windows": len(sorted_rows),
        }
    for row in rows:
        row.update(earliest[row["run_id"]])

    fieldnames = [
        "run_id", "window_frames", "candidate_count",
        "decision", "correct", "predicted_label", "predicted_candidate_index",
        "best_actual_label", "best_score", "second_score", "margin",
        "best_template_score", "best_high_points", "best_required_points",
        "best_plane_points", "best_reason", "target_rank", "target_score",
        "target_template_score", "target_high_points", "target_required_points",
        "target_reason", "first_decision_window", "first_decision_correct",
        "earliest_correct_window", "correct_windows", "decision_windows",
        "tested_windows", "mode",
    ]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.output}")
    print(f"samples={len(samples)} run_windows={len(rows)} runs={len(by_run)}")


if __name__ == "__main__":
    main()
