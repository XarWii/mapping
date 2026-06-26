#!/usr/bin/env python3

import argparse
from pathlib import Path

import yaml


def board_label(board_id):
    return f"board_{int(board_id):02d}"


def unique_target_name(pointcloud_dir, image_dir, source_name, target_base, candidate_id):
    source_yaml = pointcloud_dir / f"{source_name}.yaml"
    source_frame_dir = pointcloud_dir / f"{source_name}_frames"
    source_image = image_dir / f"{source_name}.bmp"

    suffixes = [""]
    if candidate_id is not None:
        suffixes.append(f"_c{int(candidate_id):02d}")
    suffixes.extend(f"_manual{idx:02d}" for idx in range(1, 100))

    for suffix in suffixes:
        name = f"{target_base}{suffix}"
        targets = [
            pointcloud_dir / f"{name}.yaml",
            pointcloud_dir / f"{name}_frames",
            image_dir / f"{name}.bmp",
        ]
        sources = [source_yaml, source_frame_dir, source_image]
        ok = True
        for target, source in zip(targets, sources):
            if target.exists() and target.resolve() != source.resolve():
                ok = False
                break
        if ok:
            return name
    raise RuntimeError(f"could not find a free target name for board {target_base}")


def rename_path(src, dst, dry_run):
    if not src.exists() or src.resolve() == dst.resolve():
        return
    if dry_run:
        print(f"rename {src} -> {dst}")
        return
    src.rename(dst)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path, data, dry_run):
    if dry_run:
        print(f"write {path}")
        return
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=False, sort_keys=False, default_flow_style=False)


def update_frame_paths(data, old_name, new_name):
    old_dir = f"{old_name}_frames"
    new_dir = f"{new_name}_frames"
    if data.get("frame_accumulation_dir") == old_dir:
        data["frame_accumulation_dir"] = new_dir
    for row in data.get("frame_accumulation_files", []) or []:
        path = row.get("file", "")
        if path.startswith(old_dir + "/"):
            row["file"] = new_dir + path[len(old_dir):]


def update_pointcloud_yaml(path, old_name, new_name, new_board_id, dry_run):
    data = load_yaml(path)
    old_board_id = data.get("board_id", "")
    old_label = data.get("label", "")
    new_label = board_label(new_board_id)

    data.setdefault("auto_label", old_label)
    data.setdefault("auto_board_id", old_board_id)
    data["label_source"] = "manual"
    data["manual_label"] = new_label
    data["manual_board_id"] = str(int(new_board_id))
    data["label"] = new_label
    data["board_id"] = str(int(new_board_id))
    data["manual_correction"] = {
        "old_file_stem": old_name,
        "new_file_stem": new_name,
        "old_board_id": str(old_board_id),
        "new_board_id": str(int(new_board_id)),
    }
    update_frame_paths(data, old_name, new_name)
    save_yaml(path, data, dry_run)
    return data


def update_metadata(path, old_name, new_name, new_board_id, dry_run):
    if not path.exists():
        return
    data = load_yaml(path)
    new_label = board_label(new_board_id)
    changed = False
    for candidate in data.get("candidates", []) or []:
        if candidate.get("pointcloud_file") != f"{old_name}.yaml" and candidate.get("image_file") != f"{old_name}.bmp":
            continue
        candidate.setdefault("auto_predicted_board_id", candidate.get("predicted_board_id", ""))
        candidate.setdefault("auto_predicted_label", candidate.get("predicted_label", ""))
        candidate["label_source"] = "manual"
        candidate["manual_board_id"] = str(int(new_board_id))
        candidate["manual_label"] = new_label
        candidate["predicted_board_id"] = str(int(new_board_id))
        candidate["predicted_label"] = new_label
        candidate["needs_review"] = False
        candidate["reason"] = "manual_corrected"
        candidate["pointcloud_file"] = f"{new_name}.yaml"
        if candidate.get("image_file"):
            candidate["image_file"] = f"{new_name}.bmp"
        if candidate.get("frame_accumulation_dir") == f"{old_name}_frames":
            candidate["frame_accumulation_dir"] = f"{new_name}_frames"
        for row in candidate.get("frame_accumulation_files", []) or []:
            frame_file = row.get("file", "")
            old_prefix = f"{old_name}_frames/"
            if frame_file.startswith(old_prefix):
                row["file"] = f"{new_name}_frames/" + frame_file[len(old_prefix):]
        changed = True
        break
    if changed:
        save_yaml(path, data, dry_run)
    else:
        print(f"warning: metadata candidate not found for {old_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Correct an auto-labeled board candidate after manual review.")
    parser.add_argument("--output-root", default="Data")
    parser.add_argument("--bag-number", type=int, required=True)
    parser.add_argument("--current-name", required=True,
                        help="Current file stem, e.g. 5 or 11_c02.")
    parser.add_argument("--board-id", type=int, required=True,
                        help="Correct board id, e.g. 2.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bag_name = f"bag{args.bag_number}"
    pointcloud_dir = Path(args.output_root) / "PointCloud" / bag_name
    image_dir = Path(args.output_root) / "Image" / bag_name
    old_name = args.current_name
    source_yaml = pointcloud_dir / f"{old_name}.yaml"
    if not source_yaml.exists():
        raise RuntimeError(f"pointcloud yaml not found: {source_yaml}")

    source_data = load_yaml(source_yaml)
    candidate_id = source_data.get("candidate_index")
    new_base = str(int(args.board_id))
    new_name = unique_target_name(pointcloud_dir, image_dir, old_name, new_base, candidate_id)

    update_pointcloud_yaml(source_yaml, old_name, new_name, args.board_id, args.dry_run)
    update_metadata(image_dir / "metadata.yaml", old_name, new_name, args.board_id, args.dry_run)

    rename_path(pointcloud_dir / f"{old_name}_frames",
                pointcloud_dir / f"{new_name}_frames",
                args.dry_run)
    rename_path(source_yaml,
                pointcloud_dir / f"{new_name}.yaml",
                args.dry_run)
    rename_path(image_dir / f"{old_name}.bmp",
                image_dir / f"{new_name}.bmp",
                args.dry_run)

    print(
        f"{bag_name}: {old_name} -> {new_name}, "
        f"label={board_label(args.board_id)}"
    )


if __name__ == "__main__":
    main()
