#!/usr/bin/env python3
"""Filter a PCD file by an intensity-like field.

The script keeps points whose selected field is greater than or equal to the
threshold and writes a new ASCII PCD. It supports common PCD files using
DATA ascii or uncompressed DATA binary.
"""

import argparse
import math
import struct
from pathlib import Path


FIELD_CANDIDATES = ("intensity", "reflectivity")


def split_header(raw_bytes):
    lines = []
    offset = 0
    for raw_line in raw_bytes.splitlines(keepends=True):
        offset += len(raw_line)
        line = raw_line.decode("ascii", errors="strict").rstrip("\r\n")
        lines.append(line)
        if line.strip().upper().startswith("DATA "):
            return lines, offset
    raise ValueError("PCD header has no DATA line")


def parse_header(lines):
    data_index = None
    meta = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        key = parts[0].upper()
        values = parts[1:]
        meta[key] = values
        if key == "DATA":
            data_index = i
            break
    if data_index is None:
        raise ValueError("PCD header has no DATA line")
    for key in ("FIELDS", "SIZE", "TYPE"):
        if key not in meta:
            raise ValueError(f"PCD header has no {key} line")
    fields = meta["FIELDS"]
    sizes = [int(v) for v in meta["SIZE"]]
    types = meta["TYPE"]
    counts = [int(v) for v in meta.get("COUNT", ["1"] * len(fields))]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("FIELDS/SIZE/TYPE/COUNT lengths do not match")
    data_mode = meta["DATA"][0].lower()
    if data_mode not in ("ascii", "binary"):
        raise ValueError(f"unsupported PCD DATA mode: {data_mode}")
    return lines[:data_index], meta, data_mode


def scalar_names(fields, counts):
    names = []
    for field, count in zip(fields, counts):
        if count == 1:
            names.append(field)
        else:
            names.extend(f"{field}_{i}" for i in range(count))
    return names


def find_filter_index(fields, counts, requested):
    expanded = scalar_names(fields, counts)
    if requested:
      candidates = (requested,)
    else:
      candidates = FIELD_CANDIDATES
    for name in candidates:
        if name in fields:
            offset = 0
            for field, count in zip(fields, counts):
                if field == name:
                    return offset
                offset += count
        if name in expanded:
            return expanded.index(name)
    available = ", ".join(fields)
    raise ValueError(
        f"cannot find field {candidates}; available PCD fields: {available}"
    )


def binary_struct_format(types, sizes, counts):
    fmt = "<"
    for typ, size, count in zip(types, sizes, counts):
        typ = typ.upper()
        if typ == "F" and size == 4:
            code = "f"
        elif typ == "F" and size == 8:
            code = "d"
        elif typ == "I" and size == 1:
            code = "b"
        elif typ == "I" and size == 2:
            code = "h"
        elif typ == "I" and size == 4:
            code = "i"
        elif typ == "I" and size == 8:
            code = "q"
        elif typ == "U" and size == 1:
            code = "B"
        elif typ == "U" and size == 2:
            code = "H"
        elif typ == "U" and size == 4:
            code = "I"
        elif typ == "U" and size == 8:
            code = "Q"
        else:
            raise ValueError(f"unsupported PCD scalar type={typ} size={size}")
        fmt += code * count
    return struct.Struct(fmt)


def iter_ascii_points(raw_bytes, data_start, expected_scalars):
    text = raw_bytes[data_start:].decode("ascii", errors="strict")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        values = stripped.split()
        if len(values) != expected_scalars:
            raise ValueError(
                f"ASCII PCD row has {len(values)} values, expected {expected_scalars}"
            )
        yield values


def iter_binary_points(raw_bytes, data_start, meta, fields, sizes, types, counts):
    width = int(meta.get("WIDTH", ["0"])[0])
    height = int(meta.get("HEIGHT", ["1"])[0])
    expected_points = int(meta.get("POINTS", [str(width * height)])[0])
    unpacker = binary_struct_format(types, sizes, counts)
    payload = raw_bytes[data_start:]
    required = expected_points * unpacker.size
    if len(payload) < required:
        raise ValueError(
            f"binary payload too small: {len(payload)} bytes, expected {required}"
        )
    for i in range(expected_points):
        yield unpacker.unpack_from(payload, i * unpacker.size)


def format_scalar(value):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return f"{value:.9g}"
    return str(value)


def make_output_header(meta, kept_count):
    lines = []
    version = meta.get("VERSION", [".7"])[0]
    lines.append("# .PCD v0.7 - Point Cloud Data file format")
    lines.append(f"VERSION {version}")
    lines.append("FIELDS " + " ".join(meta["FIELDS"]))
    lines.append("SIZE " + " ".join(meta["SIZE"]))
    lines.append("TYPE " + " ".join(meta["TYPE"]))
    if "COUNT" in meta:
        lines.append("COUNT " + " ".join(meta["COUNT"]))
    lines.append(f"WIDTH {kept_count}")
    lines.append("HEIGHT 1")
    viewpoint = meta.get("VIEWPOINT", ["0", "0", "0", "1", "0", "0", "0"])
    lines.append("VIEWPOINT " + " ".join(viewpoint))
    lines.append(f"POINTS {kept_count}")
    lines.append("DATA ascii")
    return "\n".join(lines) + "\n"


def filter_pcd(input_path, output_path, threshold, field_name):
    raw_bytes = Path(input_path).read_bytes()
    lines, data_start = split_header(raw_bytes)
    header_lines, meta, data_mode = parse_header(lines)
    fields = meta["FIELDS"]
    sizes = [int(v) for v in meta["SIZE"]]
    types = meta["TYPE"]
    counts = [int(v) for v in meta.get("COUNT", ["1"] * len(fields))]
    expected_scalars = sum(counts)
    filter_index = find_filter_index(fields, counts, field_name)

    def iter_points():
        if data_mode == "ascii":
            return iter_ascii_points(raw_bytes, data_start, expected_scalars)
        return iter_binary_points(raw_bytes, data_start, meta, fields, sizes, types, counts)

    total = 0
    kept_count = 0
    for row in iter_points():
        total += 1
        try:
            value = float(row[filter_index])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= threshold:
            kept_count += 1

    with Path(output_path).open("w", encoding="ascii") as out:
        out.write(make_output_header(meta, kept_count))
        for row in iter_points():
            try:
                value = float(row[filter_index])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= threshold:
                out.write(" ".join(format_scalar(value) for value in row))
                out.write("\n")
    return total, kept_count, fields[filter_index] if filter_index < len(fields) else None


def main():
    parser = argparse.ArgumentParser(
        description="Keep only PCD points whose intensity/reflectivity is above a threshold."
    )
    parser.add_argument("input_pcd", help="Input .pcd file")
    parser.add_argument("output_pcd", help="Output filtered .pcd file")
    parser.add_argument(
        "-t", "--threshold", type=float, required=True,
        help="Keep points with field value >= threshold"
    )
    parser.add_argument(
        "-f", "--field", default="",
        help="Field to threshold. Default: auto-detect intensity then reflectivity"
    )
    args = parser.parse_args()

    total, kept, _ = filter_pcd(
        args.input_pcd, args.output_pcd, args.threshold, args.field or None
    )
    ratio = kept / total if total else 0.0
    print(
        f"filtered {args.input_pcd} -> {args.output_pcd}: "
        f"kept {kept}/{total} points ({ratio:.3%}) with threshold >= {args.threshold:g}"
    )


if __name__ == "__main__":
    main()
