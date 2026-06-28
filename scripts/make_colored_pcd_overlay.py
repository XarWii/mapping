#!/usr/bin/env python3
"""Create a colored overlay PCD from two ASCII PCD clouds."""

import argparse
import struct
from pathlib import Path

from compare_pcd_maps import load_ascii_pcd


def pack_rgb_float(r, g, b):
    packed = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", packed))[0]


def write_overlay(path, clouds):
    total_points = sum(len(points) for points, _ in clouds)
    with Path(path).open("w", encoding="ascii") as output:
        output.write("# .PCD v0.7 - Point Cloud Data file format\n")
        output.write("VERSION 0.7\n")
        output.write("FIELDS x y z rgb\n")
        output.write("SIZE 4 4 4 4\n")
        output.write("TYPE F F F F\n")
        output.write("COUNT 1 1 1 1\n")
        output.write(f"WIDTH {total_points}\n")
        output.write("HEIGHT 1\n")
        output.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        output.write(f"POINTS {total_points}\n")
        output.write("DATA ascii\n")
        for points, rgb in clouds:
            for point in points:
                output.write(
                    f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {rgb:.9g}\n"
                )


def parse_color(text):
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected R,G,B")
    values = [int(part) for part in parts]
    if any(value < 0 or value > 255 for value in values):
        raise argparse.ArgumentTypeError("RGB values must be in [0, 255]")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, help="First ASCII PCD path")
    parser.add_argument("--second", required=True, help="Second ASCII PCD path")
    parser.add_argument("--output", required=True, help="Output colored overlay PCD")
    parser.add_argument("--first-color", type=parse_color, default="255,255,255")
    parser.add_argument("--second-color", type=parse_color, default="255,0,0")
    args = parser.parse_args()

    first_xyz, _, _ = load_ascii_pcd(args.first)
    second_xyz, _, _ = load_ascii_pcd(args.second)
    first_rgb = pack_rgb_float(*args.first_color)
    second_rgb = pack_rgb_float(*args.second_color)
    write_overlay(args.output, [(first_xyz, first_rgb), (second_xyz, second_rgb)])
    print(f"wrote {args.output}")
    print(f"  first:  {args.first} points={len(first_xyz)} color={args.first_color}")
    print(f"  second: {args.second} points={len(second_xyz)} color={args.second_color}")


if __name__ == "__main__":
    main()
