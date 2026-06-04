#!/usr/bin/env python3

import argparse
import csv
import math
import sys
import threading


def import_rosbag():
    try:
        import rosbag  # pylint: disable=import-error
    except ImportError as exc:
        print("Failed to import rosbag. Did you source /opt/ros/noetic/setup.bash?", file=sys.stderr)
        raise exc
    return rosbag


def stamp_to_sec(stamp):
    if stamp is None:
        return 0.0
    return float(stamp.to_sec())


def msg_time(msg, bag_time):
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and stamp_to_sec(stamp) > 0.0:
        return stamp_to_sec(stamp)
    return stamp_to_sec(bag_time)


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def roll_from_quat(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    return math.atan2(sinr_cosp, cosr_cosp)


def pitch_from_quat(q):
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        return math.copysign(math.pi / 2.0, sinp)
    return math.asin(sinp)


def unwrap_angles(values):
    if not values:
        return []
    out = [values[0]]
    offset = 0.0
    prev = values[0]
    for value in values[1:]:
        delta = value - prev
        if delta > math.pi:
            offset -= 2.0 * math.pi
        elif delta < -math.pi:
            offset += 2.0 * math.pi
        out.append(value + offset)
        prev = value
    return out


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def stddev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) * (v - m) for v in values) / (len(values) - 1))


def peak_to_peak(values):
    return max(values) - min(values) if values else 0.0


def percentile(values, pct):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def corrcoef(xs, ys):
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) * (x - mx) for x in xs)
    vy = sum((y - my) * (y - my) for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def nearest_score(scores, stamp, max_age):
    if not scores:
        return None
    best = min(scores, key=lambda item: abs(item[0] - stamp))
    if abs(best[0] - stamp) <= max_age:
        return best[1]
    return None


def read_bag(path, pose_topic, score_topic):
    rosbag = import_rosbag()
    poses = []
    scores = []
    with rosbag.Bag(path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[pose_topic, score_topic]):
            if topic == pose_topic:
                p = msg.pose.position
                q = msg.pose.orientation
                poses.append(
                    {
                        "t": msg_time(msg, t),
                        "x": float(p.x),
                        "y": float(p.y),
                        "z": float(p.z),
                        "roll": roll_from_quat(q),
                        "pitch": pitch_from_quat(q),
                        "yaw": yaw_from_quat(q),
                    }
                )
            elif topic == score_topic:
                scores.append((msg_time(msg, t), float(msg.data)))
    poses.sort(key=lambda item: item["t"])
    scores.sort(key=lambda item: item[0])
    return poses, scores


class LiveCollector:
    def __init__(self, pose_topic, score_topic):
        import rospy  # pylint: disable=import-error
        from geometry_msgs.msg import PoseStamped  # pylint: disable=import-error
        from std_msgs.msg import Float32  # pylint: disable=import-error

        self._rospy = rospy
        self._lock = threading.Lock()
        self._poses = []
        self._scores = []
        self._pose_sub = rospy.Subscriber(
            pose_topic, PoseStamped, self._pose_callback, queue_size=100
        )
        self._score_sub = rospy.Subscriber(
            score_topic, Float32, self._score_callback, queue_size=100
        )

    def _pose_callback(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        bag_time = self._rospy.Time.now()
        pose = {
            "t": msg_time(msg, bag_time),
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "roll": roll_from_quat(q),
            "pitch": pitch_from_quat(q),
            "yaw": yaw_from_quat(q),
        }
        with self._lock:
            self._poses.append(pose)

    def _score_callback(self, msg):
        with self._lock:
            self._scores.append((stamp_to_sec(self._rospy.Time.now()), float(msg.data)))

    def snapshot(self):
        with self._lock:
            poses = list(self._poses)
            scores = list(self._scores)
        poses.sort(key=lambda item: item["t"])
        scores.sort(key=lambda item: item[0])
        return poses, scores


def read_live(pose_topic, score_topic, duration_sec):
    try:
        import rospy  # pylint: disable=import-error
    except ImportError as exc:
        print("Failed to import rospy. Did you source /opt/ros/noetic/setup.bash?", file=sys.stderr)
        raise exc

    rospy.init_node("target_pose_jitter_analyzer", anonymous=True)
    collector = LiveCollector(pose_topic, score_topic)
    start = rospy.Time.now()
    rate = rospy.Rate(10.0)
    duration_text = f"{duration_sec:.1f}s" if duration_sec > 0.0 else "until Ctrl-C"
    print(f"Collecting live samples from {pose_topic} and {score_topic} for {duration_text}...")
    try:
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start).to_sec()
            if duration_sec > 0.0 and elapsed >= duration_sec:
                break
            rate.sleep()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    return collector.snapshot()


def filter_warmup(poses, scores, warmup_sec):
    if not poses or warmup_sec <= 0.0:
        return poses, scores
    start = poses[0]["t"] + warmup_sec
    return (
        [pose for pose in poses if pose["t"] >= start],
        [score for score in scores if score[0] >= start],
    )


def analyze(poses, scores, max_score_age):
    yaws = unwrap_angles([pose["yaw"] for pose in poses])
    rolls = unwrap_angles([pose["roll"] for pose in poses])
    pitches = unwrap_angles([pose["pitch"] for pose in poses])
    for i, pose in enumerate(poses):
        pose["yaw"] = yaws[i]
        pose["roll"] = rolls[i]
        pose["pitch"] = pitches[i]
        pose["score"] = nearest_score(scores, pose["t"], max_score_age)

    xs = [pose["x"] for pose in poses]
    ys = [pose["y"] for pose in poses]
    zs = [pose["z"] for pose in poses]
    mean_x = mean(xs)
    mean_y = mean(ys)
    mean_z = mean(zs)
    mean_yaw = mean(yaws)

    pos_err = []
    yaw_err = []
    adjacent_pos_jumps = []
    adjacent_yaw_jumps = []
    dts = []
    for i, pose in enumerate(poses):
        err = math.sqrt(
            (pose["x"] - mean_x) ** 2
            + (pose["y"] - mean_y) ** 2
            + (pose["z"] - mean_z) ** 2
        )
        pose["pos_err_m"] = err
        pose["yaw_err_rad"] = pose["yaw"] - mean_yaw
        pos_err.append(err)
        yaw_err.append(abs(pose["yaw_err_rad"]))
        if i > 0:
            prev = poses[i - 1]
            adjacent_pos_jumps.append(
                math.sqrt(
                    (pose["x"] - prev["x"]) ** 2
                    + (pose["y"] - prev["y"]) ** 2
                    + (pose["z"] - prev["z"]) ** 2
                )
            )
            adjacent_yaw_jumps.append(abs(pose["yaw"] - prev["yaw"]))
            dts.append(pose["t"] - prev["t"])

    score_pairs = [(pose["score"], pose["pos_err_m"]) for pose in poses if pose["score"] is not None]
    score_values = [item[0] for item in score_pairs]
    score_errs = [item[1] for item in score_pairs]

    duration = poses[-1]["t"] - poses[0]["t"] if len(poses) >= 2 else 0.0
    freq = (len(poses) - 1) / duration if duration > 0.0 and len(poses) >= 2 else 0.0

    return {
        "count": len(poses),
        "duration": duration,
        "freq": freq,
        "median_dt": percentile(dts, 50.0) if dts else 0.0,
        "mean_xyz": (mean_x, mean_y, mean_z),
        "mean_rpy": (mean(rolls), mean(pitches), mean_yaw),
        "std_xyz": (stddev(xs), stddev(ys), stddev(zs)),
        "std_rpy": (stddev(rolls), stddev(pitches), stddev(yaws)),
        "ptp_xyz": (peak_to_peak(xs), peak_to_peak(ys), peak_to_peak(zs)),
        "ptp_rpy": (peak_to_peak(rolls), peak_to_peak(pitches), peak_to_peak(yaws)),
        "rms_3d": math.sqrt(stddev(xs) ** 2 + stddev(ys) ** 2 + stddev(zs) ** 2),
        "pos_err_rms": math.sqrt(mean([v * v for v in pos_err])) if pos_err else 0.0,
        "pos_err_p95": percentile(pos_err, 95.0),
        "pos_err_max": max(pos_err) if pos_err else 0.0,
        "yaw_abs_err_p95": percentile(yaw_err, 95.0),
        "yaw_abs_err_max": max(yaw_err) if yaw_err else 0.0,
        "max_adjacent_pos_jump": max(adjacent_pos_jumps) if adjacent_pos_jumps else 0.0,
        "p95_adjacent_pos_jump": percentile(adjacent_pos_jumps, 95.0),
        "max_adjacent_yaw_jump": max(adjacent_yaw_jumps) if adjacent_yaw_jumps else 0.0,
        "p95_adjacent_yaw_jump": percentile(adjacent_yaw_jumps, 95.0),
        "score_count": len(score_values),
        "score_mean": mean(score_values) if score_values else float("nan"),
        "score_min": min(score_values) if score_values else float("nan"),
        "score_p05": percentile(score_values, 5.0) if score_values else float("nan"),
        "score_pos_err_corr": corrcoef(score_values, score_errs),
    }


def mm(value_m):
    return value_m * 1000.0


def deg(value_rad):
    return value_rad * 180.0 / math.pi


def fmt(value, precision=3):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


def print_report(metrics, pose_topic, score_topic):
    sx, sy, sz = metrics["std_xyz"]
    px, py, pz = metrics["ptp_xyz"]
    sr, sp, syaw = metrics["std_rpy"]
    pr, pp, pyaw = metrics["ptp_rpy"]
    mx, my, mz = metrics["mean_xyz"]
    mr, mp, myaw = metrics["mean_rpy"]

    print("Target pose jitter report")
    print(f"  pose_topic:  {pose_topic}")
    print(f"  score_topic: {score_topic}")
    print("")
    print("Samples")
    print(f"  pose_count:       {metrics['count']}")
    print(f"  duration_sec:     {fmt(metrics['duration'])}")
    print(f"  mean_rate_hz:     {fmt(metrics['freq'])}")
    print(f"  median_dt_sec:    {fmt(metrics['median_dt'])}")
    print("")
    print("Mean pose")
    print(f"  position_m:       x={fmt(mx)} y={fmt(my)} z={fmt(mz)}")
    print(f"  rpy_deg:          roll={fmt(deg(mr))} pitch={fmt(deg(mp))} yaw={fmt(deg(myaw))}")
    print("")
    print("Position jitter")
    print(f"  std_mm:           x={fmt(mm(sx))} y={fmt(mm(sy))} z={fmt(mm(sz))}")
    print(f"  rms_3d_std_mm:    {fmt(mm(metrics['rms_3d']))}")
    print(f"  peak_to_peak_mm:  x={fmt(mm(px))} y={fmt(mm(py))} z={fmt(mm(pz))}")
    print(f"  error_rms_mm:     {fmt(mm(metrics['pos_err_rms']))}")
    print(f"  error_p95_mm:     {fmt(mm(metrics['pos_err_p95']))}")
    print(f"  error_max_mm:     {fmt(mm(metrics['pos_err_max']))}")
    print("")
    print("Orientation jitter")
    print(f"  std_deg:          roll={fmt(deg(sr))} pitch={fmt(deg(sp))} yaw={fmt(deg(syaw))}")
    print(f"  peak_to_peak_deg: roll={fmt(deg(pr))} pitch={fmt(deg(pp))} yaw={fmt(deg(pyaw))}")
    print(f"  yaw_abs_p95_deg:  {fmt(deg(metrics['yaw_abs_err_p95']))}")
    print(f"  yaw_abs_max_deg:  {fmt(deg(metrics['yaw_abs_err_max']))}")
    print("")
    print("Adjacent-frame jumps")
    print(f"  pos_jump_p95_mm:  {fmt(mm(metrics['p95_adjacent_pos_jump']))}")
    print(f"  pos_jump_max_mm:  {fmt(mm(metrics['max_adjacent_pos_jump']))}")
    print(f"  yaw_jump_p95_deg: {fmt(deg(metrics['p95_adjacent_yaw_jump']))}")
    print(f"  yaw_jump_max_deg: {fmt(deg(metrics['max_adjacent_yaw_jump']))}")
    print("")
    print("Template score")
    print(f"  matched_scores:   {metrics['score_count']}")
    print(f"  score_mean:       {fmt(metrics['score_mean'])}")
    print(f"  score_min:        {fmt(metrics['score_min'])}")
    print(f"  score_p05:        {fmt(metrics['score_p05'])}")
    print(f"  corr(score,pos_error): {fmt(metrics['score_pos_err_corr'])}")


def write_csv(path, poses):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "x_m",
                "y_m",
                "z_m",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "score",
                "pos_err_mm",
                "yaw_err_deg",
            ]
        )
        for pose in poses:
            writer.writerow(
                [
                    f"{pose['t']:.9f}",
                    f"{pose['x']:.9f}",
                    f"{pose['y']:.9f}",
                    f"{pose['z']:.9f}",
                    f"{deg(pose['roll']):.9f}",
                    f"{deg(pose['pitch']):.9f}",
                    f"{deg(pose['yaw']):.9f}",
                    "" if pose["score"] is None else f"{pose['score']:.9f}",
                    f"{mm(pose['pos_err_m']):.9f}",
                    f"{deg(pose['yaw_err_rad']):.9f}",
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze target_pose jitter from a rosbag or from live ROS topics."
    )
    parser.add_argument(
        "bag",
        nargs="?",
        help="Input rosbag path. Required when --mode bag, ignored when --mode live.",
    )
    parser.add_argument(
        "--mode",
        choices=["bag", "live"],
        default="bag",
        help="bag: analyze an input rosbag; live: subscribe to ROS topics for a fixed duration.",
    )
    parser.add_argument(
        "--pose-topic",
        default="/reflective_board_identifier_node/target_pose",
        help="PoseStamped topic to analyze.",
    )
    parser.add_argument(
        "--score-topic",
        default="/reflective_board_identifier_node/best_score",
        help="Float32 template score topic to correlate with pose jitter.",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=0.0,
        help="Discard this many seconds from the beginning of the pose stream before analysis.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=30.0,
        help="Live mode sampling duration. Use <=0 to collect until Ctrl-C.",
    )
    parser.add_argument(
        "--max-score-age-sec",
        type=float,
        default=1.0,
        help="Maximum time offset for pairing a score sample with a pose sample.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV output path for per-sample pose/error data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "bag":
        if not args.bag:
            print("A rosbag path is required when --mode bag.", file=sys.stderr)
            return 2
        poses, scores = read_bag(args.bag, args.pose_topic, args.score_topic)
    else:
        poses, scores = read_live(args.pose_topic, args.score_topic, args.duration_sec)

    poses, scores = filter_warmup(poses, scores, args.warmup_sec)

    if not poses:
        print(f"No pose samples found on topic {args.pose_topic}", file=sys.stderr)
        return 2

    metrics = analyze(poses, scores, args.max_score_age_sec)
    print_report(metrics, args.pose_topic, args.score_topic)

    if args.csv:
        write_csv(args.csv, poses)
        print("")
        print(f"Wrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
