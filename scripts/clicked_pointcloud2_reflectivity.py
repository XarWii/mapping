#!/usr/bin/env python3

import math
import threading

import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2


class ClickedPointCloud2Reflectivity:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/livox/lidar")
        self.clicked_topic = rospy.get_param("~clicked_topic", "/clicked_point")
        self.search_radius_m = rospy.get_param("~search_radius_m", 0.05)
        self.max_cloud_age_sec = rospy.get_param("~max_cloud_age_sec", 0.5)
        self.warn_frame_mismatch = rospy.get_param("~warn_frame_mismatch", True)

        self._lock = threading.Lock()
        self._latest_msg = None
        self._latest_stamp = rospy.Time(0)

        self._cloud_sub = rospy.Subscriber(
            self.input_topic, PointCloud2, self._cloud_callback, queue_size=2
        )
        self._clicked_sub = rospy.Subscriber(
            self.clicked_topic, PointStamped, self._clicked_callback, queue_size=10
        )

        rospy.loginfo(
            "clicked_pointcloud2_reflectivity started: input=%s clicked=%s radius=%.3fm",
            self.input_topic,
            self.clicked_topic,
            self.search_radius_m,
        )

    def _cloud_callback(self, msg):
        stamp = msg.header.stamp
        if stamp == rospy.Time(0):
            stamp = rospy.Time.now()
        with self._lock:
            self._latest_msg = msg
            self._latest_stamp = stamp

    def _clicked_callback(self, clicked):
        with self._lock:
            msg = self._latest_msg
            stamp = self._latest_stamp

        if msg is None:
            rospy.logwarn("clicked point received, but no PointCloud2 cloud has arrived yet")
            return

        cloud_age = (rospy.Time.now() - stamp).to_sec()
        if cloud_age > self.max_cloud_age_sec:
            rospy.logwarn(
                "latest PointCloud2 is stale: age=%.3fs max=%.3fs",
                cloud_age,
                self.max_cloud_age_sec,
            )

        cloud_frame = msg.header.frame_id
        clicked_frame = clicked.header.frame_id
        if self.warn_frame_mismatch and cloud_frame and clicked_frame and cloud_frame != clicked_frame:
            rospy.logwarn(
                "frame mismatch: cloud frame=%s clicked frame=%s; no TF transform is applied",
                cloud_frame,
                clicked_frame,
            )

        try:
            result = self._find_nearest_and_region_stats(clicked.point, msg)
        except Exception as exc:
            rospy.logerr("failed to read PointCloud2 fields: %s", str(exc))
            return

        if result is None:
            rospy.logwarn("PointCloud2 has no valid xyz points")
            return

        nearest, nearest_dist, field_name, region = result
        field_text = self._format_field(field_name, nearest["value"])
        region_text = self._format_region(field_name, region)

        rospy.loginfo(
            "clicked=(%.3f, %.3f, %.3f) nearest=(%.3f, %.3f, %.3f) err=%.3fm %s %s",
            clicked.point.x,
            clicked.point.y,
            clicked.point.z,
            nearest["x"],
            nearest["y"],
            nearest["z"],
            nearest_dist,
            field_text,
            region_text,
        )

    def _find_nearest_and_region_stats(self, clicked_point, msg):
        available_fields = [field.name for field in msg.fields]
        value_field = self._choose_value_field(available_fields)
        read_fields = ["x", "y", "z"]
        if value_field is not None:
            read_fields.append(value_field)

        nearest = None
        nearest_dist_sq = float("inf")
        radius_sq = self.search_radius_m * self.search_radius_m

        region_count = 0
        region_sum = 0.0
        region_min = None
        region_max = None

        for point in pc2.read_points(msg, field_names=read_fields, skip_nans=True):
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            if not self._valid_xyz(x, y, z):
                continue

            value = point[3] if value_field is not None else None
            dx = x - clicked_point.x
            dy = y - clicked_point.y
            dz = z - clicked_point.z
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq < nearest_dist_sq:
                nearest = {"x": x, "y": y, "z": z, "value": value}
                nearest_dist_sq = dist_sq

            if value_field is not None and dist_sq <= radius_sq:
                numeric_value = self._numeric_value(value)
                if numeric_value is None:
                    continue
                region_count += 1
                region_sum += numeric_value
                region_min = numeric_value if region_min is None else min(region_min, numeric_value)
                region_max = numeric_value if region_max is None else max(region_max, numeric_value)

        if nearest is None:
            return None

        region_mean = region_sum / region_count if region_count > 0 else 0.0
        region = {
            "count": region_count,
            "mean": region_mean,
            "min": region_min,
            "max": region_max,
            "available_fields": available_fields,
        }
        return nearest, math.sqrt(nearest_dist_sq), value_field, region

    @staticmethod
    def _choose_value_field(available_fields):
        for field_name in ("reflectivity", "intensity", "intensities"):
            if field_name in available_fields:
                return field_name
        if "rgb" in available_fields:
            return "rgb"
        if "rgba" in available_fields:
            return "rgba"
        return None

    @staticmethod
    def _valid_xyz(x, y, z):
        return math.isfinite(x) and math.isfinite(y) and math.isfinite(z)

    @staticmethod
    def _numeric_value(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_field(field_name, value):
        if field_name is None:
            return "value=unavailable"
        if field_name in ("rgb", "rgba"):
            return "%s=%s (color field, not raw reflectivity)" % (field_name, str(value))
        return "%s=%.1f" % (field_name, float(value))

    @staticmethod
    def _format_region(field_name, region):
        if field_name is None:
            return "available_fields=%s" % ",".join(region["available_fields"])
        if field_name in ("rgb", "rgba"):
            return "region: color-only field; available_fields=%s" % ",".join(region["available_fields"])
        if region["count"] <= 0:
            return "region: no points within radius"
        return "region: n=%d mean=%.1f min=%.1f max=%.1f radius=%.3fm" % (
            region["count"],
            region["mean"],
            region["min"],
            region["max"],
            rospy.get_param("~search_radius_m", 0.05),
        )


if __name__ == "__main__":
    rospy.init_node("clicked_pointcloud2_reflectivity")
    ClickedPointCloud2Reflectivity()
    rospy.spin()
