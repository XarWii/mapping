#!/usr/bin/env python3

import math
import struct

import rospy
from sensor_msgs.msg import PointCloud2, PointField


DATATYPE_FORMATS = {
    PointField.INT8: ("b", 1),
    PointField.UINT8: ("B", 1),
    PointField.INT16: ("h", 2),
    PointField.UINT16: ("H", 2),
    PointField.INT32: ("i", 4),
    PointField.UINT32: ("I", 4),
    PointField.FLOAT32: ("f", 4),
    PointField.FLOAT64: ("d", 8),
}


class PointCloud2IntensityFilter:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/livox/lidar")
        self.output_topic = rospy.get_param("~output_topic", "/reflectivity_filtered_cloud")
        self.threshold = float(rospy.get_param("~threshold", 150.0))
        self.field_name = str(rospy.get_param("~field", "auto"))
        self.inclusive = bool(rospy.get_param("~inclusive", True))
        self.log_stats = bool(rospy.get_param("~log_stats", True))
        queue_size = int(rospy.get_param("~queue_size", 2))

        self._struct_cache = {}
        self._warned_missing_field = False
        self._warned_bad_type = False

        self.publisher = rospy.Publisher(
            self.output_topic, PointCloud2, queue_size=queue_size
        )
        self.subscriber = rospy.Subscriber(
            self.input_topic, PointCloud2, self._cloud_callback, queue_size=queue_size
        )

        op = ">=" if self.inclusive else ">"
        rospy.loginfo(
            "pointcloud2_intensity_filter started: input=%s output=%s field=%s threshold %s %.3f",
            self.input_topic,
            self.output_topic,
            self.field_name,
            op,
            self.threshold,
        )

    def _cloud_callback(self, msg):
        field = self._select_field(msg)
        if field is None:
            if not self._warned_missing_field:
                rospy.logwarn(
                    "PointCloud2 has no usable intensity field. Available fields: %s",
                    ",".join(field.name for field in msg.fields),
                )
                self._warned_missing_field = True
            return

        reader = self._field_reader(msg, field)
        if reader is None:
            return

        total = int(msg.width) * int(msg.height)
        out_data = bytearray()
        kept = 0
        data = msg.data

        for row in range(int(msg.height)):
            row_base = row * int(msg.row_step)
            for col in range(int(msg.width)):
                point_base = row_base + col * int(msg.point_step)
                if point_base + int(msg.point_step) > len(data):
                    continue
                value = reader(data, point_base)
                if self._passes(value):
                    out_data.extend(data[point_base:point_base + int(msg.point_step)])
                    kept += 1

        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = kept
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = msg.point_step * kept
        out.data = bytes(out_data)
        out.is_dense = msg.is_dense
        self.publisher.publish(out)

        if self.log_stats:
            rospy.loginfo_throttle(
                1.0,
                "filtered %s by %s %.3f: kept %d/%d point(s)",
                field.name,
                ">=" if self.inclusive else ">",
                self.threshold,
                kept,
                total,
            )

    def _select_field(self, msg):
        fields = {field.name: field for field in msg.fields}
        if self.field_name and self.field_name != "auto":
            return fields.get(self.field_name)
        for name in ("reflectivity", "intensity", "intensities"):
            if name in fields:
                return fields[name]
        return None

    def _field_reader(self, msg, field):
        fmt_size = DATATYPE_FORMATS.get(field.datatype)
        if fmt_size is None:
            if not self._warned_bad_type:
                rospy.logwarn(
                    "field %s has unsupported datatype=%d",
                    field.name,
                    field.datatype,
                )
                self._warned_bad_type = True
            return None

        fmt, size = fmt_size
        if field.offset + size > msg.point_step:
            rospy.logwarn_throttle(
                1.0,
                "field %s offset=%d does not fit point_step=%d",
                field.name,
                field.offset,
                msg.point_step,
            )
            return None

        endian = ">" if msg.is_bigendian else "<"
        cache_key = (msg.is_bigendian, field.datatype)
        reader = self._struct_cache.get(cache_key)
        if reader is None:
            reader = struct.Struct(endian + fmt)
            self._struct_cache[cache_key] = reader

        offset = int(field.offset)

        def read_value(data, point_base):
            return reader.unpack_from(data, point_base + offset)[0]

        return read_value

    def _passes(self, value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(numeric):
            return False
        if self.inclusive:
            return numeric >= self.threshold
        return numeric > self.threshold


if __name__ == "__main__":
    rospy.init_node("pointcloud2_intensity_filter")
    PointCloud2IntensityFilter()
    rospy.spin()
