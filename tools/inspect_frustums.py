"""
inspect_frustums.py  —  Is radar actually reaching the fused frustums?
========================================================================
We proved /radar/preprocessed has real data. We proved the model detects
cars and pedestrians. But we never checked whether any of those detections
actually USED a radar point. This script answers that directly.

Subscribes to /fusion/ready_frustums, decodes 5 real messages, and reports
how many of each frustum's 1024 points are radar-sourced (modality flag = 1)
vs LiDAR-sourced (flag = 0) — straight from the 7-channel feature layout:
  [0:3] XYZ   [3] intensity   [4:6] radar vx,vy   [6] modality flag

Run with:   python3 inspect_frustums.py
(needs ROS2 sourced; no colcon build required, runs as a plain script)
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

LABELS = {0: 'Car', 1: 'Pedestrian', 2: 'Bicycle', 3: 'Truck'}
# Bag is 23.6s and loops. At ~1.9Hz that's ~45 messages per full loop.
# Listen for 70 to comfortably guarantee at least one full loop is covered
# no matter where in the loop we happen to start listening.
MAX_MESSAGES = 70


class FrustumInspector(Node):
    def __init__(self):
        super().__init__('frustum_inspector')
        self.sub = self.create_subscription(
            Float32MultiArray, '/fusion/ready_frustums', self.callback, 10)
        self.count = 0
        # Per-class tallies: how many frustums of each class, how many had radar
        self.class_seen = {}
        self.class_radar = {}
        print("Listening on /fusion/ready_frustums ... (waiting for messages)")
        print("Specifically chasing a Pedestrian frustum this time.")
        print()

    def callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)

        if data.size == 1024 * 7 + 2:
            fusion_class = data[0]
            points = data[2:].reshape(1024, 7)
        elif data.size == 1024 * 7:
            fusion_class = None
            points = data.reshape(1024, 7)
        else:
            self.get_logger().warn(f"Unexpected array size {data.size}, skipping")
            return

        modality = points[:, 6]
        n_radar = int(np.sum(modality == 1.0))
        n_lidar = int(np.sum(modality == 0.0))
        label = LABELS.get(int(fusion_class), 'Unknown') if fusion_class is not None else '?'

        self.count += 1
        self.class_seen[label] = self.class_seen.get(label, 0) + 1
        if n_radar > 0:
            self.class_radar[label] = self.class_radar.get(label, 0) + 1

        print(f"--- Frustum #{self.count}  [{label}] ---")
        print(f"  lidar points : {n_lidar} / 1024")
        print(f"  radar points : {n_radar} / 1024")

        if n_radar > 0:
            radar_pts = points[modality == 1.0]
            vx, vy = radar_pts[:, 4], radar_pts[:, 5]
            print(f"  radar vx range: {vx.min():.2f} to {vx.max():.2f} m/s")
            print(f"  radar vy range: {vy.min():.2f} to {vy.max():.2f} m/s")
            print(f"  >>> RADAR IS CONTRIBUTING to this frustum  [{label}]")
        else:
            print(f"  (lidar-only this frustum - no radar points landed inside it)")
        print()

        if self.count >= MAX_MESSAGES:
            print("=" * 60)
            print("SUMMARY BY CLASS:")
            for cls, seen in self.class_seen.items():
                hit = self.class_radar.get(cls, 0)
                print(f"  {cls:12s}: {hit} of {seen} frustums had radar points")
            print("=" * 60)
            if self.class_radar.get('Pedestrian', 0) > 0:
                print("CONFIRMED: radar is contributing specifically on the")
                print("pedestrian — the one moving object in this scene.")
            elif 'Pedestrian' in self.class_seen:
                print("Pedestrian frustums were seen, but NONE carried a radar")
                print("point. Worth a closer look at the radar->cam0 transform")
                print("or the velocity threshold.")
            else:
                print("No Pedestrian frustum was captured in this window at all.")
                print("Try running longer, or check the bag actually loops back")
                print("over the segment where the person is visible.")
            print("=" * 60)
            rclpy.shutdown()


def main():
    rclpy.init()
    node = FrustumInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()