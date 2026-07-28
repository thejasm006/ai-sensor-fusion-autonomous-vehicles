"""
watch_car_jump.py  —  Catch the exact moment the Car frustum jumps depth
============================================================================
Listens to /fusion/ready_frustums and tracks the Car class's depth frame to
frame. Stays QUIET during normal operation. The instant the Car's depth
jumps by more than JUMP_THRESHOLD metres in a single frame (physically
implausible for a parked/slow car between consecutive frames), it prints a
loud alert with a full depth histogram of that exact frame's LiDAR points.

This tells us directly: during the jump, is there ONLY a near cluster
(confirms genuine full occlusion -- no real car points survived that frame)
or is there STILL a smaller far cluster present too (would point to a
different, separate bug worth chasing).

Run this in one terminal WHILE WATCHING RVIZ in another. The moment you see
the box jump on screen, this terminal should print an alert within the same
second -- that confirms we're looking at the same event.

Run with:   python3 watch_car_jump.py
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time

JUMP_THRESHOLD = 3.0   # metres -- bigger than this between consecutive Car
                       # frames is physically implausible for this scene


class CarJumpWatcher(Node):
    def __init__(self):
        super().__init__('car_jump_watcher')
        self.sub = self.create_subscription(
            Float32MultiArray, '/fusion/ready_frustums', self.callback, 10)
        self.last_car_depth = None
        self.frame_count = 0
        self.car_frame_count = 0
        print("Watching for Car depth jumps > {:.1f}m ... (quiet until something happens)".format(JUMP_THRESHOLD))
        print("Watch RViz now. This will print the moment a jump is detected.")
        print()

    def callback(self, msg):
        self.frame_count += 1
        data = np.array(msg.data, dtype=np.float32)

        if data.size == 1024 * 7 + 2:
            fusion_class = int(data[0])
            points = data[2:].reshape(1024, 7)
        else:
            return

        if fusion_class != 0:  # only watching Car (class 0)
            return

        self.car_frame_count += 1
        modality = points[:, 6]
        lidar_pts = points[modality == 0.0]
        if len(lidar_pts) == 0:
            return

        depths = lidar_pts[:, 2]
        mean_depth = float(np.mean(depths))
        timestamp = time.strftime("%H:%M:%S")

        if self.last_car_depth is not None:
            jump = abs(mean_depth - self.last_car_depth)
            if jump > JUMP_THRESHOLD:
                print("=" * 64)
                print(f"!!! JUMP DETECTED at {timestamp} "
                      f"(car frame #{self.car_frame_count}, overall frame #{self.frame_count})")
                print(f"    previous mean depth: {self.last_car_depth:.2f}m")
                print(f"    new mean depth:      {mean_depth:.2f}m")
                print(f"    jump size:            {jump:.2f}m")
                print()
                print(f"    Depth histogram of THIS frame's {len(lidar_pts)} LiDAR points:")
                bins = [(0,5),(5,8),(8,10),(10,12),(12,14),(14,16),(16,20),(20,30),(30,50)]
                for lo, hi in bins:
                    c = int(np.sum((depths >= lo) & (depths < hi)))
                    if c > 0:
                        print(f"      {lo:3d}-{hi:3d}m : {c:4d}  {'#' * min(40, c)}")
                print()
                near_count = int(np.sum(depths < 8))
                far_count = int(np.sum(depths >= 10))
                if far_count < 5:
                    print("    -> NO meaningful far cluster survives. This looks like")
                    print("       genuine full occlusion -- the car's real points simply")
                    print("       weren't collected this frame.")
                elif far_count > 20:
                    print("    -> A real far cluster DOES still exist in this frame!")
                    print("       Worth investigating why it wasn't selected/kept.")
                else:
                    print("    -> Ambiguous -- some far points present but sparse.")
                print("=" * 64)
                print()

        self.last_car_depth = mean_depth


def main():
    rclpy.init()
    node = CarJumpWatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
