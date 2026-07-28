import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import yaml
import os
import re
from pathlib import Path

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

import tf2_ros
from tf2_ros import TransformException
import tf_transformations

class LidarRadarImageFusion(Node):

    def __init__(self):
        super().__init__('lidar_radar_image_fusion')

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/fusion/projected_image', 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.load_camera_calibration()

        self.image_sub = self.create_subscription(Image, '/yolo/detections', self.image_callback, 10)
        self.lidar_sub = self.create_subscription(PointCloud2, '/filtered_points', self.lidar_callback, 10)
        
        # UPDATED: Now subscribing to the Preprocessed topic, not the raw driver topic!
        self.radar_sub = self.create_subscription(PointCloud2, '/radar/preprocessed', self.radar_callback, 10)

        self.latest_image_msg = None
        self.latest_lidar_msg = None
        self.latest_radar_msg = None
        
        self.get_logger().info("🚀 Projection Node Online (Listening to Preprocessed Data)!")

    def load_camera_calibration(self):
        calibration_filename = 'camera_front_center_autoware_camera_calibration.yaml'
        workspace_root = Path.cwd()
        candidates = [
            workspace_root / 'src' / 'ai_sensor_fusion' / 'config' / calibration_filename,
            workspace_root / 'data' / 'raw_bags' / 'bag_1' / calibration_filename,
            workspace_root / 'data' / 'raw_bags' / 'bag_2' / calibration_filename,
        ]

        yaml_path = next((str(path) for path in candidates if path.exists()), None)
        if yaml_path is None:
            raise FileNotFoundError(f"Camera calibration YAML not found.")

        try:
            with open(yaml_path, 'r') as f:
                raw = f.read()

            raw = re.sub(r'^%YAML:.*\n', '', raw, flags=re.MULTILINE)
            raw = raw.replace('!!opencv-matrix', '')
            calib = yaml.safe_load(raw)

            camera_mat = calib['CameraMat']
            if isinstance(camera_mat, dict) and 'data' in camera_mat:
                data = camera_mat['data']
                self.fx = float(data[0])
                self.fy = float(data[4])
                self.cx = float(data[2])
                self.cy = float(data[5])
            else:
                raise ValueError('Unsupported CameraMat format')

        except Exception as ex:
            self.get_logger().error(f"Failed to load calibration: {ex}")
            raise

    def get_transform(self, source_frame):
        try:
            tf = self.tf_buffer.lookup_transform('cam0', source_frame, rclpy.time.Time())
            trans = tf.transform.translation
            rot = tf.transform.rotation
            T = tf_transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
            T[0, 3] = trans.x
            T[1, 3] = trans.y
            T[2, 3] = trans.z
            return T
        except TransformException as ex:
            return None

    def image_callback(self, msg):
        self.latest_image_msg = msg
        self.try_fuse()

    def lidar_callback(self, msg):
        self.latest_lidar_msg = msg
        self.try_fuse()

    def radar_callback(self, msg):
        self.latest_radar_msg = msg
        self.try_fuse()

    def try_fuse(self):
        if self.latest_image_msg is None or self.latest_lidar_msg is None or self.latest_radar_msg is None:
            return
        self.sync_callback(self.latest_image_msg, self.latest_lidar_msg, self.latest_radar_msg)

    def sync_callback(self, image_msg, lidar_msg, radar_msg):
        
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        h, w = img.shape[:2]
        
        # ==========================================
        # PART A: LiDAR PROJECTION
        # ==========================================
        raw_lidar = pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
        lidar_points_list = [(p[0], p[1], p[2]) for p in raw_lidar]

        if len(lidar_points_list) > 0:
            l_points = np.array(lidar_points_list, dtype=np.float64)
            l_points = np.atleast_2d(l_points)

            T_lidar = self.get_transform('velodyne')
            if T_lidar is not None:
                ones_l = np.ones((l_points.shape[0], 1))
                l_pts_h = np.concatenate([l_points, ones_l], axis=1)
                l_pts_cam = (T_lidar @ l_pts_h.T).T

                X_l, Y_l, Z_l = l_pts_cam[:, 0], l_pts_cam[:, 1], l_pts_cam[:, 2]

                mask_l = Z_l > 0.1
                X_l, Y_l, Z_l = X_l[mask_l], Y_l[mask_l], Z_l[mask_l]

                if len(Z_l) > 0:
                    u_l = (self.fx * X_l / Z_l) + self.cx
                    v_l = (self.fy * Y_l / Z_l) + self.cy

                    valid_l = (u_l >= 0) & (u_l < w) & (v_l >= 0) & (v_l < h)
                    u_valid_l = u_l[valid_l].astype(np.int32)
                    v_valid_l = v_l[valid_l].astype(np.int32)
                    z_valid_l = Z_l[valid_l] 

                    MAX_DISTANCE = 40.0
                    depth_ratio = np.clip(z_valid_l / MAX_DISTANCE, 0.0, 1.0)
                    inverted_ratio = 1.0 - depth_ratio
                    norm_depth = (inverted_ratio * 255.0).astype(np.uint8)
                    color_mapped = cv2.applyColorMap(norm_depth.reshape(-1, 1), cv2.COLORMAP_JET)

                    for i in range(len(u_valid_l)):
                        b, g, r = color_mapped[i, 0]
                        cv2.circle(img, (u_valid_l[i], v_valid_l[i]), 2, (int(b), int(g), int(r)), -1)

        # ==========================================
        # PART B: RADAR PROJECTION
        # ==========================================
        # Data is already clean from the preprocessor, we just extract it and project
        raw_radar = pc2.read_points(radar_msg, field_names=("x", "y", "z", "v"), skip_nans=True)
        radar_points_list = [(p[0], p[1], p[2], p[3]) for p in raw_radar]

        if len(radar_points_list) > 0:
            r_points_full = np.array(radar_points_list, dtype=np.float64)
            r_points = r_points_full[:, 0:3] 
            r_velocities = r_points_full[:, 3] 

            T_radar = self.get_transform('radar') 
            
            if T_radar is not None:
                ones_r = np.ones((r_points.shape[0], 1))
                r_pts_h = np.concatenate([r_points, ones_r], axis=1)
                r_pts_cam = (T_radar @ r_pts_h.T).T

                X_r, Y_r, Z_r = r_pts_cam[:, 0], r_pts_cam[:, 1], r_pts_cam[:, 2]

                # Keep points strictly in front of camera
                mask_r = Z_r > 0.1
                X_r, Y_r, Z_r = X_r[mask_r], Y_r[mask_r], Z_r[mask_r]
                valid_velocities = r_velocities[mask_r]

                if len(Z_r) > 0:
                    u_r = (self.fx * X_r / Z_r) + self.cx
                    v_r_pixel = (self.fy * Y_r / Z_r) + self.cy

                    valid_r = (u_r >= 0) & (u_r < w) & (v_r_pixel >= 0) & (v_r_pixel < h)
                    u_valid_r = u_r[valid_r].astype(np.int32)
                    v_valid_r = v_r_pixel[valid_r].astype(np.int32)
                    vel_valid = valid_velocities[valid_r]

                    for i in range(len(u_valid_r)):
                        cv2.drawMarker(img, (u_valid_r[i], v_valid_r[i]), (255, 0, 255), 
                                       markerType=cv2.MARKER_CROSS, markerSize=15, thickness=3)
                        
                        vel_text = f"{vel_valid[i]:.1f}m/s"
                        cv2.putText(img, vel_text, (u_valid_r[i] + 10, v_valid_r[i] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # ==========================================
        # PUBLISH FINAL IMAGE
        # ==========================================
        out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        out_msg.header = image_msg.header 
        self.image_pub.publish(out_msg)


def main():
    rclpy.init()
    node = LidarRadarImageFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()