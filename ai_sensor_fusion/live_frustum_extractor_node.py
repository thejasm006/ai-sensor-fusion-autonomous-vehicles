#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import yaml
import re
import cv2
from pathlib import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32MultiArray, Header
import sensor_msgs_py.point_cloud2 as pc2

import tf2_ros
from tf2_ros import TransformException
import tf_transformations

YOLO_TO_FUSION_CLASS = {
    0:  1,   # person       → Pedestrian
    1:  2,   # bicycle      → Bicycle
    2:  0,   # car          → Car
    3:  2,   # motorcycle   → Bicycle
    5:  0,   # bus          → Car
    7:  3,   # truck        → Truck
}

T_LIDAR_TO_CAM0 = np.array([
    [-0.021, -1.000,  0.005,  0.016],
    [-0.162, -0.002, -0.987, -0.066],
    [ 0.987, -0.022, -0.161, -0.712],
    [ 0.000,  0.000,  0.000,  1.000]
], dtype=np.float64)


class LiveFrustumExtractor(Node):

    MIN_LIDAR_PTS    = 5      
    MIN_HEIGHT_SPAN  = 0.30   
    CHASSIS_Z_MIN    = -1.5   

    def __init__(self):
        super().__init__('live_frustum_extractor')

        self.frustum_pub = self.create_publisher(
            Float32MultiArray, '/fusion/ready_frustums', 10)
            
        self.frustum_pc_pub = self.create_publisher(
            PointCloud2, '/fusion/visualized_frustums', 10)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.load_camera_calibration()
        self.bridge = CvBridge()

        self.mask_sub  = self.create_subscription(
            Image,            '/yolo/instance_mask', self.mask_callback,  10)
        self.bbox_sub  = self.create_subscription(
            Detection2DArray, '/detections',          self.bbox_callback,  10)
        self.lidar_sub = self.create_subscription(
            PointCloud2,      '/filtered_points',     self.lidar_callback, 10)
        self.radar_sub = self.create_subscription(
            PointCloud2,      '/radar/preprocessed',  self.radar_callback, 10)

        self.latest_mask      = None
        self.latest_bbox_msg  = None
        self.latest_lidar_msg = None
        self.latest_radar_msg = None

        self.get_logger().info('✅ LiveFrustumExtractor ready (Point Stealer Strategy Enabled)')

    def load_camera_calibration(self):
        calibration_filename = 'camera_front_center_autoware_camera_calibration.yaml'
        workspace_root = Path.cwd()
        candidates = [
            workspace_root / 'src' / 'ai_sensor_fusion' / 'config' / calibration_filename,
            workspace_root / 'data' / 'raw_bags' / 'bag_1' / calibration_filename,
            workspace_root / 'data' / 'raw_bags' / 'bag_2' / calibration_filename,
        ]
        yaml_path = next((str(p) for p in candidates if p.exists()), None)
        if yaml_path is None:
            raise FileNotFoundError(f'Camera calibration YAML not found.')

        with open(yaml_path, 'r') as f:
            raw = f.read()
        raw = re.sub(r'^%YAML:.*\n', '', raw, flags=re.MULTILINE)
        raw = raw.replace('!!opencv-matrix', '')
        calib = yaml.safe_load(raw)

        if 'ProjectionMat' in calib:
            data    = calib['ProjectionMat']['data']
            self.fx = float(data[0])
            self.cx = float(data[2])
            self.fy = float(data[5])
            self.cy = float(data[6])
        elif 'CameraMat' in calib:
            data    = calib['CameraMat']['data']
            self.fx = float(data[0])
            self.cx = float(data[2])
            self.fy = float(data[4])
            self.cy = float(data[5])
        else:
            raise ValueError('Neither ProjectionMat nor CameraMat found in YAML.')

        self.K = np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0    ]
        ], dtype=np.float64)

        self.dist = np.zeros(4, dtype=np.float64)

    def get_radar_transform(self):
        try:
            tf = self.tf_buffer.lookup_transform('cam0', 'radar', rclpy.time.Time())
            t = tf.transform.translation
            r = tf.transform.rotation
            T = tf_transformations.quaternion_matrix([r.x, r.y, r.z, r.w])
            T[0, 3] = t.x
            T[1, 3] = t.y
            T[2, 3] = t.z
            return T
        except TransformException:
            return None

    def project_to_pixels(self, cam_xyz: np.ndarray):
        if len(cam_xyz) == 0:
            empty = np.zeros(0, dtype=np.int32)
            return empty, empty, np.zeros(0, dtype=bool)

        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        pts2d, _ = cv2.projectPoints(
            cam_xyz.astype(np.float64).reshape(-1, 1, 3), rvec, tvec, self.K, self.dist)
        pts2d = pts2d.reshape(-1, 2)

        valid = (
            np.isfinite(pts2d).all(axis=1) &
            (pts2d[:, 0] > -1e5) & (pts2d[:, 0] < 1e5) &
            (pts2d[:, 1] > -1e5) & (pts2d[:, 1] < 1e5)
        )
        u = np.where(valid, pts2d[:, 0], 0).astype(np.int32)
        v = np.where(valid, pts2d[:, 1], 0).astype(np.int32)
        return u, v, valid

    def filter_clusters_by_class(self, xyz: np.ndarray, intens: np.ndarray, fusion_class: int, gap_m: float) -> tuple:
        if len(xyz) < 2:
            return xyz, intens

        order     = np.argsort(xyz[:, 2])          
        xyz_s     = xyz[order]
        intens_s  = intens[order]
        depths    = xyz_s[:, 2]
        gaps      = np.diff(depths)                 

        if fusion_class in [1, 2]:
            # ── Pedestrians/Bicycles: Noise Shield Bypass ──
            gap_idx = np.where(gaps > gap_m)[0]
            if len(gap_idx) == 0:
                return xyz_s, intens_s                      

            split_points = gap_idx + 1
            xyz_clusters = np.split(xyz_s, split_points)
            intens_clusters = np.split(intens_s, split_points)

            for i, c in enumerate(xyz_clusters):
                if len(c) >= self.MIN_LIDAR_PTS:
                    return c, intens_clusters[i]
            
            largest_idx = np.argmax([len(c) for c in xyz_clusters])
            return xyz_clusters[largest_idx], intens_clusters[largest_idx]

        else:
            # ── Cars/Trucks: Macro-Gap Shield ──
            macro_gap_threshold = 1.0 
            macro_gap_idx = np.where(gaps > macro_gap_threshold)[0]
            
            if len(macro_gap_idx) == 0:
                return xyz_s, intens_s
                
            split_points = macro_gap_idx + 1
            xyz_clusters = np.split(xyz_s, split_points)
            intens_clusters = np.split(intens_s, split_points)
            
            best_idx = 0
            max_span = -1.0
            
            for i, c in enumerate(xyz_clusters):
                span = c[-1, 2] - c[0, 2]
                if span > max_span:
                    max_span = span
                    best_idx = i
                    
            return xyz_clusters[best_idx], intens_clusters[best_idx]

    def radar_preserving_sample(self, fused: np.ndarray, target: int = 1024) -> np.ndarray:
        radar_pts = fused[fused[:, 6] == 1.0]
        lidar_pts = fused[fused[:, 6] == 0.0]
        n_need    = target - len(radar_pts)

        if n_need <= 0:
            return radar_pts[:target]
        if len(lidar_pts) == 0:
            return np.vstack((radar_pts, np.zeros((n_need, 7), dtype=np.float32)))

        idx = (np.random.choice(len(lidar_pts), n_need, replace=True)
               if len(lidar_pts) < n_need
               else np.random.choice(len(lidar_pts), n_need, replace=False))
        return np.vstack((radar_pts, lidar_pts[idx])).astype(np.float32)

    def mask_callback(self, msg):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.try_fuse()
        except Exception as e:
            self.get_logger().error(f'Mask convert error: {e}')

    def bbox_callback(self, msg):
        self.latest_bbox_msg = msg
        self.try_fuse()

    def lidar_callback(self, msg):
        self.latest_lidar_msg = msg
        self.try_fuse()

    def radar_callback(self, msg):
        self.latest_radar_msg = msg
        self.try_fuse()

    def try_fuse(self):
        if (self.latest_mask      is None or
            self.latest_bbox_msg  is None or
            self.latest_lidar_msg is None or
            self.latest_radar_msg is None):
            return
        if not self.latest_bbox_msg.detections:
            return
        self.process_frame()
        self.latest_mask = None

    def process_frame(self):
        img_h, img_w = self.latest_mask.shape

        raw_lidar = list(pc2.read_points(
            self.latest_lidar_msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True))
        if not raw_lidar:
            return

        l_arr    = np.array([tuple(p) for p in raw_lidar], dtype=np.float64)
        l_xyz_v  = l_arr[:, 0:3]
        l_intens = l_arr[:, 3].astype(np.float32)

        ground_mask = l_xyz_v[:, 2] > self.CHASSIS_Z_MIN
        l_xyz_v     = l_xyz_v[ground_mask]
        l_intens    = l_intens[ground_mask]
        if len(l_xyz_v) == 0:
            return

        l_h       = np.concatenate([l_xyz_v, np.ones((len(l_xyz_v), 1))], axis=1)
        l_cam     = (T_LIDAR_TO_CAM0 @ l_h.T).T
        l_cam_xyz = l_cam[:, 0:3].astype(np.float32)

        front_mask = l_cam_xyz[:, 2] > 0.1
        l_cam_xyz  = l_cam_xyz[front_mask]
        l_intens   = l_intens[front_mask]
        if len(l_cam_xyz) == 0:
            return

        u_l, v_l, valid_math_l = self.project_to_pixels(l_cam_xyz)
        in_frame_l = (valid_math_l & (u_l >= 0) & (u_l < img_w) & (v_l >= 0) & (v_l < img_h))
        l_cam_xyz = l_cam_xyz[in_frame_l]
        l_intens  = l_intens[in_frame_l]
        u_l       = u_l[in_frame_l]
        v_l       = v_l[in_frame_l]

        hit_ids_l = self.latest_mask[v_l, u_l]

        has_radar = False
        r_cam_xyz = np.zeros((0, 3), dtype=np.float32)
        r_vx = r_vy = np.zeros(0, dtype=np.float32)
        hit_ids_r = np.zeros(0, dtype=np.int32)

        T_radar = self.get_radar_transform()
        if T_radar is not None:
            raw_radar = list(pc2.read_points(
                self.latest_radar_msg, field_names=('x', 'y', 'z', 'v'), skip_nans=True))
            if raw_radar:
                r_arr   = np.array([tuple(p) for p in raw_radar], dtype=np.float64)
                r_xyz_s = r_arr[:, 0:3]
                r_v_raw = r_arr[:, 3].astype(np.float32)

                r_h             = np.concatenate([r_xyz_s, np.ones((len(r_xyz_s), 1))], axis=1)
                r_cam_all       = (T_radar @ r_h.T).T
                r_cam_xyz_all   = r_cam_all[:, 0:3].astype(np.float32)

                front_r = r_cam_xyz_all[:, 2] > 0.1
                r_cam_xyz = r_cam_xyz_all[front_r]
                r_v_raw   = r_v_raw[front_r]

                if len(r_cam_xyz) > 0:
                    bearing = np.arctan2(r_cam_xyz[:, 0], r_cam_xyz[:, 2])
                    r_vx    = (r_v_raw * np.sin(bearing)).astype(np.float32)
                    r_vy    = np.zeros(len(r_cam_xyz), dtype=np.float32)

                    u_r, v_r, valid_math_r = self.project_to_pixels(r_cam_xyz)
                    in_frame_r = (valid_math_r & (u_r >= 0) & (u_r < img_w) & (v_r >= 0) & (v_r < img_h))
                    r_cam_xyz = r_cam_xyz[in_frame_r]
                    r_vx      = r_vx[in_frame_r]
                    r_vy      = r_vy[in_frame_r]
                    u_r       = u_r[in_frame_r]
                    v_r       = v_r[in_frame_r]
                    hit_ids_r = self.latest_mask[v_r, u_r]
                    has_radar = True

        all_frustum_points = []
        
        frustum_count = 0
        for i, detection in enumerate(self.latest_bbox_msg.detections):
            target_mask_id = i + 1   

            yolo_coco_class = -1
            if detection.results:
                try:
                    yolo_coco_class = int(detection.results[0].hypothesis.class_id)
                except (ValueError, AttributeError):
                    pass
            fusion_class = YOLO_TO_FUSION_CLASS.get(yolo_coco_class, -1)

            # ── TUNED ASYMMETRICAL BROADENING & SHIELDS ──────────────────────
            if fusion_class == 1:       # Pedestrian
                dynamic_margin = 0.8    # TIGHTENED: Prevents swallowing the car behind them
                gap_m = 0.4             # TIGHTENED: Forces split between person and car roof
                dilation_kernel = 5     # Minor widening for edge parallax only
            elif fusion_class == 2:     # Bicycle
                dynamic_margin = 1.5    
                gap_m = 0.5
                dilation_kernel = 5
            elif fusion_class == 3:     # Truck
                dynamic_margin = 8.0
                gap_m = 0.5
                dilation_kernel = 15
            else:                       # Car (0) or Unknown
                dynamic_margin = 4.5
                gap_m = 0.3             # (Not used due to macro-gap shield override)
                dilation_kernel = 20    # AGGRESSIVE: Expands mask into pedestrian to steal hidden car points
                
            obj_mask_2d = (self.latest_mask == target_mask_id).astype(np.uint8)
            if dilation_kernel > 0:
                kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
                broadened_mask_2d = cv2.dilate(obj_mask_2d, kernel, iterations=1)
                mask_l = broadened_mask_2d[v_l, u_l] > 0
            else:
                mask_l = (hit_ids_l == target_mask_id)

            box_l_xyz    = l_cam_xyz[mask_l]
            box_l_intens = l_intens[mask_l]

            if len(box_l_xyz) < self.MIN_LIDAR_PTS:
                continue

            req_span = 0.0 if fusion_class in [1, 2] else self.MIN_HEIGHT_SPAN
            y_span = float(np.max(box_l_xyz[:, 1]) - np.min(box_l_xyz[:, 1]))
            if y_span < req_span:
                continue

            box_l_xyz, box_l_intens = self.filter_clusters_by_class(box_l_xyz, box_l_intens, fusion_class, gap_m)

            if len(box_l_xyz) < self.MIN_LIDAR_PTS:
                continue

            front_z   = float(np.min(box_l_xyz[:, 2]))
            fg_mask_l = box_l_xyz[:, 2] <= (front_z + dynamic_margin)
            box_l_xyz    = box_l_xyz[fg_mask_l]
            box_l_intens = box_l_intens[fg_mask_l]

            if len(box_l_xyz) < self.MIN_LIDAR_PTS:
                continue

            lidar_7d = np.zeros((len(box_l_xyz), 7), dtype=np.float32)
            lidar_7d[:, 0:3] = box_l_xyz
            lidar_7d[:, 3]   = box_l_intens

            radar_7d = np.zeros((0, 7), dtype=np.float32)
            if has_radar and len(hit_ids_r) > 0:
                if dilation_kernel > 0:
                    mask_r = broadened_mask_2d[v_r, u_r] > 0
                else:
                    mask_r = (hit_ids_r == target_mask_id)
                    
                if np.any(mask_r):
                    box_r_xyz = r_cam_xyz[mask_r]
                    box_r_vx  = r_vx[mask_r]
                    box_r_vy  = r_vy[mask_r]

                    fg_mask_r = box_r_xyz[:, 2] <= (front_z + dynamic_margin)
                    box_r_xyz = box_r_xyz[fg_mask_r]
                    box_r_vx  = box_r_vx[fg_mask_r]
                    box_r_vy  = box_r_vy[fg_mask_r]

                    if len(box_r_xyz) > 0:
                        radar_7d = np.zeros((len(box_r_xyz), 7), dtype=np.float32)
                        radar_7d[:, 0:3] = box_r_xyz
                        radar_7d[:, 4]   = box_r_vx
                        radar_7d[:, 5]   = box_r_vy
                        radar_7d[:, 6]   = 1.0

            fused = (np.vstack((lidar_7d, radar_7d)) if len(radar_7d) > 0 else lidar_7d)
            final = self.radar_preserving_sample(fused, 1024)

            all_frustum_points.append(final[:, 0:4])

            header  = np.array([float(fusion_class), 0.0], dtype=np.float32)
            payload = np.concatenate([header, final.flatten()])

            out_msg      = Float32MultiArray()
            out_msg.data = payload.tolist()
            self.frustum_pub.publish(out_msg)
            frustum_count += 1

        if all_frustum_points:
            combined_points = np.vstack(all_frustum_points)
            
            pc_header = Header()
            pc_header.stamp = self.get_clock().now().to_msg()
            pc_header.frame_id = 'cam0' 
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            
            pc_msg = pc2.create_cloud(pc_header, fields, combined_points.tolist())
            self.frustum_pc_pub.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LiveFrustumExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()