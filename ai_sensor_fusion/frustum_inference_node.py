import os
import math
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray

from .frustum_model_7d import (
    CompleteFusionNet,
    NUM_CLASSES,
    NUM_HEADING_BINS,
    bin_and_residual_to_angle,
)

TARGET_PTS    = 1024
MIN_FG_RATIO  = 0.05   


class LiveInferenceFusionNode(Node):

    def __init__(self):
        super().__init__('live_inference_fusion_node')

        self.frustum_sub = self.create_subscription(
            Float32MultiArray,
            '/fusion/ready_frustums',
            self.inference_callback,
            10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/fusion/ai_3d_boxes', 10)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = CompleteFusionNet(NUM_CLASSES, NUM_HEADING_BINS).to(self.device)

        default_weight = '/home/workspace/ai_training/saved_models/fusion_model_best.pth'
        weight_path    = self.declare_parameter('weight_path', default_weight).value

        if not os.path.exists(weight_path):
            self.get_logger().error(f'Weights not found: {weight_path}')
            raise FileNotFoundError(weight_path)

        ckpt = torch.load(weight_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            state = ckpt['model_state']
            self.get_logger().info(f"Loaded checkpoint dict")
        else:
            state = ckpt
            self.get_logger().info('Loaded bare state_dict checkpoint')

        self.model.load_state_dict(state)
        self.model.eval()

        self.conf_threshold = self.declare_parameter('conf_threshold', 0.6).value

        # ── VELODYNE TRANSFORMATION MATRIX ────────────────────────────────
        # Inverse of the hardcoded extractor matrix. 
        # Maps the flat cam0 predictions back to the native, un-pitched velodyne space.
        T_LIDAR_TO_CAM0 = np.array([
            [-0.021, -1.000,  0.005,  0.016],
            [-0.162, -0.002, -0.987, -0.066],
            [ 0.987, -0.022, -0.161, -0.712],
            [ 0.000,  0.000,  0.000,  1.000]
        ], dtype=np.float64)
        self.T_CAM0_TO_LIDAR = np.linalg.inv(T_LIDAR_TO_CAM0)

        self.class_labels = {0: 'Car', 1: 'Pedestrian', 2: 'Bicycle', 3: 'Truck'}
        self.class_colors = {
            'Car':        (0.0, 1.0, 0.0),
            'Pedestrian': (1.0, 0.5, 0.0),
            'Bicycle':    (0.0, 0.5, 1.0),
            'Truck':      (1.0, 0.0, 1.0),
        }

        self.marker_id = 0
        self.get_logger().info(f'✅ Inference node ready | Asphalt Clamp Enabled')

    def inference_callback(self, msg: Float32MultiArray):
        if not msg.data:
            return

        raw = np.array(msg.data, dtype=np.float32)

        yolo_class_id = -1
        if raw.size == TARGET_PTS * 7 + 2:
            yolo_class_id = int(raw[0])
            raw = raw[2:].reshape(-1, 7)
        elif raw.size == TARGET_PTS * 7:
            raw = raw.reshape(-1, 7)
        else:
            return

        if raw.shape[0] < 5:
            return   

        n      = raw.shape[0]
        choice = (np.random.choice(n, TARGET_PTS, replace=False)
                  if n >= TARGET_PTS
                  else np.random.choice(n, TARGET_PTS, replace=True))
        pts = raw[choice].copy()   

        centroid     = pts[:, 0:3].mean(axis=0)   
        pts[:, 0:3] -= centroid

        x = torch.from_numpy(pts.T).unsqueeze(0).to(self.device)  
        with torch.no_grad():
            seg_logits, center, size, head_bins, head_res, cls_logits = \
                self.model(x)

        fg_probs  = torch.softmax(seg_logits[0], dim=0)[1]    
        fg_ratio  = (fg_probs > 0.5).float().mean().item()
        if fg_ratio < MIN_FG_RATIO:
            return

        cls_prob        = torch.softmax(cls_logits[0], dim=0)
        conf_3d, cls_3d = torch.max(cls_prob, dim=0)

        # ── YOLO TRUST FIX ────────────────────────────────────────────────
        if yolo_class_id >= 0 and yolo_class_id in self.class_labels:
            final_class = yolo_class_id
            conf = 1.0   
        else:
            final_class = cls_3d.item()
            conf        = conf_3d.item()

        if conf < self.conf_threshold:
            return
            
        net_size = size[0].cpu().numpy()  
        label = self.class_labels.get(final_class, 'Unknown')

        priors = {
            'Car':        np.array([1.93, 4.63, 1.56]),
            'Pedestrian': np.array([0.73, 0.73, 1.77]),
            'Bicycle':    np.array([0.60, 1.76, 1.44]),
            'Truck':      np.array([2.51, 6.93, 2.84])
        }
        
        alpha = 0.7
        if label in priors:
            size_world = (alpha * priors[label]) + ((1.0 - alpha) * net_size)
        else:
            size_world = net_size

        # ── Domain Shift Offset Clamp ─────────────────────────────────────
        net_offset = center[0].cpu().numpy()
        if np.linalg.norm(net_offset) > 3.0:
            # If network panics, zero it out. The Asphalt Clamp below will handle height.
            net_offset = np.zeros(3)

        center_world = centroid + net_offset

        yaw = bin_and_residual_to_angle(
            head_bins, head_res, NUM_HEADING_BINS)[0].item()
        
        self.get_logger().info(
            f'[{label}] conf={conf:.2f} fg={fg_ratio:.2f} '
            f'center=({center_world[0]:.1f},{center_world[1]:.1f},{center_world[2]:.1f})\n'
            f'    net_WxLxH=({net_size[0]:.2f},{net_size[1]:.2f},{net_size[2]:.2f}) '
            f'blended_WxLxH=({size_world[0]:.2f},{size_world[1]:.2f},{size_world[2]:.2f}) '
            f'yaw={math.degrees(yaw):.0f}°')

        self.publish_marker(center_world, size_world, yaw, label)

    def publish_marker(self, center, size, yaw, label):
        markers = MarkerArray()

        # ── Map cam0 center back to Velodyne frame ────────────────────────
        center_cam0 = np.array([center[0], center[1], center[2], 1.0])
        center_velo = (self.T_CAM0_TO_LIDAR @ center_cam0)[:3]

        width  = float(max(0.1, abs(size[0])))
        length = float(max(0.1, abs(size[1])))
        height = float(max(0.1, abs(size[2])))

        # ── THE ASPHALT CLAMP ─────────────────────────────────────────────
        # The road surface is at Z = -1.5m in the LiDAR frame.
        # This absolutely guarantees the box cannot sink into the ground.
        box_bottom_z = center_velo[2] - (height / 2.0)
        if box_bottom_z < -1.5:
            # Physically push the center up so the bottom sits flush on the road
            center_velo[2] = -1.5 + (height / 2.0)

        m = Marker()
        # Publish directly to Velodyne to immune the box from the camera's TF pitch
        m.header.frame_id = 'velodyne'  
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns              = 'ai_3d_boxes'
        m.id              = self.marker_id
        m.type            = Marker.CUBE
        m.action          = Marker.ADD
        m.lifetime        = rclpy.duration.Duration(seconds=0.3).to_msg()

        m.pose.position.x = float(center_velo[0])
        m.pose.position.y = float(center_velo[1])
        m.pose.position.z = float(center_velo[2])

        # ── Velodyne Orientation ──────────────────────────────────────────
        yaw_velo = yaw
        half_yaw = yaw_velo * 0.5
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = float(math.sin(half_yaw))
        m.pose.orientation.w = float(math.cos(half_yaw))

        # ── Velodyne Scale Mapping ────────────────────────────────────────
        m.scale.x = length   # Forward
        m.scale.y = width    # Lateral
        m.scale.z = height   # Vertical

        r, g, b = self.class_colors.get(label, (1.0, 1.0, 1.0))
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.55
        markers.markers.append(m)

        # ── Text Label ────────────────────────────────────────────────────
        t = Marker()
        t.header   = m.header
        t.ns       = 'ai_labels'
        t.id       = self.marker_id + 5000
        t.type     = Marker.TEXT_VIEW_FACING
        t.action   = Marker.ADD
        t.lifetime = m.lifetime
        t.pose.position.x = float(center_velo[0])
        t.pose.position.y = float(center_velo[1])
        t.pose.position.z = float(center_velo[2]) + (height / 2.0) + 0.3 
        t.pose.orientation.w = 1.0
        t.scale.z = 0.4
        t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0; t.color.a = 1.0
        t.text = label
        markers.markers.append(t)

        self.marker_pub.publish(markers)
        self.marker_id = (self.marker_id + 1) % 5000


def main(args=None):
    rclpy.init(args=args)
    node = LiveInferenceFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()