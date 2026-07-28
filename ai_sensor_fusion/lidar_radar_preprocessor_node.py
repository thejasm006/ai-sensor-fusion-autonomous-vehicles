import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np

class SpatialPreprocessor(Node):
    def __init__(self):
        super().__init__('spatial_preprocessor')
        
        # ==========================================
        # LiDAR Pub/Sub Setup
        # ==========================================
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/velodyne/points_raw', 
            self.lidar_callback,
            10)
            
        self.lidar_pub = self.create_publisher(PointCloud2, '/filtered_points', 10)
        
        # ==========================================
        # Radar Pub/Sub Setup
        # ==========================================
        self.radar_sub = self.create_subscription(
            PointCloud2, 
            '/PointCloudDetection', 
            self.radar_callback, 
            10)
            
        self.radar_pub = self.create_publisher(PointCloud2, '/radar/preprocessed', 10)

        self.get_logger().info('🚀 Unified Spatial Preprocessor (LiDAR + Radar) Started!')

    # =========================================================
    # 1. LiDAR Preprocessing Callback
    # =========================================================
    def lidar_callback(self, msg):
        # 1. Read points including the 'intensity' field
        pt_list = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity")))
        
        if len(pt_list) == 0:
            return

        # 2. Extract XYZ and Intensity into separate arrays
        raw_data = np.array([(float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3])) for pt in pt_list], dtype=np.float32)
        xyz = raw_data[:, :3]
        intensity = raw_data[:, 3]

        # 3. Instantiate Open3D cloud and map intensity into the color channel
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        # Open3D colors expect normalized ranges [0, 1]
        max_intensity = np.max(intensity) if np.max(intensity) > 0 else 1.0
        normalized_intensity = intensity / max_intensity
        colors = np.vstack([normalized_intensity, normalized_intensity, normalized_intensity]).T
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 4. ROI Filtering (Crop the world)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(0, -10, -1.5), 
            max_bound=(40, 10, 3.0)
        )
        pcd = pcd.crop(bbox)

        # 5. Voxel Downsampling
        pcd = pcd.voxel_down_sample(voxel_size=0.1)

        # 6. Z-Threshold Ground Removal (Replaces RANSAC)
        if len(pcd.points) > 0:
            ground_threshold = -1.5 
            points_np = np.asarray(pcd.points)
            
            # Create a boolean mask: True for points ABOVE the ground threshold
            object_mask = points_np[:, 2] > ground_threshold
            
            # Apply mask to select object points 
            # (Open3D automatically slices the .colors array to match, preserving intensity)
            pcd_objects = pcd.select_by_index(np.where(object_mask)[0])

            # 7. Re-extract geometry and original un-normalized intensity values
            out_xyz = np.asarray(pcd_objects.points, dtype=np.float32)
            out_colors = np.asarray(pcd_objects.colors, dtype=np.float32)
            
            if len(out_xyz) == 0:
                return

            # Re-scale intensity back to the original dynamic scale
            out_intensity = out_colors[:, 0] * max_intensity
            
            # Create structured array with proper field layout
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
            ]
            
            # Create output data as structured array with correct layout (16 bytes per point)
            out_data = np.zeros((len(out_xyz),), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)])
            out_data['x'] = out_xyz[:, 0]
            out_data['y'] = out_xyz[:, 1]
            out_data['z'] = out_xyz[:, 2]
            out_data['intensity'] = out_intensity
            
            # Reconstruct PointCloud2 with explicit field mapping
            out_msg = pc2.create_cloud(msg.header, fields, out_data)
            self.lidar_pub.publish(out_msg)

    # =========================================================
    # 2. Radar Preprocessing Callback
    # =========================================================
    def radar_callback(self, msg):
        # 1. Read raw radar points (extracting velocity 'v')
        raw_radar = list(pc2.read_points(msg, field_names=("x", "y", "z", "v"), skip_nans=True))
        
        if len(raw_radar) == 0:
            return
            
        radar_points = np.array([(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in raw_radar], dtype=np.float32)

        # 2. PREPROCESSING: Static Clutter Filtering
        # Extract velocity ('v') and only keep points moving faster than 0.5 m/s
        velocities = radar_points[:, 3]
        moving_mask = np.abs(velocities) > 0.05  # Threshold can be tuned based on sensor noise characteristics
        filtered_radar = radar_points[moving_mask]

        if len(filtered_radar) == 0:
            return

        # 3. Publish the clean, preprocessed Radar points
        # Define the custom fields (x, y, z, v) to pack them back into a ROS message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='v', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Create output data as structured numpy array to ensure clean memory mapping
        out_data = np.zeros((len(filtered_radar),), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('v', np.float32)])
        out_data['x'] = filtered_radar[:, 0]
        out_data['y'] = filtered_radar[:, 1]
        out_data['z'] = filtered_radar[:, 2]
        out_data['v'] = filtered_radar[:, 3]
        
        # Create and publish the new cloud using the exact same timestamp as the original
        out_msg = pc2.create_cloud(msg.header, fields, out_data)
        self.radar_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SpatialPreprocessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()