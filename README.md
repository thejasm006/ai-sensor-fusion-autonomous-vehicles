# AI-Based 3D Sensor Fusion Stack for Autonomous Vehicles

![ROS 2 Humble](https://img.shields.io/badge/ROS2-Humble-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Instance_Segmentation-green) ![Platform](https://img.shields.io/badge/Platform-THI_CARISSMA_ANTON-red)

A real-time, multi-modal 3D object detection and tracking pipeline designed for automated vehicles. The system fuses 2D RGB camera streams, 3D LiDAR point clouds, and 4D imaging radar velocity vectors using a modified **Frustum-PointNet++** architecture (`CompleteFusionNet`) operating on **7D point tensors**.

Developed and validated on **ANTON** (a Renault Twizy automated research platform at THI CARISSMA).

---

## Key Features & Architecture
Camera (RGB) ──────► yolo_seg_node (YOLOv8-seg) ─────► 2D Instance Masks ┐
│
LiDAR (3D)  ──────► preprocessor_node (Open3D)  ─────► Filtered Cloud ───┼─► live_frustum_extractor_node
│   (7D Frustum Tensor Generator)
Radar (4D)  ──────► preprocessor_node (Velocity) ────► Moving Radar Pts  ┘                 │
▼
RViz2 ◄───── fusion_tracker_node ◄───── frustum_inference_node ◄───────────────── 7D Frustum Tensors
(3D Boxes)   (3D NMS & Tracking)       (CompleteFusionNet 3D Box)

1. **2D Instance Segmentation Guidance:** Utilizes YOLOv8-seg (`yolo26s-seg`) with ByteTrack persistent IDs to extract precise pixel-level object boundaries rather than loose bounding boxes.
2. **7D Tensor Point Augmentations:** Enriches 3D spatial points into 7-channel features:
   $$\text{Point Feature} = [X, Y, Z, \text{Intensity}, V_x, V_y, \text{Modality Flag}]$$
3. **Deep Learning 3D Box Regression:** Implements `CompleteFusionNet` (trained on nuScenes) featuring a 3D T-Net for spatial alignment, PointNet++ segmentation, and BoxNet size/heading regression across 4 classes (**Car**, **Pedestrian**, **Bicycle**, **Truck**).
4. **Occlusion-Induced Instability Mitigation:** Features mask dilation (kernel size: 20px) to recover occluded vehicle boundary points and a dedicated temporal tracking safety net (`fusion_tracker_node`).
5. **Radar Doppler Tangential Analysis:** Built-in analytical suite diagnosing Doppler radar velocity drop-offs during perpendicular target trajectories.

---

## Hardware & System Specifications

| Component | Specification |
|---|---|
| **Vehicle Platform** | ANTON — Renault Twizy Research Vehicle (THI CARISSMA) |
| **Primary Camera** | FLIR Blackfly S BFS-PGE-23S3 (`cam0` front-center, $1920 \times 1200$) |
| **3D LiDAR** | Velodyne VLP-16 (16-beam, 10 Hz) |
| **4D Radar** | Continental ARS548 / Radar-substitute stream (`x, y, z, v`) |
| **Software Stack** | ROS 2 Humble, Docker (`osrf/ros:humble-desktop`), PyTorch, Open3D, OpenCV |

---

## Pipeline Nodes Breakdown

### 1. `yolo_seg_node.py`
Publishes pixel-wise instance masks (`/yolo/instance_mask`) and 2D bounding boxes (`/yolo/detections`). Includes dynamic hood-exclusion filters to prevent ego-vehicle self-detection.

### 2. `lidar_radar_preprocessor_node.py`
Executes spatial Region-of-Interest (ROI) cropping ($X \in [0, 40\text{m}]$, $Y \in [-10, 10\text{m}]$, $Z \in [-1.5, 3.0\text{m}]$), Open3D voxel downsampling ($0.1\text{m}$), ground-plane removal, and radar moving-target extraction ($|v| > 0.05\text{ m/s}$).

### 3. `live_frustum_extractor_node.py`
Slices 3D point clouds based on camera-projected 2D masks. Utilizes mask dilation to handle partial vehicle occlusions and structures point fields into normalized 1024-point 7D feature tensors (`/fusion/ready_frustums`).

### 4. `frustum_inference_node.py`
Loads `fusion_model_best.pth` (`CompleteFusionNet`). Executes T-Net centroid translation, feature segmentation, 12-bin heading estimation, and soft-prior size blending ($0.7 \times \text{prior} + 0.3 \times \text{network\_output}$).

### 5. `fusion_tracker_node.py`
Applies 3D Non-Maximum Suppression (NMS) on the ground plane, filters cross-class proximity artifacts, suppresses implausible single-frame spatial jumps ($> 4.5\text{m}$), and maintains persistent object IDs across full physical occlusions.

---

## Analytical & Diagnostic Tools (`tools/`)

* **`watch_car_jump.py`**: Monitors raw frustum point cloud depth distributions during pedestrian crossings to isolate physical occlusion events.
* **`radar_pedestrian_proximity.py`**: Evaluates radar return density and Doppler velocity profiles relative to camera-detected targets. Confirmed that near-zero radial velocity ($0.001\text{ m/s}$) occurs during perpendicular pedestrian walking paths.
* **`read_tf_static.py`**: Directly decodes `/tf_static` rigid-body transformation matrices between sensor coordinate frames (`velodyne`, `cam0`, `radar`).

---

## Getting Started

### Prerequisites
* ROS 2 Humble
* PyTorch with CUDA support
* OpenCV, Open3D, `transforms3d`

### Installation & Build
```bash
# Clone the repository into your ROS 2 workspace
cd ~/ros2_ws/src
git clone [https://github.com/YOUR_GITHUB_USERNAME/ai-sensor-fusion-autonomous-vehicles.git](https://github.com/YOUR_GITHUB_USERNAME/ai-sensor-fusion-autonomous-vehicles.git) ai_sensor_fusion

# Install dependencies and build
cd ~/ros2_ws
colcon build --packages-select ai_sensor_fusion
source install/setup.bash


Running the Pipeline

# Terminal 1: Play ROS Bag
ros2 bag play /path/to/studentProject/

# Terminal 2: Run Full Fusion Stack
ros2 run ai_sensor_fusion yolo_seg_node
ros2 run ai_sensor_fusion lidar_radar_preprocessor_node
ros2 run ai_sensor_fusion live_frustum_extractor_node
ros2 run ai_sensor_fusion frustum_inference_node --ros-args -p weight_path:=/path/to/fusion_model_best.pth
ros2 run ai_sensor_fusion fusion_tracker_node

# Terminal 3: RViz Visualization
rviz2

Engineering Results:
Occlusion Suppression: Sustained continuous tracking on target vehicles across 3,300+ frames (over 5 minutes of continuous operation) without ID loss or position jumping during pedestrian crossings.

Coordinate Frame Alignment: Resolved coordinate frame conventions between camera-optical space ($Y\text{-down}$) and Velodyne coordinate space ($Z\text{-up}$), guaranteeing sub-centimeter alignment ($< 8\text{mm}$ error) across transformation pipelines.
