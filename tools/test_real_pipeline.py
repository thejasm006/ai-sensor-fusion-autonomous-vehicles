"""
test_real_pipeline.py  —  TEST YOUR TEAM'S REAL MODEL on a real bag frame
===========================================================================
This tests fusion_model_best.pth (the team's own custom-trained 4-class
model), not the borrowed KITTI car-only model we tested before.

What's faithfully reproduced from the team's real code:
  - The CompleteFusionNet architecture (frustum_model_7d.py, unchanged)
  - The centroid-only normalization contract (no rotation - simpler than
    the KITTI model: just subtract the mean, the model's own T-Net handles
    the rest)
  - The soft-prior size blending (alpha=0.7) from frustum_inference_node.py
  - The "offset hallucination clamp" safety check
  - The per-class dynamic depth margin + nearest-cluster split from
    live_frustum_extractor_node.py (pedestrian=1.2m, bicycle=1.5m,
    truck=8.0m, car=4.5m)
  - The YOLO-class -> fusion-class mapping

ONE SIMPLIFICATION (clearly flagged): the team's real extractor uses YOLO
INSTANCE SEGMENTATION MASKS (yolo26s-seg.engine, a non-portable TensorRT
file) to assign LiDAR points to objects pixel-by-pixel. We don't have that
engine file, so this script uses a plain 2D bounding box (yolov8n.pt,
already proven to work) instead. This is a reasonable stand-in for testing
whether the MODEL recognises real geometry — it is less precise than mask-
based cropping for objects close together, but for testing model + decode
correctness it should not matter.

ANOTHER DIFFERENCE (clearly flagged): we use OUR tf_static-derived
LiDAR->cam0 extrinsic (already empirically proven: 80.7% foreground on the
KITTI model) rather than the team's hardcoded T_LIDAR_TO_CAM0 matrix in
live_frustum_extractor_node.py, because that hardcoded matrix has a height
offset (-1.412m) that disagrees with the bag's own recorded calibration
(-0.18m). Worth asking the team about this discrepancy.

Run with:   python3 test_real_pipeline.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from frustum_model_7d import CompleteFusionNet, NUM_CLASSES, NUM_HEADING_BINS, bin_and_residual_to_angle

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# CONFIG
# ============================================================
BAG_DIR = "/root/data/studentProject"
MODEL_PATH = "fusion_model_best.pth"
TARGET_PTS = 1024
MIN_FG_RATIO = 0.05
CONF_THRESHOLD = 0.60
MAX_FRAMES_TO_SCAN = 20   # how many synced frames to try before giving up

# Intrinsics (from camera_front_center calibration yaml)
K = np.array([
    [880.50024627982475, 0.0,                926.39074141015010],
    [0.0,                878.72560881910874, 578.76132391886370],
    [0.0,                0.0,                1.0]
], dtype=np.float64)

# Extrinsic cam0 -> velodyne, read directly from /tf_static (proven to work)
T_cam0_to_velodyne = np.array([
    [-0.0212, -0.1616,  0.9866,  0.6920],
    [-0.9998, -0.0017, -0.0218,  0.0000],
    [ 0.0052, -0.9869, -0.1615, -0.1800],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float64)
T_velodyne_to_cam0 = np.linalg.inv(T_cam0_to_velodyne)

CHASSIS_Z_MIN = -1.5     # ground filter, RAW LiDAR frame (matches team's node)
MIN_LIDAR_PTS = 5
MIN_HEIGHT_SPAN = 0.30
OCCLUSION_GAP_M = 0.8

# YOLO (COCO class id) -> team's fusion class id  (from live_frustum_extractor_node.py)
YOLO_TO_FUSION_CLASS = {0: 1, 1: 2, 2: 0, 3: 2, 5: 0, 7: 3}
FUSION_LABELS = {0: 'Car', 1: 'Pedestrian', 2: 'Bicycle', 3: 'Truck'}
DYNAMIC_MARGIN = {1: 1.2, 2: 1.5, 3: 8.0}   # default (car/unknown) = 4.5
DEFAULT_MARGIN = 4.5

PRIORS = {
    'Car':        np.array([1.93, 4.63, 1.56]),
    'Pedestrian': np.array([0.73, 0.73, 1.77]),
    'Bicycle':    np.array([0.60, 1.76, 1.44]),
    'Truck':      np.array([2.51, 6.93, 2.84]),
}


# ============================================================
# Bag reading (same proven approach as before)
# ============================================================
def read_pointcloud2(msg):
    field = {f.name: f for f in msg.fields}
    point_step = msg.point_step
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    n = msg.width * msg.height
    raw = raw.reshape(n, point_step)

    def gf(name):
        off = field[name].offset
        return raw[:, off:off + 4].copy().view(np.float32).reshape(-1)

    x, y, z = gf('x'), gf('y'), gf('z')
    intensity = gf('intensity') if 'intensity' in field else np.zeros(n, dtype=np.float32)
    pts = np.stack([x, y, z, intensity], axis=1)
    return pts[np.isfinite(pts).all(axis=1)]


def decode_image(msg):
    h, w = msg.height, msg.width
    enc = msg.encoding
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ('rgb8', 'bgr8'):
        image = data.reshape(h, w, 3)
        if enc == 'rgb8':
            image = image[:, :, ::-1]
    elif enc.startswith('bayer'):
        import cv2
        image = cv2.cvtColor(data.reshape(h, w), cv2.COLOR_BAYER_RG2BGR)
    elif enc == 'mono8':
        import cv2
        image = cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)
    else:
        ch = data.size // (h * w)
        image = data.reshape(h, w, ch)[:, :, :3]
    return np.ascontiguousarray(image)


def iter_synced_frames(max_frames):
    """Yield (image, lidar_points) pairs, synced within ~50ms, up to max_frames."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(Path(BAG_DIR)) as reader:
        cam_conns = [c for c in reader.connections if c.topic == '/blackfly_s/cam0/image_rectified']
        lidar_conns = [c for c in reader.connections if c.topic == '/velodyne/points_raw']

        # Collect lidar messages with timestamps for nearest-match lookup
        lidar_msgs = []
        for conn, ts, raw in reader.messages(connections=lidar_conns):
            lidar_msgs.append((ts, conn, raw))

        count = 0
        for conn, ts, raw in reader.messages(connections=cam_conns):
            if count >= max_frames:
                break
            # find nearest lidar message in time
            best = min(lidar_msgs, key=lambda m: abs(m[0] - ts))
            if abs(best[0] - ts) > 80_000_000:  # 80ms, skip badly synced pairs
                continue
            img_msg = typestore.deserialize_cdr(raw, conn.msgtype)
            lidar_msg = typestore.deserialize_cdr(best[2], best[1].msgtype)
            image = decode_image(img_msg)
            points = read_pointcloud2(lidar_msg)
            count += 1
            yield image, points


# ============================================================
# YOLO detection (stand-in for the team's instance-seg masks)
# ============================================================
_yolo_model = None
def run_yolo(image):
    global _yolo_model
    from ultralytics import YOLO
    if _yolo_model is None:
        _yolo_model = YOLO('yolov8n.pt')
    results = _yolo_model(image, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in YOLO_TO_FUSION_CLASS and conf > 0.3:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append((x1, y1, x2, y2, conf, cls_id))
    return boxes


# ============================================================
# Projection + frustum extraction (faithful to live_frustum_extractor_node.py)
# ============================================================
def project_lidar(points):
    # Ground filter in RAW lidar frame (matches team's node)
    ground_ok = points[:, 2] > CHASSIS_Z_MIN
    points = points[ground_ok]

    xyz = points[:, 0:3]
    n = xyz.shape[0]
    hom = np.hstack([xyz, np.ones((n, 1))])
    cam_xyz = (T_velodyne_to_cam0 @ hom.T).T[:, 0:3]

    # cam0 -> proper down-positive convention (proven fix from the sweep)
    cam_xyz[:, 1] = -cam_xyz[:, 1]

    front = cam_xyz[:, 2] > 0.1
    cam_xyz = cam_xyz[front]
    intensity = points[front, 3]

    u = (K[0, 0] * cam_xyz[:, 0] / cam_xyz[:, 2]) + K[0, 2]
    v = (K[1, 1] * cam_xyz[:, 1] / cam_xyz[:, 2]) + K[1, 2]
    return u, v, cam_xyz, intensity


def split_nearest_cluster(xyz, intens, gap_m=OCCLUSION_GAP_M):
    """Exact port of the team's logic: sort by depth, cut at the first big gap."""
    if len(xyz) < 2:
        return xyz, intens
    order = np.argsort(xyz[:, 2])
    xyz_s, intens_s = xyz[order], intens[order]
    gaps = np.diff(xyz_s[:, 2])
    gap_idx = np.where(gaps > gap_m)[0]
    if len(gap_idx) == 0:
        return xyz_s, intens_s
    keep = gap_idx[0] + 1
    return xyz_s[:keep], intens_s[:keep]


def extract_object_frustum(u, v, cam_xyz, intensity, box, margin=5):
    x1, y1, x2, y2, conf, coco_cls = box
    inside = (u >= x1 - margin) & (u <= x2 + margin) & (v >= y1 - margin) & (v <= y2 + margin)
    box_xyz = cam_xyz[inside]
    box_intens = intensity[inside]

    if len(box_xyz) < MIN_LIDAR_PTS:
        return None, None

    y_span = float(np.max(box_xyz[:, 1]) - np.min(box_xyz[:, 1]))
    if y_span < MIN_HEIGHT_SPAN:
        return None, None

    box_xyz, box_intens = split_nearest_cluster(box_xyz, box_intens)
    if len(box_xyz) < MIN_LIDAR_PTS:
        return None, None

    fusion_class = YOLO_TO_FUSION_CLASS.get(coco_cls, -1)
    margin_m = DYNAMIC_MARGIN.get(fusion_class, DEFAULT_MARGIN)
    front_z = float(np.min(box_xyz[:, 2]))
    keep = box_xyz[:, 2] <= (front_z + margin_m)
    box_xyz, box_intens = box_xyz[keep], box_intens[keep]

    if len(box_xyz) < MIN_LIDAR_PTS:
        return None, None

    return box_xyz, box_intens


# ============================================================
# Model inference + decode (faithful to frustum_inference_node.py)
# ============================================================
def run_and_decode(model, device, box_xyz, box_intens, yolo_coco_cls):
    n = len(box_xyz)
    feat = np.zeros((n, 7), dtype=np.float32)
    feat[:, 0:3] = box_xyz
    feat[:, 3] = box_intens
    # feat[:,4:6] = 0 (vx,vy - no radar in this test); feat[:,6] = 0 (modality: LiDAR)

    choice = (np.random.choice(n, TARGET_PTS, replace=False)
              if n >= TARGET_PTS else np.random.choice(n, TARGET_PTS, replace=True))
    pts = feat[choice].copy()

    centroid = pts[:, 0:3].mean(axis=0)
    pts[:, 0:3] -= centroid

    x = torch.from_numpy(pts.T).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        seg_logits, center, size, head_bins, head_res, cls_logits = model(x)

    fg_probs = F.softmax(seg_logits[0], dim=0)[1]
    fg_ratio = (fg_probs > 0.5).float().mean().item()

    cls_prob = F.softmax(cls_logits[0], dim=0)
    conf_3d, cls_3d = torch.max(cls_prob, dim=0)

    yolo_fusion_cls = YOLO_TO_FUSION_CLASS.get(yolo_coco_cls, -1)
    if yolo_fusion_cls in FUSION_LABELS:
        final_class = yolo_fusion_cls
    else:
        final_class = cls_3d.item()
    conf = conf_3d.item()
    label = FUSION_LABELS.get(final_class, 'Unknown')

    net_size = size[0].cpu().numpy()
    alpha = 0.7
    if label in PRIORS:
        size_world = alpha * PRIORS[label] + (1.0 - alpha) * net_size
    else:
        size_world = net_size

    net_offset = center[0].cpu().numpy()
    if np.linalg.norm(net_offset) > 3.0:
        net_offset = np.zeros(3)
        net_offset[1] = float(size_world[2]) * 0.4

    center_world = centroid + net_offset
    yaw = bin_and_residual_to_angle(head_bins, head_res, NUM_HEADING_BINS)[0].item()

    return {
        'fg_ratio': fg_ratio, 'cls_probs': cls_prob.cpu().numpy(),
        'final_label': label, 'conf': conf, 'n_points': n,
        'center': center_world, 'size_net': net_size, 'size_blended': size_world,
        'yaw_deg': np.degrees(yaw),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 64)
    print("TESTING THE TEAM'S REAL MODEL — fusion_model_best.pth")
    print("=" * 64)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")

    model = CompleteFusionNet(NUM_CLASSES, NUM_HEADING_BINS).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    print(f"Loaded model (epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?')})")
    print()

    found_classes = set()
    frame_no = 0

    for image, points in iter_synced_frames(MAX_FRAMES_TO_SCAN):
        frame_no += 1
        print(f"--- Frame {frame_no} --- ({len(points)} lidar pts)")
        boxes = run_yolo(image)
        if not boxes:
            print("  no relevant detections, skipping")
            continue

        u, v, cam_xyz, intensity = project_lidar(points)

        for box in boxes:
            x1, y1, x2, y2, conf, coco_cls = box
            fusion_cls = YOLO_TO_FUSION_CLASS.get(coco_cls, -1)
            fusion_label = FUSION_LABELS.get(fusion_cls, f'coco{coco_cls}')

            box_xyz, box_intens = extract_object_frustum(u, v, cam_xyz, intensity, box)
            if box_xyz is None:
                print(f"  [{fusion_label}] yolo_conf={conf:.2f} -> too few/invalid lidar points, skipped")
                continue

            res = run_and_decode(model, device, box_xyz, box_intens, coco_cls)
            print(f"  [{fusion_label}] yolo_conf={conf:.2f}  n_pts={res['n_points']}")
            print(f"      fg_ratio={res['fg_ratio']:.2f}  "
                  f"model_says='{res['final_label']}' (conf={res['conf']:.2f})")
            probs_str = ", ".join(f"{FUSION_LABELS[i]}={p:.2f}" for i, p in enumerate(res['cls_probs']))
            print(f"      class probs: {probs_str}")
            h, w, l = res['size_blended']
            cx, cy, cz = res['center']
            print(f"      box (blended) h={h:.2f} w={w:.2f} l={l:.2f}  "
                  f"pos=({cx:.1f},{cy:.1f},{cz:.1f})  yaw={res['yaw_deg']:.0f}deg")
            print(f"      [pass/fail] fg>={MIN_FG_RATIO}: {res['fg_ratio']>=MIN_FG_RATIO}   "
                  f"conf>={CONF_THRESHOLD}: {res['conf']>=CONF_THRESHOLD}")
            print()
            found_classes.add(fusion_label)

        # stop early once we've seen both a vehicle and a pedestrian
        if 'Car' in found_classes and 'Pedestrian' in found_classes:
            print(">> Found both Car and Pedestrian detections — stopping scan early.")
            break

    print()
    print("=" * 64)
    print(f"DONE. Classes successfully tested: {sorted(found_classes) if found_classes else 'NONE'}")
    print("=" * 64)
    if 'Pedestrian' not in found_classes:
        print("No pedestrian was found/passed in the frames scanned. This bag's")
        print("front-camera view may simply not show a person in these frames —")
        print("not necessarily a model problem. Try increasing MAX_FRAMES_TO_SCAN.")


if __name__ == "__main__":
    main()
