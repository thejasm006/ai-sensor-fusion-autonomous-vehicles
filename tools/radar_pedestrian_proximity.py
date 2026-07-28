"""
radar_pedestrian_proximity.py  —  Does ANY raw radar point land near the pedestrian?
======================================================================================
The inspector proved zero radar points reach any frustum. This script answers the
deeper question: is that because the moving-object filter (|v|>0.05) is rejecting a
real-but-slow return near the pedestrian (tangential-motion theory), or because radar
essentially sees nothing there at all (weak return / small radar cross-section)?

Method:
  1. Grab one synced camera+LiDAR frame, run YOLO, find the person box.
  2. Project LiDAR into our VALIDATED cam0 frame, take points inside that box,
     compute their centroid -> "where the pedestrian actually is."
  3. Read the RAW /PointCloudDetection topic directly (UNFILTERED -- bypassing the
     |v|>0.05 cut entirely) for the same moment.
  4. Transform every raw radar point into the SAME validated cam0 frame.
  5. Find the closest radar point(s) to the pedestrian centroid and report their
     distance AND velocity.

Interpretation:
  - Close radar point (within ~1-2m) with |v| just under 0.05  -> tangential motion
    theory confirmed: radar sees them, but their radial speed barely fails the cut.
  - No radar point anywhere near                                -> weak/no return,
    a different limitation (small human radar cross-section), not a filter issue.

Run with:   python3 radar_pedestrian_proximity.py
"""
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAG_DIR = "/root/data/studentProject"

K = np.array([
    [880.50024627982475, 0.0,                926.39074141015010],
    [0.0,                878.72560881910874, 578.76132391886370],
    [0.0,                0.0,                1.0]
], dtype=np.float64)

# velodyne <- cam0 (from tf_static, validated)
T_cam0_to_velodyne = np.array([
    [-0.0212, -0.1616,  0.9866,  0.6920],
    [-0.9998, -0.0017, -0.0218,  0.0000],
    [ 0.0052, -0.9869, -0.1615, -0.1800],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float64)
T_velodyne_to_cam0 = np.linalg.inv(T_cam0_to_velodyne)

# velodyne <- radar (from tf_static, Transform 4)
def quat_to_matrix(qx, qy, qz, qw, tx, ty, tz):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ])
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [tx, ty, tz]
    return T

T_velodyne_from_radar = quat_to_matrix(0.0002, 0.0075, -0.0310, 0.9995,
                                        1.5630, 0.0000, -0.8670)
# Chain: radar -> velodyne -> cam0
T_cam0_from_radar = T_velodyne_to_cam0 @ T_velodyne_from_radar


def read_pointcloud2_xyz_intensity(msg):
    field = {f.name: f for f in msg.fields}
    point_step = msg.point_step
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    n = msg.width * msg.height
    raw = raw.reshape(n, point_step)
    def gf(name):
        off = field[name].offset
        return raw[:, off:off+4].copy().view(np.float32).reshape(-1)
    x, y, z = gf('x'), gf('y'), gf('z')
    intensity = gf('intensity') if 'intensity' in field else np.zeros(n, dtype=np.float32)
    pts = np.stack([x, y, z, intensity], axis=1)
    return pts[np.isfinite(pts).all(axis=1)]


def read_pointcloud2_xyzv(msg):
    field = {f.name: f for f in msg.fields}
    point_step = msg.point_step
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    n = msg.width * msg.height
    raw = raw.reshape(n, point_step)
    def gf(name):
        off = field[name].offset
        return raw[:, off:off+4].copy().view(np.float32).reshape(-1)
    x, y, z, v = gf('x'), gf('y'), gf('z'), gf('v')
    pts = np.stack([x, y, z, v], axis=1)
    return pts[np.isfinite(pts).all(axis=1)]


def decode_image(msg):
    h, w = msg.height, msg.width
    enc = msg.encoding
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ('rgb8', 'bgr8'):
        image = data.reshape(h, w, 3)
        if enc == 'rgb8':
            image = image[:, :, ::-1]
    else:
        import cv2
        if enc.startswith('bayer'):
            image = cv2.cvtColor(data.reshape(h, w), cv2.COLOR_BAYER_RG2BGR)
        elif enc == 'mono8':
            image = cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)
        else:
            ch = data.size // (h * w)
            image = data.reshape(h, w, ch)[:, :, :3]
    return np.ascontiguousarray(image)


def main():
    print("=" * 64)
    print("RADAR <-> PEDESTRIAN PROXIMITY CHECK")
    print("=" * 64)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    MAX_FRAMES_TO_SCAN = 250  # studentProject has 235 camera frames total -- scan all of them

    with Reader(Path(BAG_DIR)) as reader:
        cam_conns = [c for c in reader.connections if c.topic == '/blackfly_s/cam0/image_rectified']
        lidar_conns = [c for c in reader.connections if c.topic == '/velodyne/points_raw']
        radar_conns = [c for c in reader.connections if c.topic == '/PointCloudDetection']

        # Pre-load lidar and raw radar messages with timestamps for nearest-match
        lidar_msgs = [(ts, conn, raw) for conn, ts, raw in reader.messages(connections=lidar_conns)]
        radar_msgs = [(ts, conn, raw) for conn, ts, raw in reader.messages(connections=radar_conns)]

        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')

        image = None
        points_lidar = None
        points_radar_raw = None
        person_box = None
        frame_no = 0

        for conn, ts, raw in reader.messages(connections=cam_conns):
            frame_no += 1
            if frame_no > MAX_FRAMES_TO_SCAN:
                break

            candidate_image = decode_image(typestore.deserialize_cdr(raw, conn.msgtype))
            results = model(candidate_image, verbose=False)[0]

            found = None
            for box in results.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.3:  # person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    found = (x1, y1, x2, y2, float(box.conf[0]))
                    break

            print(f"  Frame {frame_no}: {'>>> PERSON FOUND, conf=' + f'{found[4]:.2f}' + ' <<<' if found else 'no person'}"
                  if (found is not None or frame_no % 10 == 0) else "", end="")
            if found is not None or frame_no % 10 == 0:
                print()

            if found is not None:
                image = candidate_image
                person_box = found
                best_lidar = min(lidar_msgs, key=lambda m: abs(m[0] - ts))
                points_lidar = read_pointcloud2_xyz_intensity(
                    typestore.deserialize_cdr(best_lidar[2], best_lidar[1].msgtype))
                best_radar = min(radar_msgs, key=lambda m: abs(m[0] - ts))
                points_radar_raw = read_pointcloud2_xyzv(
                    typestore.deserialize_cdr(best_radar[2], best_radar[1].msgtype))
                break

    if person_box is None:
        print()
        print(f"No person detected by YOLO in any of the first {MAX_FRAMES_TO_SCAN} frames.")
        print("The 70-message inspector result remains the statistically solid finding.")
        return

    print()
    print(f"LiDAR points: {len(points_lidar)}")
    print(f"RAW radar points (unfiltered): {len(points_radar_raw)}")
    print()

    print("STEP 1: Locating the pedestrian via YOLO + LiDAR")
    print("-" * 60)
    x1, y1, x2, y2, conf = person_box
    print(f"  Person box: conf={conf:.2f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

    # Project LiDAR into validated cam0 frame
    xyz = points_lidar[:, 0:3]
    n = xyz.shape[0]
    hom = np.hstack([xyz, np.ones((n, 1))])
    cam_xyz = (T_velodyne_to_cam0 @ hom.T).T[:, 0:3]
    cam_xyz[:, 1] = -cam_xyz[:, 1]  # validated Y-flip
    front = cam_xyz[:, 2] > 0.1
    cam_xyz_f = cam_xyz[front]

    u = (K[0, 0] * cam_xyz_f[:, 0] / cam_xyz_f[:, 2]) + K[0, 2]
    v = (K[1, 1] * cam_xyz_f[:, 1] / cam_xyz_f[:, 2]) + K[1, 2]
    margin = 5
    inside = (u >= x1-margin) & (u <= x2+margin) & (v >= y1-margin) & (v <= y2+margin)
    person_lidar_pts = cam_xyz_f[inside]

    if len(person_lidar_pts) == 0:
        print("  No LiDAR points landed inside the person box. Can't locate them.")
        return

    pedestrian_centroid = np.mean(person_lidar_pts, axis=0)
    print(f"  Pedestrian LiDAR points: {len(person_lidar_pts)}")
    print(f"  Pedestrian centroid (cam0, our validated frame): "
          f"x={pedestrian_centroid[0]:.2f}  y={pedestrian_centroid[1]:.2f}  "
          f"z={pedestrian_centroid[2]:.2f}  (metres)")
    print()

    # --- Step 2: transform ALL raw radar points into the SAME cam0 frame ---
    print("STEP 2: Transforming raw radar points into the same cam0 frame")
    print("-" * 60)
    r_xyz = points_radar_raw[:, 0:3]
    r_v = points_radar_raw[:, 3]
    nr = r_xyz.shape[0]
    r_hom = np.hstack([r_xyz, np.ones((nr, 1))])
    r_cam = (T_cam0_from_radar @ r_hom.T).T[:, 0:3]
    r_cam[:, 1] = -r_cam[:, 1]  # same validated Y-flip, for consistency

    distances = np.linalg.norm(r_cam - pedestrian_centroid, axis=1)
    order = np.argsort(distances)

    print(f"  {len(r_xyz)} raw radar points checked.")
    print()
    print("  Closest 5 radar points to the pedestrian:")
    print(f"  {'dist(m)':>8} {'velocity(m/s)':>14} {'cam0 x,y,z':>30}")
    for i in order[:5]:
        d = distances[i]
        vel = r_v[i]
        x, y, z = r_cam[i]
        flag = "  <-- within filter threshold!" if abs(vel) > 0.05 else ""
        print(f"  {d:8.2f} {vel:14.3f}   ({x:6.2f},{y:6.2f},{z:6.2f}){flag}")

    print()
    print("=" * 60)
    closest_dist = distances[order[0]]
    closest_vel = r_v[order[0]]
    if closest_dist < 2.0:
        print(f"FOUND a radar point {closest_dist:.2f}m from the pedestrian, "
              f"velocity={closest_vel:.3f} m/s.")
        if abs(closest_vel) <= 0.05:
            print("This SUPPORTS the tangential-motion theory: radar likely sees")
            print("them, but the radial speed component is too small to clear")
            print("the |v|>0.05 moving-object filter.")
        else:
            print("This point would have cleared the filter -- worth checking why")
            print("it didn't end up classified as 'moving' in the preprocessed topic.")
    else:
        print(f"Closest radar point is {closest_dist:.2f}m away -- not really 'near'")
        print("the pedestrian. This leans toward a weak/no-return explanation")
        print("(small radar cross-section) rather than a filtering issue.")
    print("=" * 60)


if __name__ == "__main__":
    main()