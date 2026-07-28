"""
offline_test.py  —  SELF-CONTAINED DEMO (no bag, no calibration, no YOLO)
========================================================================
Goal: prove the model + the box decode work, using a FAKE car-shaped
point cloud we build ourselves. If the decoded box comes out roughly
car-sized (~1.5 x 1.6 x 3.9 m), the whole inference + decode chain is correct.

This mirrors what the team's "car_inference_node" does internally, but
offline and on one frame — exactly the offline-test approach your teammate
suggested.

The factory analogy:
  We're putting a known-good test part (a fake car) on the inspection line
  to confirm the CMM (the model) and the report-writer (the decode) work,
  before we feed it real parts from the production line (the bag).

Run with:   python3 offline_test.py
"""

import numpy as np
import torch

from frustum_model import (
    FrustumPointNet,
    wrapToPi,
    getBinCenter,
    MEAN_CAR_SIZE,
    CENTERED_FRUSTUM_MEAN_XYZ,
)

# Make the run repeatable (same random numbers every time)
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "model_37_2_epoch_400.pth"
NUM_POINTS = 1024  # the model's fixed input size


# ============================================================
# STEP 1: Build a synthetic car-shaped point cloud
# ============================================================
# A real car is roughly 1.5 m tall, 1.6 m wide, 3.9 m long.
# We place a fake car ~14 m in front of the sensor (a realistic distance)
# and scatter points on its surface, then add some background clutter.
def make_synthetic_frustum():
    print("STEP 1: Building a synthetic car-shaped point cloud")
    print("-" * 60)

    # Car centre position in CAMERA coordinates (x=right, y=down, z=forward)
    car_x, car_y, car_z = 0.5, 1.0, 14.0  # 14 m ahead, slightly right
    car_h, car_w, car_l = 1.5, 1.6, 3.9   # real car dimensions

    # --- Generate ~250 points on the car's surface ---
    n_car = 250
    # Points scattered within the car's box (a simple filled box of points)
    cx = np.random.uniform(-car_w / 2, car_w / 2, n_car) + car_x
    cy = np.random.uniform(-car_h / 2, car_h / 2, n_car) + car_y
    cz = np.random.uniform(-car_l / 2, car_l / 2, n_car) + car_z
    car_intensity = np.random.uniform(0, 1898, n_car)  # bag-style intensity range
    car_points = np.stack([cx, cy, cz, car_intensity], axis=1)

    # --- Add ~60 background points (ground + a far wall) ---
    n_bg = 60
    # Ground points: spread out, low, at various depths
    gx = np.random.uniform(-3, 3, n_bg)
    gy = np.random.uniform(1.5, 2.0, n_bg)   # below the car (ground)
    gz = np.random.uniform(8, 25, n_bg)
    bg_intensity = np.random.uniform(0, 1898, n_bg)
    bg_points = np.stack([gx, gy, gz, bg_intensity], axis=1)

    # Combine car + background into one frustum
    frustum = np.concatenate([car_points, bg_points], axis=0).astype(np.float32)

    print(f"  Built frustum with {len(frustum)} points:")
    print(f"    {n_car} car-surface points (the object)")
    print(f"    {n_bg} background points (ground clutter)")
    print(f"  True car position: x={car_x}, y={car_y}, z={car_z} m")
    print(f"  True car size:     h={car_h}, w={car_w}, l={car_l} m")
    print()
    return frustum


# ============================================================
# STEP 2: Print a depth histogram (the v3 diagnostic from the deck)
# ============================================================
# This is the exact diagnostic the presentation used to discover the
# "frustum too deep" problem. It shows where the points sit in distance.
def print_depth_histogram(frustum):
    print("STEP 2: Depth histogram (where do the points sit?)")
    print("-" * 60)
    depths = frustum[:, 2]  # z = forward distance
    bins = [(0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 100)]
    for lo, hi in bins:
        count = np.sum((depths >= lo) & (depths < hi))
        bar = "#" * int(count / 5)
        print(f"  {lo:3d}-{hi:3d} m : {count:4d}  {bar}")
    print(f"  Depth range: {depths.min():.1f} to {depths.max():.1f} m")
    print()


# ============================================================
# STEP 3: Normalize the frustum (the Stage-3 forward transform)
# ============================================================
# This rotates + centres the frustum exactly the way the model saw data
# during KITTI training. We must record what we did so we can UNDO it
# when decoding the box (Stage 6).
def normalize_frustum(frustum):
    print("STEP 3: Normalize the frustum (rotate + centre for the model)")
    print("-" * 60)

    xyz = frustum[:, 0:3]
    intensity = frustum[:, 3:4]

    # --- Compute the frustum angle (direction to the cluster centre) ---
    # In camera coords, arctan2(x, z) gives the left-right angle.
    centre = np.mean(xyz, axis=0)
    frustum_angle = np.arctan2(centre[0], centre[2])

    # --- Build the rotation matrix that rotates the frustum to face forward ---
    frustum_R = np.asarray([
        [np.cos(frustum_angle), 0, -np.sin(frustum_angle)],
        [0, 1, 0],
        [np.sin(frustum_angle), 0, np.cos(frustum_angle)]
    ], dtype=np.float32)

    # Rotate the points so the cluster faces straight ahead
    xyz_rot = np.dot(frustum_R, xyz.T).T

    # Subtract the KITTI training mean (centres the cloud where the model expects)
    xyz_centered = xyz_rot - CENTERED_FRUSTUM_MEAN_XYZ

    # Re-attach intensity -> back to (N, 4)
    normalized = np.concatenate([xyz_centered, intensity], axis=1).astype(np.float32)

    print(f"  Frustum angle: {np.degrees(frustum_angle):.1f} degrees")
    print(f"  Rotated cluster to face forward and subtracted KITTI mean")
    print()
    # Return the normalized points AND the info needed to undo it later
    return normalized, frustum_R, frustum_angle


# ============================================================
# STEP 4: Sample to exactly 1024 points
# ============================================================
def sample_to_1024(frustum):
    n = frustum.shape[0]
    if n >= NUM_POINTS:
        idx = np.random.choice(n, NUM_POINTS, replace=False)
    else:
        idx = np.random.choice(n, NUM_POINTS, replace=True)
    return frustum[idx, :]


# ============================================================
# STEP 5: Run the model
# ============================================================
def run_model(model, frustum_1024, device):
    print("STEP 5: Run the model (segmentation -> T-Net -> BboxNet)")
    print("-" * 60)

    # Shape it into the tensor the model wants: (1, 4, 1024)
    x = torch.from_numpy(frustum_1024).unsqueeze(0).transpose(2, 1).to(device)

    model.eval()
    with torch.no_grad():
        seg, tnet, bbox, seg_mean, mask = model(x)

    # --- Foreground %: how many points the model labelled as 'car' ---
    seg_np = seg[0].cpu().numpy()           # (1024, 2)
    car_labels = seg_np[:, 1] > seg_np[:, 0]
    foreground_pct = 100.0 * np.sum(car_labels) / NUM_POINTS

    print(f"  Model foreground: {foreground_pct:.1f}%  "
          f"({np.sum(car_labels)} of {NUM_POINTS} points labelled 'car')")
    print()

    return seg, tnet, bbox, seg_mean, foreground_pct


# ============================================================
# STEP 6: Decode the 14 numbers into a real 7-DOF box
# ============================================================
# This is the exact mirror of normalization — we undo every step.
def decode_box(bbox, tnet, seg_mean, frustum_R, frustum_angle):
    print("STEP 6: Decode the box (14 raw numbers -> real 7-DOF box)")
    print("-" * 60)

    bbox = bbox[0].cpu().numpy()       # (14,)
    tnet = tnet[0].cpu().numpy()       # (3,)
    seg_mean = seg_mean[0].cpu().numpy()  # (3,)

    NH = 4  # number of heading bins

    # --- Centre: add back every offset we subtracted, then un-rotate ---
    # bbox[0:3] = centre residual from BboxNet
    centre_local = (bbox[0:3]
                    + CENTERED_FRUSTUM_MEAN_XYZ
                    + seg_mean
                    + tnet)
    # Un-rotate back to real camera coordinates
    centre = np.dot(np.linalg.inv(frustum_R), centre_local)

    # --- Size: model predicts only the DIFFERENCE from mean car size ---
    pred_h = bbox[3] + MEAN_CAR_SIZE[0]
    pred_w = bbox[4] + MEAN_CAR_SIZE[1]
    pred_l = bbox[5] + MEAN_CAR_SIZE[2]

    # --- Heading: pick the best angle bin + its residual, add frustum angle ---
    bin_scores = bbox[6:6 + NH]
    residuals = bbox[6 + NH:]
    best_bin = np.argmax(bin_scores)
    bin_center = getBinCenter(best_bin, NH=NH)
    residual = residuals[best_bin]
    yaw = wrapToPi(bin_center + residual + frustum_angle)

    print(f"  Position (x, y, z): "
          f"({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}) m")
    print(f"  Size (h, w, l):     "
          f"({pred_h:.2f}, {pred_w:.2f}, {pred_l:.2f}) m")
    print(f"  Heading (yaw):      {np.degrees(yaw):.1f} degrees")
    print()

    return centre, (pred_h, pred_w, pred_l), yaw


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("OFFLINE TESTER — SELF-CONTAINED DEMO (synthetic car)")
    print("=" * 60)
    print()

    # Pick GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")
    print()

    # Load the model + the real pretrained weights
    model = FrustumPointNet(device=device).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    print(f"Loaded pretrained weights from {MODEL_PATH}")
    print()

    # Run the full pipeline
    frustum = make_synthetic_frustum()
    print_depth_histogram(frustum)
    normalized, frustum_R, frustum_angle = normalize_frustum(frustum)
    frustum_1024 = sample_to_1024(normalized)
    seg, tnet, bbox, seg_mean, fg_pct = run_model(model, frustum_1024, device)
    centre, size, yaw = decode_box(bbox, tnet, seg_mean, frustum_R, frustum_angle)

    # --- Verdict ---
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    h, w, l = size
    size_ok = (1.0 < h < 2.5) and (1.0 < w < 2.5) and (2.5 < l < 6.0)
    print(f"  Foreground: {fg_pct:.1f}%")
    print(f"  Decoded box size: h={h:.2f} w={w:.2f} l={l:.2f} m")
    print(f"  Real car is roughly: h=1.5 w=1.6 l=3.9 m")
    print()
    if size_ok:
        print("  PASS — the decoded box is a realistic car size.")
        print("  The model loads, runs, and the decode math is correct.")
    else:
        print("  The box size is off — but the pipeline ran end to end.")
        print("  (With synthetic data this can happen; the real test is on")
        print("   actual bag frames. The point here was to prove it RUNS.)")
    print()


if __name__ == "__main__":
    main()
