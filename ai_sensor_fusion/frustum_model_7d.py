#!/usr/bin/env python3
"""
frustum_model_7d.py  —  SINGLE SOURCE OF TRUTH
===============================================
Import ALL constants and classes from here. Never redefine NUM_CLASSES,
NUM_HEADING_BINS, or the heading utilities in any other file.

━━━ Coordinate conventions (used everywhere in this pipeline) ━━━━━━━━━━━━

  cam0 frame :  X = right   Y = down   Z = forward (depth into scene)

  nuScenes box size order:  size = [width, length, height]
    size[0]  width   lateral extent     → cam0 X axis
    size[1]  length  forward extent     → cam0 Z axis
    size[2]  height  vertical extent    → cam0 Y axis

  nuScenes yaw convention:
    yaw = 0 means the object faces the nuScenes global X-axis.
    cam0 forward is Z, so a straight-ahead object has yaw ≈ π/2 in nuScenes.
    The inference node corrects this with:
        yaw_cam0 = yaw − π/2
    before building the RViz quaternion (pure Y-axis rotation in cam0).

  RViz CUBE Marker scale in cam0 frame:
    m.scale.x  →  cam0 X  →  width   = size[0]
    m.scale.y  →  cam0 Y  →  height  = size[2]   ← NOTE: height, not length
    m.scale.z  →  cam0 Z  →  length  = size[1]

━━━ BoxNet output layout (23 values, FIXED — must match trained .pth) ━━━━━

  [0 :3 ]  center XYZ offset, relative to T-Net translation origin
  [3 :6 ]  size [width, length, height] via F.softplus + 0.1 (always > 0.1 m)
  [6 :18]  12 heading-bin logits  (30° per bin, spanning −180° to +180°)
  [18:19]  heading residual (continuous, in radians)
  [19:23]  4 class logits  (Car=0  Pedestrian=1  Bicycle=2  Truck=3)

━━━ Training ↔ inference centroid contract ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  The CALLER (dataset __getitem__ OR inference node) MUST centroid-normalise
  the XYZ channels before passing x to the network:
      centroid       = mean(pts[:, 0:3], axis=0)
      pts[:, 0:3]   -= centroid
  After the forward pass the inference node restores real-world coords:
      center_world   = network_output_center_abs + centroid
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Global constants ──────────────────────────────────────────────────────────
NUM_CLASSES      = 4    # 0=Car  1=Pedestrian  2=Bicycle  3=Truck
NUM_HEADING_BINS = 12   # 360° / 12 = 30° per bin


# ── Heading utilities ─────────────────────────────────────────────────────────

def angle_to_bin_and_residual(yaw: torch.Tensor,
                               num_bins: int = NUM_HEADING_BINS):
    """
    TRAINING-TIME ONLY.
    Convert yaw (B, 1) → bin index (B,) and residual (B, 1).
    Bins are uniformly spaced over [−π, π].
    The residual is the continuous angle within the winning bin.
    """
    bin_size   = 2.0 * np.pi / num_bins
    # Wrap yaw safely to [−π, π] using atan2 of (sin, cos)
    yaw_norm   = torch.atan2(torch.sin(yaw), torch.cos(yaw)).squeeze(1)  # (B,)
    bin_idx    = ((yaw_norm + np.pi) / bin_size).long().clamp(0, num_bins - 1)
    bin_center = bin_idx.float() * bin_size - np.pi + bin_size / 2.0
    residual   = yaw_norm - bin_center
    return bin_idx, residual.unsqueeze(1)                # (B,), (B, 1)


def bin_and_residual_to_angle(bin_logits: torch.Tensor,
                               residuals:  torch.Tensor,
                               num_bins:   int = NUM_HEADING_BINS) -> torch.Tensor:
    """
    INFERENCE-TIME ONLY.
    Convert predicted bin logits (B, num_bins) and residual (B, 1 or B,)
    → continuous yaw angle (B,).
    Safe squeeze handles both residual tensor shapes.
    """
    bin_size   = 2.0 * np.pi / num_bins
    bin_idx    = torch.argmax(bin_logits, dim=1).float()          # (B,)
    bin_center = bin_idx * bin_size - np.pi + bin_size / 2.0
    # Handle both (B, 1) and (B,) residual tensors safely
    res        = residuals.squeeze(-1) if residuals.dim() > 1 else residuals
    return bin_center + res                                        # (B,)


# ── T-Net ─────────────────────────────────────────────────────────────────────

class TNet3D(nn.Module):
    """
    Spatial transformer network.

    WHY IT EXISTS:
    The raw frustum points are in absolute cam0 coordinates (e.g. a car
    at 15 m depth). Without centering, BoxNet would have to memorise
    absolute depth and lateral positions for every possible scene location.
    T-Net predicts a translation that centres the cluster near the origin,
    so BoxNet only has to learn local shape-to-box relationships.

    The output layer is zero-initialised so the network starts as the
    identity transform (no translation) at epoch 0. This prevents
    catastrophic early-training divergence before the rest of the
    network has learned meaningful features.

    Input  (B, 7, N)  →  translation (B, 3)
    """
    def __init__(self, in_channels: int = 7):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64,  1)
        self.conv2 = nn.Conv1d(64,          128, 1)
        self.conv3 = nn.Conv1d(128,         256, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(256)
        self.fc1   = nn.Linear(256, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64,  3)
        self.bn4   = nn.BatchNorm1d(128)
        self.bn5   = nn.BatchNorm1d(64)
        # Identity init — zero weights, network starts as a no-op
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):               # (B, 7, N)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.max(out, 2)[0]      # global max-pool → (B, 256)
        out = F.relu(self.bn4(self.fc1(out)))
        out = F.relu(self.bn5(self.fc2(out)))
        return self.fc3(out)            # (B, 3)


# ── Segmentation network ──────────────────────────────────────────────────────

class FrustumPointNetSeg(nn.Module):
    """
    Per-point foreground / background segmentation.

    ARCHITECTURE:
    Builds two kinds of features per point:
      • Local (f2, 64-d):  captures fine per-point geometry
      • Global (512-d):    max-pooled across all N points, tiled back
    These are concatenated → 576-d per-point descriptor → 2-class logits.

    The same global feature vector (B, 512) is passed to BoxNet so it
    can regress the box from scene-level context.

    CRITICAL SHAPE CONTRACT:
    global_feat is returned as (B, 512), not (B, 512, 1).
    The trained .pth weights were saved with .squeeze(2) applied here.
    Changing this shape breaks checkpoint loading without retraining.

    Returns:
      seg_logits  (B, 2, N)   raw logits: class 0 = background, 1 = foreground
      global_feat (B, 512)    for BoxNet
    """
    def __init__(self, in_channels: int = 7):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64,  1)
        self.conv2 = nn.Conv1d(64,          64,  1)
        self.conv3 = nn.Conv1d(64,          64,  1)
        self.conv4 = nn.Conv1d(64,          128, 1)
        self.conv5 = nn.Conv1d(128,         512, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(64)
        self.bn3   = nn.BatchNorm1d(64)
        self.bn4   = nn.BatchNorm1d(128)
        self.bn5   = nn.BatchNorm1d(512)

        # Seg head: local(64) + global(512) = 576 → 2 classes
        self.seg_conv1 = nn.Conv1d(64 + 512, 256, 1)
        self.seg_conv2 = nn.Conv1d(256,       128, 1)
        self.seg_conv3 = nn.Conv1d(128,       2,   1)
        self.seg_bn1   = nn.BatchNorm1d(256)
        self.seg_bn2   = nn.BatchNorm1d(128)
        self.seg_drop  = nn.Dropout(p=0.3)

    def forward(self, x):                              # (B, 7, N)
        N  = x.size(2)
        f1 = F.relu(self.bn1(self.conv1(x)))
        f2 = F.relu(self.bn2(self.conv2(f1)))          # (B,  64, N)  ← local features
        f3 = F.relu(self.bn3(self.conv3(f2)))
        f4 = F.relu(self.bn4(self.conv4(f3)))
        f5 = F.relu(self.bn5(self.conv5(f4)))          # (B, 512, N)

        global_feat = torch.max(f5, 2, keepdim=True)[0]   # (B, 512, 1)
        global_tile = global_feat.expand(-1, -1, N)        # (B, 512, N)

        concat     = torch.cat([f2, global_tile], dim=1)   # (B, 576, N)
        out        = F.relu(self.seg_bn1(self.seg_conv1(concat)))
        out        = self.seg_drop(out)
        out        = F.relu(self.seg_bn2(self.seg_conv2(out)))
        seg_logits = self.seg_conv3(out)                   # (B, 2, N)

        # squeeze to (B, 512) — must match the trained checkpoint
        return seg_logits, global_feat.squeeze(2)


# ── Box regression network ────────────────────────────────────────────────────

class BoxNet(nn.Module):
    """
    Regresses all 23 3-D bounding-box parameters from the 512-d global feature.

    Size uses F.softplus(raw) + 0.1 so all three dimensions are always
    strictly positive and cannot collapse to zero during training.
    """
    def __init__(self,
                 num_classes:      int = NUM_CLASSES,
                 num_heading_bins: int = NUM_HEADING_BINS):
        super().__init__()
        self.num_heading_bins = num_heading_bins
        self.fc1  = nn.Linear(512, 256)
        self.fc2  = nn.Linear(256, 128)
        self.bn1  = nn.BatchNorm1d(256)
        self.bn2  = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.3)
        out_dim   = 3 + 3 + num_heading_bins + 1 + num_classes   # 23
        self.fc_out = nn.Linear(128, out_dim)

    def forward(self, global_feat):          # (B, 512)
        out   = F.relu(self.bn1(self.fc1(global_feat)))
        out   = self.drop(out)
        out   = F.relu(self.bn2(self.fc2(out)))
        preds = self.fc_out(out)             # (B, 23)

        idx        = 0
        center     = preds[:, idx:idx+3];                      idx += 3
        size       = F.softplus(preds[:, idx:idx+3]) + 0.1;   idx += 3
        head_bins  = preds[:, idx:idx+self.num_heading_bins];  idx += self.num_heading_bins
        head_res   = preds[:, idx:idx+1];                      idx += 1
        cls_logits = preds[:, idx:]                            # (B, 4)

        return center, size, head_bins, head_res, cls_logits


# ── Top-level pipeline ────────────────────────────────────────────────────────

class CompleteFusionNet(nn.Module):
    """
    End-to-end frustum PointNet pipeline.

    The forward pass EXACTLY mirrors train_real_dataset.py — any change
    here must be reflected there and vice versa, otherwise the inference
    node will silently load incompatible weights.

    FORWARD PASS STEPS:
      1. T-Net predicts a translation; XYZ channels are shifted by it.
         This gives SegNet/BoxNet a canonical, position-invariant view.
      2. SegNet processes the aligned cloud → seg_logits + global_feat.
      3. BoxNet regresses the 3-D box from global_feat alone (no XYZ).
      4. center_abs = center_rel + translation
         Positions the prediction back in the centroid-normalised cam0 frame.
         The inference node then adds the real centroid to get world coords.

    INPUT CHANNELS (B, 7, N):
      ch[0:3]  XYZ in cam0 frame, CENTROID-NORMALISED by the caller
      ch[3]    intensity (LiDAR return strength)
      ch[4:6]  vx, vy  (radar Doppler velocity; 0.0 for LiDAR points)
      ch[6]    modality flag  (0.0 = LiDAR,  1.0 = radar)

    OUTPUTS (6 tensors):
      seg_logits  (B, 2,  N)  per-point FG/BG logits
      center_abs  (B, 3)      XYZ in centroid-normalised cam0 frame
      size        (B, 3)      [width, length, height] in metres
      head_bins   (B, 12)     heading bin logits
      head_res    (B, 1)      heading residual in radians
      cls_logits  (B, 4)      class logits

    NOTE: No soft-masking in forward. The FG kill-switch is a
    post-inference gate in the inference node, not a network operation.
    """
    def __init__(self,
                 num_classes:      int = NUM_CLASSES,
                 num_heading_bins: int = NUM_HEADING_BINS):
        super().__init__()
        self.tnet   = TNet3D()
        self.segnet = FrustumPointNetSeg()
        self.boxnet = BoxNet(num_classes, num_heading_bins)

    def forward(self, x):                        # (B, 7, N)
        # Step 1 — T-Net alignment
        translation  = self.tnet(x)              # (B, 3)
        x_aligned    = x.clone()
        x_aligned[:, 0:3, :] -= translation.unsqueeze(2)

        # Step 2 — Segmentation + global feature extraction
        seg_logits, global_feat = self.segnet(x_aligned)

        # Step 3 — Box regression from global feature
        center_rel, size, head_bins, head_res, cls_logits = \
            self.boxnet(global_feat)

        # Step 4 — Restore position in centroid-normalised cam0 frame
        # The inference node adds the real centroid afterwards.
        center_abs = center_rel + translation

        return seg_logits, center_abs, size, head_bins, head_res, cls_logits
