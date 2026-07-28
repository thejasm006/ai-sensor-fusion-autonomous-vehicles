#!/usr/bin/env python3
"""
fusion_tracker_node.py
=======================
Adds the two missing pieces from the pipeline's own originally-envisioned
architecture (the "Post-Processing: Confidence Filter / 3D NMS / Tracking"
stage from the very first pipeline diagram): 3D Non-Maximum Suppression and
persistent multi-object tracking.

WHY THIS IS A NEW, SEPARATE NODE (the team's files are not modified):
  frustum_inference_node.py already works and is validated end-to-end. Rather
  than risk it, this node listens to its OUTPUT (/fusion/ai_3d_boxes) and adds
  tracking on top, publishing a NEW topic. Nothing about the existing,
  working pipeline changes -- this is purely additive.

WHAT IT DOES, IN PLAIN TERMS:
  1. 3D NMS ("did two stations tag the same car twice?")
     If two same-class boxes in a single frame overlap heavily, keep only
     one. Honest note: on THIS dataset this rarely fires, because there's
     only one proposal generator per object here (camera+LiDAR fused
     together, not separate competing camera/radar proposal generators like
     the original Frustum-PointNets design envisioned). It's still a
     correct, cheap safety net worth having.
  2. Tracking ("give each object a name tag that stays with it")
     Matches each new detection to the closest existing track of the SAME
     class within a distance gate. Matched tracks keep their ID. Unmatched
     detections become new tracks. Tracks unseen for several frames are
     dropped. This is the same idea behind classic trackers like SORT,
     simplified for this project.

3D IoU NOTE: full oriented-3D IoU needs rotated-rectangle polygon clipping.
This uses an axis-aligned ground-plane footprint approximation instead -- a
clearly labelled simplification, not presented as exact, but sufficient for
a duplicate-detection safety check.

[30/06/2026] FRAME CONVENTION UPDATE -- the team's "Point Stealer Strategy"
files changed the inference node's output convention significantly:
  - Boxes now publish in the 'velodyne' frame, not 'cam0'
  - Rotation is now a Z-axis quaternion (ground-plane yaw), not Y-axis
  - scale.x=length(fwd), scale.y=width(lateral), scale.z=height(up) --
    previously scale.x=width, scale.y=height, scale.z=length
This file's parsing, NMS footprint plane, and publish-side geometry were
all updated to match. The tracking/suppression ALGORITHM itself (Steps 1-5,
both occlusion checks, the grace window) is untouched -- only how we read
position/orientation/size IN and republish them OUT changed.

INPUT:  /fusion/ai_3d_boxes      (MarkerArray, from frustum_inference_node)
OUTPUT: /fusion/tracked_3d_boxes (MarkerArray, same cubes + "Class #ID" labels)

Run with:   python3 fusion_tracker_node.py
(standalone script, no colcon build needed -- same pattern as the diagnostics)
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import math

# Must match frustum_inference_node.py's class_colors EXACTLY to decode
# class identity back out of a marker's color (markers don't carry a label
# string field on the CUBE itself, only on the separate TEXT marker).
CLASS_COLORS = {
    'Car':        (0.0, 1.0, 0.0),
    'Pedestrian': (1.0, 0.5, 0.0),
    'Bicycle':    (0.0, 0.5, 1.0),
    'Truck':      (1.0, 0.0, 1.0),
}

MAX_MATCH_DISTANCE = 2.5    # metres a track may "move" between frames and still match
TRACK_TIMEOUT_FRAMES = 15   # drop a track after this many consecutive unmatched frames
NMS_IOU_THRESHOLD = 0.5     # footprint overlap above this = treat as duplicate
CROSS_CLASS_SUPPRESSION_DISTANCE = 2.0
# [25/06/2026] If an unmatched detection lands within this distance of an
# EXISTING track of a DIFFERENT class, it's almost certainly an occlusion
# artifact -- e.g. a "Car" reading that's really just a pedestrian standing
# directly in front of it, confirmed via watch_car_jump.py: during real
# occlusion events, ALL 1024 of the frustum's points sat at the pedestrian's
# depth with zero far points -- there is no recoverable car signal in that
# frame at all, so the only correct move is to discard the reading rather
# than letting it spawn a phantom box.

IMPLAUSIBLE_JUMP_DISTANCE = 4.5
# [25/06/2026 v3] RECENT_TRACK_GRACE_FRAMES used to be a SMALLER, separate
# window (6) than TRACK_TIMEOUT_FRAMES. That created an exploitable gap: a
# track aged past 6 but not yet past the full timeout was "too old to use
# as a same-class reference, but not yet deleted" -- confirmed live: several
# consecutive other-class readings during one crossing pushed the car track
# into exactly that gap before the bad reading arrived, so NEITHER check
# could catch it. Fix: as long as a track is alive at all, it's eligible as
# a reference -- no separate, narrower grace window.
RECENT_TRACK_GRACE_FRAMES = TRACK_TIMEOUT_FRAMES


def color_to_class(r, g, b, tol=0.15):
    for label, (cr, cg, cb) in CLASS_COLORS.items():
        if abs(r - cr) < tol and abs(g - cg) < tol and abs(b - cb) < tol:
            return label
    return 'Unknown'


def quaternion_to_yaw(qx, qy, qz, qw):
    """[30/06/2026] UPDATED: the new frustum_inference_node builds its
    quaternion as a Z-axis rotation (x=0, y=0, z=sin(yaw/2), w=cos(yaw/2)),
    matching the velodyne frame's convention (Z = up, yaw = ground-plane
    heading). Previously this read qy for a Y-axis rotation (the old
    cam0-frame convention, where Y pointed down). This reverses the NEW
    convention exactly."""
    return 2.0 * math.atan2(qz, qw)


def footprint_iou(boxA, boxB):
    """Axis-aligned ground-plane footprint IoU -- a labelled simplification
    of true oriented 3D IoU, used only as a defensive duplicate check.
    [30/06/2026] Caller now passes (x, y, length, width) -- the velodyne
    frame's horizontal ground plane is X-Y (X=forward, Y=lateral, Z=up),
    NOT X-Z as it was under the old cam0-frame convention (where Y was
    the vertical/down axis). The function body's math is unchanged --
    it's purely the CALLER's job to pass the correct two horizontal axes."""
    ax, az, aw, al = boxA
    bx, bz, bw, bl = boxB
    a_x1, a_x2 = ax - aw / 2, ax + aw / 2
    a_z1, a_z2 = az - al / 2, az + al / 2
    b_x1, b_x2 = bx - bw / 2, bx + bw / 2
    b_z1, b_z2 = bz - bl / 2, bz + bl / 2

    inter_x = max(0.0, min(a_x2, b_x2) - max(a_x1, b_x1))
    inter_z = max(0.0, min(a_z2, b_z2) - max(a_z1, b_z1))
    inter_area = inter_x * inter_z

    union = (aw * al) + (bw * bl) - inter_area
    return inter_area / union if union > 0 else 0.0


class Track:
    _next_id = 0

    def __init__(self, label, x, y, z, w, h, l, yaw):
        self.id = Track._next_id
        Track._next_id += 1
        self.label = label
        self.x, self.y, self.z = x, y, z
        self.w, self.h, self.l = w, h, l
        self.yaw = yaw
        self.age_since_seen = 0

    def update(self, x, y, z, w, h, l, yaw):
        self.x, self.y, self.z = x, y, z
        self.w, self.h, self.l = w, h, l
        self.yaw = yaw
        self.age_since_seen = 0


class FusionTrackerNode(Node):
    def __init__(self):
        super().__init__('fusion_tracker_node')
        self.sub = self.create_subscription(
            MarkerArray, '/fusion/ai_3d_boxes', self.callback, 10)
        self.pub = self.create_publisher(
            MarkerArray, '/fusion/tracked_3d_boxes', 10)
        self.tracks = []
        self.nms_removed_total = 0
        self.frame_count = 0
        self.get_logger().info(
            '✅ Fusion Tracker Node ready | listening on /fusion/ai_3d_boxes '
            '| publishing /fusion/tracked_3d_boxes')

    def callback(self, msg: MarkerArray):
        # --- Step 1: parse incoming CUBE markers into plain detections ---
        detections = []
        for m in msg.markers:
            if m.type != Marker.CUBE:
                continue
            label = color_to_class(m.color.r, m.color.g, m.color.b)
            x, y, z = m.pose.position.x, m.pose.position.y, m.pose.position.z
            yaw = quaternion_to_yaw(m.pose.orientation.x, m.pose.orientation.y,
                                     m.pose.orientation.z, m.pose.orientation.w)
            # [30/06/2026] UPDATED: the new inference node maps
            # scale.x=length(fwd), scale.y=width(lateral), scale.z=height(up)
            # -- previously scale.x=width, scale.y=height, scale.z=length.
            # We keep our internal Track fields named w/h/l (width/height/
            # length) for minimal downstream diff -- only the SOURCE axis
            # each one is read from has changed.
            l, w, h = m.scale.x, m.scale.y, m.scale.z
            detections.append({'label': label, 'x': x, 'y': y, 'z': z,
                                'w': w, 'h': h, 'l': l, 'yaw': yaw})

        # --- Step 2: 3D NMS within this single frame ---
        kept = []
        used = [False] * len(detections)
        for i, det in enumerate(detections):
            if used[i]:
                continue
            kept.append(det)
            for j in range(i + 1, len(detections)):
                if used[j] or detections[j]['label'] != det['label']:
                    continue
                # [30/06/2026] Pass (x, y, length, width) -- velodyne's
                # horizontal ground plane is X-Y now, not X-Z (see
                # footprint_iou's docstring above for why this changed).
                iou = footprint_iou(
                    (det['x'], det['y'], det['l'], det['w']),
                    (detections[j]['x'], detections[j]['y'],
                     detections[j]['l'], detections[j]['w']))
                if iou > NMS_IOU_THRESHOLD:
                    used[j] = True
                    self.nms_removed_total += 1

        # --- Step 3: match kept detections to existing tracks ---
        matched_track_ids = set()
        suppressed_count = 0
        for det in kept:
            best_track, best_dist = None, MAX_MATCH_DISTANCE
            for t in self.tracks:
                if t.label != det['label'] or t.id in matched_track_ids:
                    continue
                dist = math.sqrt((t.x - det['x'])**2 + (t.y - det['y'])**2 + (t.z - det['z'])**2)
                if dist < best_dist:
                    best_dist, best_track = dist, t
            if best_track is not None:
                best_track.update(det['x'], det['y'], det['z'],
                                   det['w'], det['h'], det['l'], det['yaw'])
                matched_track_ids.add(best_track.id)
            else:
                # Check 1: is this suspiciously co-located with an EXISTING
                # track of a DIFFERENT class? (the original occlusion signature)
                is_occlusion_artifact = False
                for t in self.tracks:
                    if t.label == det['label']:
                        continue
                    dist = math.sqrt((t.x - det['x'])**2 + (t.y - det['y'])**2 + (t.z - det['z'])**2)
                    if dist < CROSS_CLASS_SUPPRESSION_DISTANCE:
                        is_occlusion_artifact = True
                        break

                # Check 2: independent of Check 1 -- is this an implausibly
                # large jump from a recently-seen SAME-class track? This
                # catches the case where Check 1's reference (the other
                # class's track) was itself unstable at that exact moment.
                if not is_occlusion_artifact:
                    for t in self.tracks:
                        if t.label != det['label']:
                            continue
                        if t.age_since_seen > RECENT_TRACK_GRACE_FRAMES:
                            continue  # too long ago to treat as "the same object jumping"
                        dist = math.sqrt((t.x - det['x'])**2 + (t.y - det['y'])**2 + (t.z - det['z'])**2)
                        if dist > IMPLAUSIBLE_JUMP_DISTANCE:
                            is_occlusion_artifact = True
                            break

                if is_occlusion_artifact:
                    suppressed_count += 1
                else:
                    new_track = Track(det['label'], det['x'], det['y'], det['z'],
                                       det['w'], det['h'], det['l'], det['yaw'])
                    self.tracks.append(new_track)
                    matched_track_ids.add(new_track.id)

        # --- Step 4: age out tracks not matched this frame ---
        still_alive = []
        for t in self.tracks:
            if t.id not in matched_track_ids:
                t.age_since_seen += 1
            if t.age_since_seen <= TRACK_TIMEOUT_FRAMES:
                still_alive.append(t)
        self.tracks = still_alive

        # --- Step 5: publish ALL active tracks (not just this frame's
        # detections). A track that wasn't matched this frame still gets
        # republished at its LAST KNOWN GOOD position -- this is what keeps
        # a temporarily-occluded object's box frozen in place instead of
        # disappearing or jumping, for as long as it stays within the
        # timeout window. ---
        out = MarkerArray()
        for t in self.tracks:
            cube = Marker()
            # [30/06/2026] frame_id changed from 'cam0' to 'velodyne' to
            # match where the incoming position data actually lives now.
            cube.header.frame_id = 'velodyne'
            cube.header.stamp = self.get_clock().now().to_msg()
            cube.ns = 'tracked_3d_boxes'
            cube.id = t.id
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.lifetime = rclpy.duration.Duration(seconds=1.5).to_msg()
            cube.pose.position.x = t.x
            cube.pose.position.y = t.y
            cube.pose.position.z = t.z
            half = t.yaw * 0.5
            # [30/06/2026] Z-axis rotation now (ground-plane yaw in the
            # velodyne frame), previously Y-axis (cam0 frame, Y=down).
            cube.pose.orientation.z = math.sin(half)
            cube.pose.orientation.w = math.cos(half)
            # [30/06/2026] scale.x=length(fwd), scale.y=width(lateral),
            # scale.z=height(up) -- matches the new inference node exactly,
            # so our republished boxes have the same geometry convention.
            cube.scale.x = max(0.1, t.l)
            cube.scale.y = max(0.1, t.w)
            cube.scale.z = max(0.1, t.h)
            r, g, b = CLASS_COLORS.get(t.label, (1.0, 1.0, 1.0))
            cube.color.r, cube.color.g, cube.color.b, cube.color.a = r, g, b, 0.55
            out.markers.append(cube)

            text = Marker()
            text.header = cube.header
            text.ns = 'tracked_labels'
            text.id = t.id + 5000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.lifetime = cube.lifetime
            text.pose.position.x = t.x
            text.pose.position.y = t.y
            # [30/06/2026] In cam0, Y pointed DOWN, so subtracting moved the
            # label UP above the box. In velodyne, Z points UP, so we now
            # ADD half the height (+ a small offset) to place it above.
            text.pose.position.z = t.z + t.h / 2.0 + 0.3
            text.pose.orientation.w = 1.0
            text.scale.z = 0.4
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f"{t.label} #{t.id}"
            out.markers.append(text)

        self.pub.publish(out)
        self.frame_count += 1
        if self.frame_count % 5 == 0:  # don't flood the terminal
            import time
            ts = time.strftime("%H:%M:%S")
            id_list = ", ".join(f"{t.label}#{t.id}(z={t.z:.1f})" for t in self.tracks)
            print(f"[{ts}] frame {self.frame_count}: {len(detections)} in -> "
                  f"{len(kept)} after NMS -> {len(self.tracks)} active tracks "
                  f"[{id_list}] (NMS removed: {self.nms_removed_total}, "
                  f"occlusion-suppressed: {suppressed_count})")


def main():
    rclpy.init()
    node = FusionTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
