"""
read_tf_static.py  —  Extract sensor mounting transforms (v2, robust)
=====================================================================
Same goal as before, but now using the 'rosbags' library to decode the
messages properly — no fragile manual byte-counting. The library knows
the exact ROS 2 message format, so it can't get the alignment wrong.

What this does (read-only, changes nothing):
  - Opens the .db3 bag
  - Reads every /tf_static message
  - Prints each transform: parent frame, child frame, translation, rotation
  - Builds the 4x4 matrix and highlights camera/lidar transforms

Run with:   python3 read_tf_static.py
"""

import numpy as np
from pathlib import Path

# rosbags handles the decoding correctly
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# Point this at the FOLDER that contains the .db3 (not the .db3 itself)
BAG_DIR = "/root/data/studentProject"


def quaternion_to_matrix(qx, qy, qz, qw):
    """Convert a rotation quaternion into a 3x3 rotation matrix."""
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-10:
        return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])


def main():
    print("=" * 64)
    print("READING /tf_static — the sensor mounting transforms (v2)")
    print("=" * 64)
    print()

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    all_transforms = []

    with Reader(Path(BAG_DIR)) as reader:
        # Find the /tf_static connection(s)
        tf_connections = [c for c in reader.connections if c.topic == '/tf_static']
        if not tf_connections:
            print("ERROR: no /tf_static topic in this bag.")
            return

        for conn, timestamp, rawdata in reader.messages(connections=tf_connections):
            # Decode the TFMessage properly
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            for tf in msg.transforms:
                t = tf.transform.translation
                r = tf.transform.rotation
                all_transforms.append({
                    'parent': tf.header.frame_id,
                    'child': tf.child_frame_id,
                    'translation': (t.x, t.y, t.z),
                    'quaternion': (r.x, r.y, r.z, r.w),
                })

    # Deduplicate (static transforms repeat across the 3 messages)
    seen = set()
    unique = []
    for tf in all_transforms:
        key = (tf['parent'], tf['child'])
        if key not in seen:
            seen.add(key)
            unique.append(tf)

    print(f"Found {len(unique)} unique transform(s).")
    print()
    print("-" * 64)
    print("ALL TRANSFORMS:")
    print("-" * 64)
    for i, tf in enumerate(unique):
        tx, ty, tz = tf['translation']
        qx, qy, qz, qw = tf['quaternion']
        print(f"\nTransform {i + 1}:")
        print(f"  parent frame : {tf['parent']}")
        print(f"  child frame  : {tf['child']}")
        print(f"  translation  : x={tx:.4f}  y={ty:.4f}  z={tz:.4f}  (metres)")
        print(f"  rotation quat: x={qx:.4f}  y={qy:.4f}  z={qz:.4f}  w={qw:.4f}")

        R = quaternion_to_matrix(qx, qy, qz, qw)
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [tx, ty, tz]
        print(f"  4x4 matrix T (maps a point in '{tf['child']}' -> '{tf['parent']}'):")
        for row in range(4):
            print("     [{:9.4f} {:9.4f} {:9.4f} {:9.4f}]".format(*T[row]))

    # Highlight camera/lidar transforms
    print()
    print("=" * 64)
    print("CAMERA / LIDAR RELATED FRAMES")
    print("=" * 64)
    keywords = ['velodyne', 'lidar', 'cam', 'camera', 'os_', 'blackfly']
    relevant = [tf for tf in unique
                if any(k in tf['parent'].lower() or k in tf['child'].lower()
                       for k in keywords)]
    if relevant:
        for tf in relevant:
            print(f"  {tf['child']}  ->  {tf['parent']}")
    else:
        print("  None found by name — check the full list above.")

    print()
    print("Copy ALL of this output and send it back.")
    print("I'll identify the exact velodyne -> cam0 chain and build the extractor.")


if __name__ == "__main__":
    main()