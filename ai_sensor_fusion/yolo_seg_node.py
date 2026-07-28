import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from rclpy.qos import qos_profile_sensor_data

class YoloSegmentationNode(Node):
    def __init__(self):
        super().__init__('yolo_segmentation_node')
        
        # 1. Sensor Subscriptions (Listening to your bag camera topic)
        self.subscription = self.create_subscription(
            Image,
            '/blackfly_s/cam0/image_rectified',  
            self.image_callback,
            10
        )
        
        # 2. Output Detections Topic for RViz2 Dashboard
        self.publisher_ = self.create_publisher(Image, '/yolo/detections', 10)
        self.bridge = CvBridge()
        
        # 3. EXISTING: 2D Bounding Box publisher (Kept identical for compatibility)
        self.publisher = self.create_publisher(Detection2DArray, '/detections', 10)

        # --- NEW: Instance Mask Publisher ---
        # This publishes the single-channel 8-bit image array for your 3D extractor
        self.mask_publisher = self.create_publisher(Image, '/yolo/instance_mask', 10)

        # 4. Load the High-Performance Compiled TensorRT Engine
        self.get_logger().info('Initializing TensorRT YOLO Instance Segmentation Engine (All 80 COCO Classes)...')
        self.model = YOLO('/home/workspace/models/yolo26s-seg.engine', task='segment')
        self.get_logger().info('TensorRT Engine Loaded. Generating Distinct Color Schemes...')
        
        # 5. Generate 80 unique, distinct colors deterministically using a fixed seed
        np.random.seed(42)  # Ensures colors remain identical on every execution loop
        self.color_map = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
        
        # Explicit priority overrides for standard automotive targets (BGR Convention)
        self.color_map[0] = [255, 0, 0]    # Person: Pure Blue
        self.color_map[2] = [0, 0, 255]    # Car: Pure Red
        self.color_map[3] = [255, 255, 0]  # Motorcycle: Cyan
        self.color_map[5] = [255, 0, 255]  # Bus: Magenta
        self.color_map[7] = [0, 255, 255]  # Truck: Yellow
        self.color_map[58] = [0, 255, 0]   # Tree/Vegetation: Pure Green

    def image_callback(self, msg):
        try:
            # Convert incoming ROS image data stream to an OpenCV matrix
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Safeguard: Extract image dimensions globally inside the scope of this callback frame
            h_img, w_img = cv_image.shape[:2]
            
            # --- NEW: Create a blank canvas for the instance mask ---
            # 0 = Background. Any other number equals the object ID.
            instance_mask_img = np.zeros((h_img, w_img), dtype=np.uint8)

            # Run inference inside TensorRT over all 80 classes, filtering bad data via higher confidence thresholds
            results = self.model.track(cv_image, conf=0.6, iou=0.6, persist=True, retina_masks=True, verbose=False)
            
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    detection = Detection2D()

                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                    bbox = detection.bbox
                    bbox.center.position.x = float((x_min + x_max) / 2.0)
                    bbox.center.position.y = float((y_min + y_max) / 2.0)
                    bbox.center.theta = 0.0
                    bbox.size_x = float(x_max - x_min)
                    bbox.size_y = float(y_max - y_min)

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(int(box.cls[0]))
                    hypothesis.hypothesis.score = float(box.conf[0])
                    
                    detection.results.append(hypothesis)
                    detections_msg.detections.append(detection)

            self.publisher.publish(detections_msg)

            # Setup a clean overlay mask layer canvas
            mask_overlay = np.zeros_like(cv_image, dtype=np.uint8)
            
            # Array to hold text annotations so they don't get alpha-blended
            annotations = []

            if results[0].masks is not None and results[0].boxes is not None:
                raw_masks = results[0].masks.data.cpu().numpy()  
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                # Extract Bounding Box coordinates to calculate object positions
                boxes_data = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [-1] * len(boxes_data) # Fallback if ID is missing on frame 1

                for i, raw_mask in enumerate(raw_masks):
                    cls_id = class_ids[i]
                    x1, y1, x2, y2 = boxes_data[i] #unpack box coordinates

                    # ====================================================
                    # EGO-VEHICLE SPATIAL EXCLUSION FILTER
                    # ====================================================
                    ego_hood_y_threshold = h_img * 0.85  
                    box_width = x2 - x1
                    
                    if y2 > ego_hood_y_threshold and box_width > (w_img * 0.30):
                        continue # Skip processing this mask entirely
                    # ====================================================

                    if cls_id >= len(self.color_map):
                        instance_color = [0, 165, 255] # Fallback Orange
                    else:
                        instance_color = self.color_map[cls_id].tolist()
                    
                    class_name = self.model.names[cls_id].capitalize()
                    conf = confidences[i]
                    t_id = track_ids[i]

                    # Calculate Target Centroid & Draw Crosshair ---
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.drawMarker(mask_overlay, (cx, cy), instance_color, markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2, line_type=cv2.LINE_AA)
                    
                    id_string = f"#{t_id} " if t_id != -1 else ""
                    label_text = f"{class_name} {id_string}({conf:.2f})"
                    annotations.append((int(x1), int(y1), label_text, instance_color))

                    # Crisp Boundary Resolution
                    resized_mask = cv2.resize(raw_mask, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                    
                    # Hard Binary Threshold
                    _, sharp_binary_mask = cv2.threshold(resized_mask, 0.5, 255, cv2.THRESH_BINARY)
                    sharp_binary_mask = sharp_binary_mask.astype(np.uint8)
                    
                    # --- NEW: Paint the Instance ID onto the hidden data layer ---
                    # We add (i + 1) to ensure the first instance is 1, so 0 remains background
                    instance_mask_img[sharp_binary_mask == 255] = (i + 1)
                    # -------------------------------------------------------------

                    # Morphological Closing & Contour Filling 
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    closed_mask = cv2.morphologyEx(sharp_binary_mask, cv2.MORPH_CLOSE, kernel)
                    
                    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    solid_mask = np.zeros_like(closed_mask)
                    if contours:
                        cv2.drawContours(solid_mask, contours, -1, 255, thickness=cv2.FILLED)
                    
                    # Paint target color onto canvas
                    color_mask = np.zeros_like(cv_image, dtype=np.uint8)
                    color_mask[solid_mask == 255] = instance_color
                    mask_overlay = cv2.bitwise_or(mask_overlay, color_mask)
            
            # Alpha blend mask canvas with original camera frame
            fused_display = cv2.addWeighted(cv_image, 1.0, mask_overlay, 0.5, 0)
            
            # Draw explicit contours
            if results[0].masks is not None:
                gray_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2GRAY)
                contours, _ = cv2.findContours(gray_overlay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(fused_display, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw text plates
            for x, y, text, color in annotations:
                (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(fused_display, (x, y - t_h - 10), (x + t_w, y), color, -1)
                cv2.putText(fused_display, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Ship converted visualization topic
            ros_out_msg = self.bridge.cv2_to_imgmsg(fused_display, encoding='bgr8')
            ros_out_msg.header = msg.header
            self.publisher_.publish(ros_out_msg)

            # --- NEW: Publish the Hidden Data Mask for the 3D Node ---
            mask_ros_msg = self.bridge.cv2_to_imgmsg(instance_mask_img, encoding='mono8')
            mask_ros_msg.header = msg.header
            self.mask_publisher.publish(mask_ros_msg)
            # -------------------------------------------------------------

        except Exception as e:
            self.get_logger().error(f'Failure inside TensorRT loop: {str(e)}')
        
        

def main(args=None):
    rclpy.init(args=args)
    node = YoloSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()