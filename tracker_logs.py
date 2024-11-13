import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from PIL import Image
import torchreid
from torchreid.reid.utils import load_pretrained_weights
from SpeedEye.deep_sort.deep_sort.detection import Detection
from SpeedEye.deep_sort.deep_sort import nn_matching
from SpeedEye.deep_sort.deep_sort.tracker import Tracker as DeepSort
import os
import shutil
from datetime import datetime
from datetime import timedelta

class PersonDetectionMatcher:
    def __init__(self, confidence_threshold=0.5, feature_similarity_threshold=0.7):
        # Clean up existing matched images directory
        self.image_dir = 'matched_images'
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir)

        self.model = YOLO('yolov8n.pt')

        max_cosine_distance = 0.3
        nn_budget = None

        metric1 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker1 = DeepSort(metric1, max_iou_distance=0.7, max_age=30, n_init=3)
        self.tracker2 = DeepSort(metric2, max_iou_distance=0.7, max_age=30, n_init=3)

        self.feature_encoder = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            loss='softmax',
            pretrained=True
        )

        load_pretrained_weights(self.feature_encoder,
                                '/Users/aayush/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth')

        self.feature_encoder.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_encoder = self.feature_encoder.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.persistent_matches = {}
        self.matched_images = {}

        self.confidence_threshold = confidence_threshold
        self.feature_similarity_threshold = feature_similarity_threshold
        self.jaywalking_region = np.array([(1048, 600), (733, 955), (7, 864), (7, 571)], np.int32)
        self.logs = []
        self.last_jaywalking_timestamp = {}


    def is_in_jaywalking_region(self, bbox):
        """Check if the bottom center of the bounding box is in the jaywalking region"""
        bottom_center = (int((bbox[0] + bbox[2]) / 2), bbox[3])
        return cv2.pointPolygonTest(self.jaywalking_region, bottom_center, False) >= 0

    def extract_features(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        person_roi = frame[y1:y2, x1:x2]

        if person_roi.size == 0:
            return np.zeros(512)

        pil_image = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_encoder(input_batch)

        features = features.squeeze().cpu().numpy()
        features = features / np.linalg.norm(features)

        return features

    def detect_people(self, frame):
        results = self.model(frame, classes=0)
        detections = []

        for r in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, class_id = r
            if score > self.confidence_threshold:
                bbox = [int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)]
                original_bbox = (int(x1), int(y1), int(x2), int(y2))
                detections.append({
                    'bbox': bbox,
                    'confidence': score,
                    'original_bbox': original_bbox
                })

        return detections

    def save_person_image(self, frame, bbox, camera_id, person_id, timestamp):
        """Extract and save person image from frame"""
        x1, y1, x2, y2 = bbox
        person_img = frame[y1:y2, x1:x2]
        if person_img.size > 0:
            filename = f"{self.image_dir}/person_{camera_id}_{person_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, person_img)
            return filename
        return None

    def update_persistent_matches(self, det1, det2, similarity, frame1, frame2, current_timestamp):
        """Update persistent matches between cameras"""
        camera1_id = det1['id']
        camera2_id = det2['id']

        if camera1_id not in self.persistent_matches:
            # New match
            self.persistent_matches[camera1_id] = {
                'camera2_id': camera2_id,
                'match_duration': timedelta(seconds=0),
                'last_updated': current_timestamp,
                'similarity': similarity
            }

            # Save initial images
            cam1_img_path = self.save_person_image(frame1, det1['bbox'], 1, camera1_id, current_timestamp)
            cam2_img_path = self.save_person_image(frame2, det2['bbox'], 2, camera2_id, current_timestamp)

            self.matched_images[camera1_id] = {
                'cam1_image_path': cam1_img_path,
                'cam2_image_path': cam2_img_path,
                'timestamp': current_timestamp
            }
        else:
            # Update existing match
            existing_match = self.persistent_matches[camera1_id]
            if existing_match['camera2_id'] == camera2_id:
                # Update duration
                time_diff = current_timestamp - existing_match['last_updated']
                existing_match['match_duration'] += time_diff
                existing_match['last_updated'] = current_timestamp
                existing_match['similarity'] = max(existing_match['similarity'], similarity)

                # Update images if this is a better view (e.g., person is closer to camera)
                if similarity > existing_match['similarity']:
                    cam1_img_path = self.save_person_image(frame1, det1['bbox'], 1, camera1_id, current_timestamp)
                    cam2_img_path = self.save_person_image(frame2, det2['bbox'], 2, camera2_id, current_timestamp)

                    self.matched_images[camera1_id] = {
                        'cam1_image_path': cam1_img_path,
                        'cam2_image_path': cam2_img_path,
                        'timestamp': current_timestamp
                    }

    def update_tracker(self, frame, detections, tracker):
        detection_list = []

        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            features = self.extract_features(frame, det['original_bbox'])
            detection_list.append(Detection(bbox, confidence, features))

        tracker.predict()
        tracker.update(detection_list)

        tracked_detections = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            tracked_det = {
                'bbox': (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                'id': track.track_id,
                'confidence': track.confidence if hasattr(track, 'confidence') else 1.0
            }
            tracked_detections.append(tracked_det)

        return tracked_detections

    from datetime import datetime, timedelta

    def process_frames(self, frame1, frame2):
        # Previous detection code remains the same
        raw_detections1 = self.detect_people(frame1)
        raw_detections2 = self.detect_people(frame2)

        detections1 = self.update_tracker(frame1, raw_detections1, self.tracker1)
        detections2 = self.update_tracker(frame2, raw_detections2, self.tracker2)

        similarities = []
        current_timestamp = datetime.now()

        # Process detections in jaywalking region
        for i, det1 in enumerate(detections1):
            if self.is_in_jaywalking_region(det1['bbox']):
                features1 = self.extract_features(frame1, det1['bbox'])

                for j, det2 in enumerate(detections2):
                    features2 = self.extract_features(frame2, det2['bbox'])
                    similarity = 1 - cosine(features1, features2)

                    if similarity > self.feature_similarity_threshold:
                        similarities.append((i, j, similarity, features1, features2))
                        self.update_persistent_matches(det1, det2, similarity, frame1, frame2, current_timestamp)

                        # Log the match if it's been persistent
                        if (det1['id'] in self.persistent_matches and
                                self.persistent_matches[det1['id']]['match_duration'] > timedelta(
                                    seconds=5)):  # Adjust threshold as needed

                            self.logs.append({
                                'timestamp': current_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                'camera1_id': det1['id'],
                                'camera2_id': det2['id'],
                                'similarity': similarity,
                                'match_duration': str(self.persistent_matches[det1['id']]['match_duration']),
                                'camera1_bbox': det1['bbox'],
                                'camera2_bbox': det2['bbox'],
                                'camera1_image': self.matched_images[det1['id']]['cam1_image_path'],
                                'camera2_image': self.matched_images[det1['id']]['cam2_image_path']
                            })

        # Process existing matches for people who left jaywalking region
        for camera1_id in list(self.persistent_matches.keys()):
            match_info = self.persistent_matches[camera1_id]

            # If match has been persistent enough, continue tracking in camera 2
            if match_info['match_duration'] > timedelta(seconds=5):  # Adjust threshold as needed
                camera2_id = match_info['camera2_id']

                # Find person in camera 2
                for det2 in detections2:
                    if det2['id'] == camera2_id:
                        # Update camera 2 image
                        cam2_img_path = self.save_person_image(frame2, det2['bbox'], 2, camera2_id, current_timestamp)
                        if cam2_img_path:
                            self.matched_images[camera1_id]['cam2_image_path'] = cam2_img_path

        return detections1, detections2, similarities

    def save_logs(self):
        """Save logs with extended information"""
        if len(self.logs) > 0:
            logs_dir = 'logs'
            os.makedirs(logs_dir, exist_ok=True)
            log_file = os.path.join(logs_dir, 'jaywalking_log.txt')

            with open(log_file, 'w') as f:
                for log in self.logs:
                    f.write(f"Timestamp: {log['timestamp']}\n")
                    f.write(f"Camera1 ID: {log['camera1_id']}, Camera2 ID: {log['camera2_id']}\n")
                    f.write(f"Similarity: {log['similarity']:.2f}\n")
                    f.write(f"Match Duration: {log['match_duration']}\n")
                    f.write(f"Camera1 BBox: {log['camera1_bbox']}\n")
                    f.write(f"Camera2 BBox: {log['camera2_bbox']}\n")
                    f.write(f"Camera1 Image: {log['camera1_image']}\n")
                    f.write(f"Camera2 Image: {log['camera2_image']}\n")
                    f.write('\n')

def visualize_matches(frame1, frame2, detections1, detections2, similarities, jaywalking_region):
    vis_frame1 = frame1.copy()
    vis_frame2 = frame2.copy()

    # Draw jaywalking region
    cv2.polylines(vis_frame1, [jaywalking_region], True, (0, 255, 0), 2)

    colors = np.random.RandomState(42).randint(0, 255, (1000, 3))

    # Draw all detections in camera 1
    for i, det in enumerate(detections1):
        x1, y1, x2, y2 = det['bbox']

        # Check if person is jaywalking
        is_jaywalking = any(s[0] == i for s in similarities)

        # Use red for jaywalkers, blue for others
        color = (0, 0, 255) if is_jaywalking else (255, 0, 0)  # BGR format

        # Draw rectangle
        cv2.rectangle(vis_frame1, (x1, y1), (x2, y2), color, 2)

        status = f"ID{det['id']} JAYWALKING" if is_jaywalking else f"ID{det['id']}"

        # Draw ID/status with larger font
        cv2.putText(vis_frame1, status, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Draw all detections in camera 2 with similarity scores and matching IDs
    for j, det in enumerate(detections2):
        x1, y1, x2, y2 = det['bbox']
        color = (255, 0, 0)  # Default blue color

        # Draw rectangle
        cv2.rectangle(vis_frame2, (x1, y1), (x2, y2), color, 2)

        # Find all similarities and matching IDs for this detection
        matches = [(s[2], detections1[s[0]]['id']) for s in similarities if s[1] == j]

        # Draw ID and best similarity with matching ID if any
        if matches:
            # Sort by similarity to get the best match
            best_match = max(matches, key=lambda x: x[0])
            similarity, matching_id = best_match

            # Draw ID, matching ID and similarity score with larger font
            text = f"ID{det['id']} -> ID{matching_id} Sim:{similarity:.2f}"
            cv2.putText(vis_frame2, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        else:
            # Draw just the ID for unmatched detections
            cv2.putText(vis_frame2, f"ID{det['id']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    return vis_frame1, vis_frame2

def main():
    cap1 = cv2.VideoCapture('ayush_copy_3.mov')
    cap2 = cv2.VideoCapture('Ayush_face_copy.mp4')

    detector = PersonDetectionMatcher()
    frame_count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame_count += 1
        print(f"\nProcessing frame {frame_count}")

        # Process frames
        detections1, detections2, similarities = detector.process_frames(frame1, frame2)

        # Visualize results
        vis_frame1, vis_frame2 = visualize_matches(frame1, frame2,
                                                   detections1, detections2,
                                                   similarities, detector.jaywalking_region)

        # Display frames
        cv2.imshow('Camera 1', vis_frame1)
        cv2.imshow('Camera 2', vis_frame2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)

    # Save logs to a file
    if len(detector.logs) > 0:
        logs_dir = 'logs'
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, 'jaywalking_log.txt')
        with open(log_file, 'w') as f:
            for log in detector.logs:
                f.write(f"Timestamp: {log['timestamp']}\n")
                f.write(f"Camera1 ID: {log['camera1_id']}, Camera2 ID: {log['camera2_id']}\n")
                f.write(f"Similarity: {log['similarity']:.2f}\n")
                f.write(f"Camera1 BBox: {log['camera1_bbox']}\n")
                f.write(f"Camera2 BBox: {log['camera2_bbox']}\n")
                f.write('\n')
    detector.save_logs()

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()