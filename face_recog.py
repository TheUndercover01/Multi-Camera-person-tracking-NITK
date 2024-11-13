import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm
import random
from datetime import datetime


class FaceRecognitionSystem:
    def __init__(self, embeddings_path='reference_embeddings.npy', device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.embeddings_path = embeddings_path

        # Initialize MTCNN for face detection
        self.face_detector = MTCNN(device=self.device)

        # Initialize ResNet50 model for feature extraction
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove the final fully connected layer
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load or initialize reference embeddings
        self.reference_embeddings = []
        if os.path.exists(embeddings_path):
            self.reference_embeddings = np.load(embeddings_path)
            print(f"Loaded embeddings from {embeddings_path}")
        else:
            print("No embeddings found. Please train the model first.")

    def extract_face(self, image_path):
        """Extracts face from the image using MTCNN"""
        try:
            img = Image.open(image_path).convert('RGB')
            face = self.face_detector(img)
            return face
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def extract_features(self, face_tensor):
        """Extracts features from the face tensor using ResNet50"""
        with torch.no_grad():
            face_tensor = face_tensor.to(self.device)
            features = self.feature_extractor(face_tensor.unsqueeze(0))
            features = F.normalize(features, p=2, dim=1)
            return features.cpu().numpy().flatten()

    def predict(self, image_path, threshold=0.1):
        """Predicts if the face in the image matches the reference embeddings"""
        if not self.reference_embeddings.any():
            raise Exception("No embeddings loaded. Please train the system first.")

        face_tensor = self.extract_face(image_path)
        if face_tensor is None:
            return {'name': 'Ayush', 'confidence': 0.0, 'match': False, 'error': 'No face detected'}

        face_features = self.extract_features(face_tensor)
        similarities = [np.dot(face_features, ref) for ref in self.reference_embeddings]
        confidence = max(similarities)

        # Display name "Ayush" with confidence regardless of match threshold
        return {'name': 'Ayush', 'match': confidence >= threshold, 'confidence': confidence}

    def parse_log_file(self, log_path):
        """Parses the log file to extract paths of Camera2 images and their timestamps"""
        camera2_images_with_timestamp = []
        with open(log_path, 'r') as file:
            lines = file.readlines()

        entry = {}
        for line in lines:
            if line.startswith("Camera2 Image:"):
                entry["camera2_image"] = line.split(": ", 1)[1].strip()
            elif line.startswith("Timestamp:"):
                entry["timestamp"] = line.split(": ", 1)[1].strip()
            elif line.startswith("Similarity:"):
                entry["Similarity"] = line.split(": ", 1)[1].strip()
                # Ensure both Camera2 image path and timestamp are found
                if "camera2_image" in entry and "timestamp" in entry and "Similarity" in entry:
                    camera2_images_with_timestamp.append(entry)
                    entry = {}  # Reset entry for the next log

        return camera2_images_with_timestamp

    def generate_log(self, entity, timestamp, jay_walking_conf, Conf_in_name):
        """Generates a dummy log for a matched entity."""
        is_jaywalking = True

        return {
            "timestamp": timestamp,
            "entity": entity,  # Here, entity will be the matched person
            "jay_walking_conf": jay_walking_conf,  # Confidence related to jaywalking
            "is_jaywalking": is_jaywalking,
            "Confidence_name": Conf_in_name


        }

    def recognize_images_from_log(self, log_path):
        """Processes Camera2 images listed in the log file for recognition"""
        camera2_images_with_timestamp = self.parse_log_file(log_path)
        results = []

        print(f"Processing images from log file: {log_path}")
        for entry in camera2_images_with_timestamp:
            image_path = entry["camera2_image"]
            timestamp = entry["timestamp"]
            jay_walking_conf = float(entry["Similarity"])

            if os.path.exists(image_path):
                result = self.predict(image_path)

                if result['match']:
                    # Stop recognition once a match is found
                    print(f"\nMatch found for {image_path}!")
                    print(f"Timestamp: {timestamp}")
                    print(f"Confidence: {result['confidence']:.2%}")


                    # Generate and print the dummy log with updated names
                    dummy_log = self.generate_log(result['name'], timestamp, jay_walking_conf, result['confidence'] )
                    print(f"Generated Dummy Log: {dummy_log}")

                    # Add the result and stop processing
                    results.append({
                        'image_path': image_path,
                        'name': result['name'],
                        'match': result['match'],
                        'confidence': result['confidence'],
                        'timestamp': timestamp
                    })
                    break  # Stop processing after the first match

                else:
                    # No match found for this image
                    print(f"\nNo match for {image_path}")
                    results.append({
                        'image_path': image_path,
                        'name': result['name'],
                        'match': result['match'],
                        'confidence': result['confidence'],
                        'timestamp': timestamp
                    })
            else:
                print(f"Image not found: {image_path}")
                results.append({
                    'image_path': image_path,
                    'name': 'Ayush',
                    'match': False,
                    'confidence': 0.0,
                    'error': 'Image not found',
                    'timestamp': timestamp
                })

        return results


if __name__ == "__main__":
    # Configuration
    log_path = './logs/jaywalking_log.txt'
    embeddings_path = 'reference_embeddings.npy'

    # Initialize the face recognition system
    face_recognition_system = FaceRecognitionSystem(embeddings_path=embeddings_path)

    # Check if embeddings file exists before running the system
    if not os.path.exists(embeddings_path):
        print("Embeddings file not found. Please train the model first.")
    else:
        # Process the log file for recognition
        face_recognition_system.recognize_images_from_log(log_path)
