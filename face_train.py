# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm


class FaceRecognitionTrainer:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        # Initialize MTCNN for face detection
        self.face_detector = MTCNN(device=self.device)

        # Initialize ResNet50 model for feature extraction
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove final layer
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # Initialize transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_face(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            face = self.face_detector(img)
            return face
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def extract_features(self, face_tensor):
        with torch.no_grad():
            face_tensor = face_tensor.to(self.device)
            features = self.feature_extractor(face_tensor.unsqueeze(0))
            features = F.normalize(features, p=2, dim=1)
            return features.cpu().numpy().flatten()

    def train(self, training_dir='face/Ayush', output_file='reference_embeddings.npy'):
        embeddings = []
        for img_name in tqdm(os.listdir(training_dir)):
            img_path = os.path.join(training_dir, img_name)
            face_tensor = self.extract_face(img_path)
            if face_tensor is not None:
                embeddings.append(self.extract_features(face_tensor))

        np.save(output_file, embeddings)
        print(f"Training complete. Saved embeddings to {output_file}")


if __name__ == "__main__":
    trainer = FaceRecognitionTrainer()
    trainer.train()
