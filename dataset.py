from torch.utils.data import Dataset
from PIL import Image
import os
import random
import random
import numpy as np
import cv2
from utils.landmark import LandmarkDetector


class LFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.identity_map = {}
        self.landmark_detector = LandmarkDetector()

        for person_name in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person_name)
            if os.path.isdir(person_dir):
                images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
                if len(images) >= 2:
                    self.image_paths.extend(images)
                    self.identity_map[person_name] = images

    def __len__(self):
        return len(self.image_paths)

    def load_and_align(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aligned = self.landmark_detector.align_face(img)

        if aligned is None:
            print(f"[WARN] No face detected in: {path}")
            img = cv2.resize(img, (112, 112))
            aligned = Image.fromarray(img)  # manually convert to PIL
        return aligned 

    def __getitem__(self, idx):
        target_path = self.image_paths[idx]
        person_name = os.path.basename(os.path.dirname(target_path))
        source_path = random.choice(self.identity_map[person_name])

        source_img = self.load_and_align(source_path)
        target_img = self.load_and_align(target_path)

        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)

        return source_img, target_img
