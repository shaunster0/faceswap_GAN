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
        self.root_dir = root_dir
        self.transform = transform
        self.landmark_detector = LandmarkDetector()
        self.identity_map = self._build_identity_map()

    def _build_identity_map(self):
        identity_map = {}
        for person_name in os.listdir(self.root_dir):
            person_dir = os.path.join(self.root_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            images = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith('.jpg')]
            if len(images) >= 1:
                identity_map[person_name] = images
        return identity_map

    def __len__(self):
        return sum(len(imgs) for imgs in self.identity_map.values())

    def __getitem__(self, idx):
        # Pick a random target identity
        target_identity = random.choice(list(self.identity_map.keys()))
        target_path = random.choice(self.identity_map[target_identity])

        # Pick a different source identity
        source_identity = target_identity
        while source_identity == target_identity:
            source_identity = random.choice(list(self.identity_map.keys()))
        source_path = random.choice(self.identity_map[source_identity])

        source_img = self.load_and_align(source_path)
        target_img = self.load_and_align(target_path)

        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)

        return source_img, target_img

    def load_and_align(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aligned = self.landmark_detector.align_face(img)

        if aligned is None:
            print(f"[WARN] No face detected in: {path}")
            img = cv2.resize(img, (112, 112))
            aligned = Image.fromarray(img)  # manually convert to PIL
        return aligned 

