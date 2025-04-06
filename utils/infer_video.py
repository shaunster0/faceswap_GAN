import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from models.generator import AADGenerator
from models.encoder import ArcFaceIdentityEncoder, AttributeEncoder
from utils.landmark import LandmarkDetector
from utils.blend import blend_back  # Shared

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models
generator = AADGenerator().to(DEVICE)
identity_encoder = ArcFaceIdentityEncoder('arcface_model/backbone.pth').to(DEVICE)
attribute_encoder = AttributeEncoder().to(DEVICE)

ckpt = torch.load("checkpoints/checkpoint_epoch_10.pth", map_location=DEVICE)
generator.load_state_dict(ckpt['generator'])
attribute_encoder.load_state_dict(ckpt['attribute_encoder'])

generator.eval()
identity_encoder.eval()
attribute_encoder.eval()

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

inv_transform = transforms.Compose([
    transforms.Normalize([0], [2]),
    transforms.Normalize([-0.5]*3, [1])
])

face_aligner = LandmarkDetector()

# Prepare source
source_img = cv2.imread("sample_images/source.jpg")
source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
aligned_info = face_aligner.face_helper.get(source_img, return_trans_inv=True, crop_size=112)
aligned_face = Image.fromarray(aligned_info[0][0])
source_tensor = transform(aligned_face).unsqueeze(0).to(DEVICE)
z_id = identity_encoder(source_tensor)

# Open video
cap = cv2.VideoCapture("sample_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_swapped.mp4", fourcc, 25.0, (int(cap.get(3)), int(cap.get(4))))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    aligned_info = face_aligner.face_helper.get(frame_rgb, return_trans_inv=True, crop_size=112)

    if not aligned_info:
        print(f"[WARN] No face in frame {frame_idx}")
        out.write(frame)
        frame_idx += 1
        continue

    aligned_face, M_inv = aligned_info[0]
    face_tensor = transform(Image.fromarray(aligned_face)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        z_attrs = attribute_encoder(face_tensor)
        generated = generator(z_id, z_attrs)
        gen_img = generated.squeeze(0).cpu()
        gen_img = inv_transform(gen_img).clamp(0, 1).permute(1, 2, 0).numpy()
        gen_img = (gen_img * 255).astype(np.uint8)

        blended = blend_back(frame_rgb, gen_img, M_inv)
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        out.write(blended_bgr)

    frame_idx += 1

cap.release()
out.release()
print("[✓] Video written to output_swapped.mp4")
