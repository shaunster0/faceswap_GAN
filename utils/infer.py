'''
usage:
python infer.py \
  --source ./sample_images/source.jpg \
  --target ./sample_images/target.jpg \
  --checkpoint ./checkpoints/checkpoint_epoch_10.pth \
  --output ./outputs/final_result.jpg
  
'''

import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import os

from models.generator import AADGenerator
from models.encoder import ArcFaceIdentityEncoder, AttributeEncoder
from utils.landmark import LandmarkDetector
from utils.blend import blend_back

# ───── argparse ─────
parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--target', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───── Transforms ─────
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

inv_transform = transforms.Compose([
    transforms.Normalize([0], [2]),  # unscale std
    transforms.Normalize([-0.5]*3, [1])  # uncenter
])

# ───── Models ─────
generator = AADGenerator().to(DEVICE)
identity_encoder = ArcFaceIdentityEncoder('arcface_model/backbone.pth').to(DEVICE)
attribute_encoder = AttributeEncoder().to(DEVICE)

# ───── Load Checkpoint ─────
ckpt = torch.load(args.checkpoint, map_location=DEVICE)
generator.load_state_dict(ckpt['generator'])
attribute_encoder.load_state_dict(ckpt['attribute_encoder'])
generator.eval()
identity_encoder.eval()
attribute_encoder.eval()

# ───── Face Detection ─────
face_aligner = LandmarkDetector()

# ───── Load and Align Face ─────
def align_and_get_transform(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aligned_info = face_aligner.face_helper.get(img, return_trans_inv=True, crop_size=112)
    if not aligned_info:
        raise RuntimeError(f"No face detected in {img_path}")

    aligned_img, M_inv = aligned_info[0]
    aligned_img_pil = Image.fromarray(aligned_img)
    tensor = transform(aligned_img_pil).unsqueeze(0).to(DEVICE)
    return tensor, img, M_inv

# ───── Inference ─────
with torch.no_grad():
    source_tensor, _, _ = align_and_get_transform(args.source)
    target_tensor, target_img_raw, M_inv = align_and_get_transform(args.target)

    z_id = identity_encoder(source_tensor)
    z_attrs = attribute_encoder(target_tensor)
    generated = generator(z_id, z_attrs)

    # Unnormalize generated image
    output_img = generated.squeeze(0).cpu()
    output_img = inv_transform(output_img).clamp(0, 1).permute(1, 2, 0).numpy()
    output_img = (output_img * 255).astype(np.uint8)

    # Warp back to target image
    # warped_back = cv2.warpAffine(output_img, M_inv, (target_img_raw.shape[1], target_img_raw.shape[0]))

    # # Create mask from generated face
    # face_mask = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
    # mask = cv2.warpAffine((face_mask > 10).astype(np.uint8)*255, M_inv, (target_img_raw.shape[1], target_img_raw.shape[0]))

    # # Feather the mask for blending
    # mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # # Blend into original target
    # blended = target_img_raw.copy()
    # for c in range(3):
    #     blended[:, :, c] = (mask / 255.0) * warped_back[:, :, c] + (1 - mask / 255.0) * target_img_raw[:, :, c]
    
    blended = blend_back(target_img_raw, output_img, M_inv)

    # Save result
    Image.fromarray(blended.astype(np.uint8)).save(args.output)
    print(f"[✓] Swapped face blended and saved to {args.output}")

