import cv2
import numpy as np

def blend_back(original_img, generated_face, M_inv):
    # Warp generated face back to original image space
    warped_back = cv2.warpAffine(generated_face, M_inv, (original_img.shape[1], original_img.shape[0]))

    # Create binary face mask
    face_mask = cv2.cvtColor(generated_face, cv2.COLOR_RGB2GRAY)
    mask = cv2.warpAffine((face_mask > 10).astype(np.uint8) * 255, M_inv, (original_img.shape[1], original_img.shape[0]))

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # Alpha blend generated face and target
    blended = original_img.copy()
    for c in range(3):
        blended[:, :, c] = (mask / 255.0) * warped_back[:, :, c] + (1 - mask / 255.0) * original_img[:, :, c]

    return blended

