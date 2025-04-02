import os
import numpy as np
from insightface_func.face_detect_crop_single import Face_detect_crop
from PIL import Image


class LandmarkDetector:
    def __init__(self, model_root="insightface_func/models/", ctx_id=-1):
        self.face_helper = Face_detect_crop(name="antelope", root=model_root)
        self.face_helper.prepare(ctx_id=ctx_id)  # -1 for CPU, or 0 for GPU

    def align_face(self, img):
        faces = self.face_helper.get(img, crop_size=112)
        if faces is None or len(faces) == 0:
            return None
        aligned_np = faces[0][0]  # unwrap the inner list
        aligned_img = Image.fromarray(aligned_np.astype(np.uint8))
        return aligned_img




# import numpy as np
# import onnxruntime as ort
# import cv2
# from insightface.utils.face_align import norm_crop


# class LandmarkDetector:
#     def __init__(self):
#         self.detector = ort.InferenceSession("../insightface_func/models/antelope/scrfd_10g_bnkps.onnx")
#         self.arcface_preprocessor = ort.InferenceSession("../insightface_func/models/antelope/glintr100.onnx")

#     def align_face(self, img):
#         input_blob = cv2.resize(img, (640, 640))
#         input_blob = input_blob.transpose(2, 0, 1).astype(np.float32)
#         input_blob = np.expand_dims(input_blob, axis=0)
#         outputs = self.detector.run(None, {"input.1": input_blob})
#         scores = outputs[0]     # output0
#         bboxes = outputs[1]     # output1
#         landmarks = outputs[2]  # output2

#         if landmarks.shape[1] == 0:
#             return None

#         # Take the first detection's landmarks
#         keypoints = landmarks[0][0]  # shape: (5, 2)
#         aligned = norm_crop(img, keypoints)
#         return aligned
