import cv2
import numpy as np
from typing import List, Tuple
import logging
from app.models.schemas import LandmarkPoint

logger = logging.getLogger(__name__)


class FaceAligner:
    def __init__(self):
        # Key landmark indices for alignment (approximate - need to verify from landmark_annotations.jpg)
        self.LEFT_EYE_INDICES = [33, 133]  # Need verification
        self.RIGHT_EYE_INDICES = [362, 263]  # Need verification
        self.NOSE_TIP_INDEX = 1  # Need verification

    def estimate_rotation_angle(self, landmarks: List[LandmarkPoint]) -> float:
        """Estimate face rotation angle from landmarks"""
        try:
            # Convert to numpy array
            points = np.array([[lm.x, lm.y] for lm in landmarks])

            # Find eye centers (this needs proper landmark mapping)
            left_eye_center = points[33]  # Approximate left eye center
            right_eye_center = points[263]  # Approximate right eye center

            # Calculate angle between eyes
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))

            logger.info(f"Estimated rotation angle: {angle:.2f} degrees")
            return angle

        except Exception as e:
            logger.warning(f"Could not estimate rotation angle: {e}")
            return 0.0

    def align_face(self, image: np.ndarray, landmarks: List[LandmarkPoint]) -> Tuple[np.ndarray, List[LandmarkPoint]]:
        """Align face to upright position"""
        angle = self.estimate_rotation_angle(landmarks)

        if abs(angle) < 5:  # Only rotate if significant angle
            return image, landmarks

        # Rotate image
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)

        # Rotate landmarks
        aligned_landmarks = []
        for lm in landmarks:
            point = np.array([lm.x, lm.y, 1])
            rotated_point = rotation_matrix @ point
            aligned_landmarks.append(LandmarkPoint(x=rotated_point[0], y=rotated_point[1]))

        logger.info(f"Aligned face by {angle:.2f} degrees")
        return aligned_image, aligned_landmarks