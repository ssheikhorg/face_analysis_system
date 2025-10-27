import base64
import cv2
import numpy as np
from typing import List, Dict, Any
import logging
from io import BytesIO
from PIL import Image

from app.models.schemas import LandmarkPoint, ProcessResponse
from app.services.face_aligner import FaceAligner
from app.services.svg_generator import SVGGenerator

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self):
        self.face_aligner = FaceAligner()
        self.svg_generator = SVGGenerator()

    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise ValueError("Invalid image data")

    def process_segmentation_map(self, segmentation_base64: str, landmarks: List[LandmarkPoint]) -> Dict[
        int, List[LandmarkPoint]]:
        """Process segmentation map to extract region contours"""
        # This is a simplified version - you'll need to implement proper segmentation processing
        # based on the actual segmentation_map.png structure

        segmentation_image = self.base64_to_image(segmentation_base64)

        # Placeholder: Create dummy contours based on landmark regions
        # You'll need to replace this with actual segmentation processing
        contours = {}

        # Example: Create some basic regions from landmarks
        # You'll need to map this to the actual segmentation regions
        if len(landmarks) > 10:
            # Cheek region (example)
            cheek_points = landmarks[100:120]  # This needs proper mapping
            contours[1] = cheek_points

            # Forehead region (example)
            forehead_points = landmarks[10:30]  # This needs proper mapping
            contours[2] = forehead_points

            # Add more regions based on your segmentation map analysis

        logger.info(f"Extracted {len(contours)} regions from segmentation")
        return contours

    def process_request(self, request_data) -> ProcessResponse:
        """Main processing function"""
        try:
            # Decode input image
            original_image = self.base64_to_image(request_data.image)

            # Align face
            aligned_image, aligned_landmarks = self.face_aligner.align_face(
                original_image, request_data.landmarks
            )

            # Process segmentation map to get contours
            contours = self.process_segmentation_map(
                request_data.segmentation_map, aligned_landmarks
            )

            # Generate SVG
            height, width = aligned_image.shape[:2]
            svg_string = self.svg_generator.generate_svg(contours, width, height)
            svg_base64 = self.svg_generator.encode_svg_to_base64(svg_string)

            # Prepare mask contours data
            mask_contours = {}
            for region_id, points in contours.items():
                mask_contours[region_id] = [[p.x, p.y] for p in points]

            return ProcessResponse(
                svg=svg_base64,
                mask_contours=mask_contours
            )

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise