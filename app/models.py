from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class LandmarkPoint(BaseModel):
    x: float
    y: float


class ProcessRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    landmarks: List[LandmarkPoint] = Field(..., description="List of facial landmarks")
    segmentation_map: str = Field(..., description="Base64 encoded segmentation map")


class ProcessResponse(BaseModel):
    svg: str = Field(..., description="Base64 encoded SVG")
    mask_contours: Dict[int, List] = Field(
        ..., description="Contour data for each region"
    )


class JobStatusResponse(BaseModel):
    id: str
    status: str
    result: Optional[ProcessResponse] = None


class ErrorResponse(BaseModel):
    detail: str
