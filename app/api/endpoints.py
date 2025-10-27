from fastapi import APIRouter, HTTPException, status
import logging
from app.models.schemas import ProcessRequest, ProcessResponse, JobStatusResponse, ErrorResponse
from app.services.image_processor import ImageProcessor

logger = logging.getLogger(__name__)
router = APIRouter()
image_processor = ImageProcessor()


@router.post(
    "/frontal/crop/submit",
    response_model=ProcessResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Processing Error"}
    }
)
async def process_facial_image(request: ProcessRequest):
    """
    Process facial image and return SVG with contour masks
    """
    try:
        # Basic validation - check if we have enough landmarks for a face
        if len(request.landmarks) < 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Insufficient landmarks detected - no face found"
            )

        logger.info(f"Processing image with {len(request.landmarks)} landmarks")

        result = image_processor.process_request(request)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )


# Placeholder for bonus endpoints
@router.get("/frontal/crop/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of async job (for bonus implementation)"""
    return JobStatusResponse(id=job_id, status="completed")  # Placeholder