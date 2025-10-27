from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
from prometheus_client import make_asgi_app, Counter, Histogram
import time
from app.models import ProcessRequest, ProcessResponse, ErrorResponse, JobStatusResponse
from app.processors import FacialSVGProcessor
from app.celery_worker import process_image_task
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Facial SVG Service",
    description="Generate SVG contours from facial landmarks and segmentation",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "request_count", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "request_duration_seconds", "Request duration", ["method", "endpoint"]
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = FacialSVGProcessor()
job_status = {}


@app.post(
    "/api/v1/frontal/crop/submit",
    response_model=JobStatusResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Processing Error"},
    },
)
async def process_facial_image(
    request: ProcessRequest, background_tasks: BackgroundTasks
):
    """
    Process facial image and return SVG with contour masks
    """
    start_time = time.time()

    try:
        # Basic validation
        landmarks = processor._parse_landmarks(request.landmarks)
        if len(landmarks) < 10:
            REQUEST_COUNT.labels("POST", "/api/v1/frontal/crop/submit", "422").inc()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Insufficient landmarks detected - no face found",
            )

        # Create job ID
        job_id = str(uuid.uuid4())

        # For immediate processing (remove delay for load testing)
        result = processor.process_request(request)
        job_status[job_id] = {"status": "completed", "result": result}

        REQUEST_COUNT.labels("POST", "/api/v1/frontal/crop/submit", "200").inc()
        REQUEST_DURATION.labels("POST", "/api/v1/frontal/crop/submit").observe(
            time.time() - start_time
        )

        return JobStatusResponse(id=job_id, status="completed", result=result)

    except HTTPException:
        REQUEST_COUNT.labels("POST", "/api/v1/frontal/crop/submit", "422").inc()
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        REQUEST_COUNT.labels("POST", "/api/v1/frontal/crop/submit", "500").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}",
        )


# Async version with Celery (for bonus points)
@app.post("/api/v1/frontal/crop/submit-async")
async def process_facial_image_async(request: ProcessRequest):
    """Async version using Celery"""
    try:
        # Basic validation
        landmarks = processor._parse_landmarks(request.landmarks)
        if len(landmarks) < 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Insufficient landmarks detected - no face found",
            )

        job_id = str(uuid.uuid4())

        # Send task to Celery
        task = process_image_task.apply_async(args=[request.dict()], task_id=job_id)
        job_status[job_id] = {"status": "pending", "task_id": task.id}

        return JobStatusResponse(id=job_id, status="pending")

    except Exception as e:
        logger.error(f"Error submitting async job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/frontal/crop/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of async job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = job_status[job_id]

    if job_data["status"] == "pending":
        # Check Celery task status
        from app.celery_worker import process_image_task

        task = process_image_task.AsyncResult(job_data["task_id"])

        if task.ready():
            if task.successful():
                job_status[job_id] = {"status": "completed", "result": task.result}
                return JobStatusResponse(
                    id=job_id, status="completed", result=task.result
                )
            else:
                job_status[job_id] = {"status": "failed", "error": str(task.result)}
                return JobStatusResponse(id=job_id, status="failed")
        else:
            return JobStatusResponse(id=job_id, status="pending")

    return JobStatusResponse(
        id=job_id, status=job_data["status"], result=job_data.get("result")
    )


@app.get("/")
async def root():
    return {"message": "Facial SVG Service API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
