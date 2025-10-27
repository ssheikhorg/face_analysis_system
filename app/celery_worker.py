import os
from celery import Celery
import logging
from app.processors import FacialSVGProcessor
from app.models import ProcessRequest

logger = logging.getLogger(__name__)

# Celery configuration
celery_app = Celery(
    "facial_svg_worker",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

processor = FacialSVGProcessor()


@celery_app.task(bind=True, name="process_image_task")
def process_image_task(self, request_data):
    """Celery task to process facial image"""
    try:
        # Simulate processing delay (for bonus feature demonstration)
        import time

        time.sleep(20)  # 20 second delay as mentioned in requirements

        request = ProcessRequest(**request_data)
        result = processor.process_request(request)
        return result.dict()
    except Exception as e:
        logger.error(f"Error in celery task: {e}")
        raise self.retry(exc=e, countdown=60)
