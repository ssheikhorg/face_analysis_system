from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from rich.logging import RichHandler

from app.api.endpoints import router as api_router

# Set up rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Facial SVG Service",
    description="Generate SVG contours from facial landmarks and segmentation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Facial SVG Service API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
