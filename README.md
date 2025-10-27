# Facial SVG Service

A FastAPI-based service that generates SVG facial contour masks from images, landmarks, and segmentation maps.

## Features

- FastAPI REST API with validation and error handling
- Facial landmark processing with automatic face alignment
- SVG generation with smooth contour overlays
- Async processing with Celery and Redis
- Docker Compose setup with multiple services
- Prometheus metrics and monitoring
- Load testing optimized mode

## API Endpoints

- `POST /api/v1/frontal/crop/submit` - Process facial image and return SVG contours
- `POST /api/v1/frontal/crop/submit-async` - Submit async processing job
- `GET /api/v1/frontal/crop/status/{job_id}` - Check job status
- `GET /metrics` - Prometheus metrics
- `GET /health` - Health check

## Quick Start

```bash
# Clone and run
docker-compose up --build

# API will be available at http://localhost:8000