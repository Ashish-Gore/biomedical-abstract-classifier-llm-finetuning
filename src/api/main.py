"""
FastAPI main application for the research paper analysis API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from pathlib import Path
import uvicorn
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
from api.endpoints import router, initialize_analyzer
from api.models import ErrorResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Research Paper Analysis API",
    description="API for classifying research paper abstracts and extracting disease information",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["analysis"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    try:
        # Get model path from environment or use default
        model_path = os.getenv("MODEL_PATH", "./models")
        use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
        
        # Check if model exists
        if not Path(model_path).exists():
            logger.warning(f"Model path {model_path} does not exist. Using placeholder.")
            # For demo purposes, we'll create a placeholder
            # In production, this should be handled differently
            return
        
        # Initialize analyzer
        initialize_analyzer(model_path, use_quantization)
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}")
        # Don't raise here to allow the API to start even without model


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Research Paper Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/")
async def api_root():
    """API root endpoint."""
    return {
        "message": "Research Paper Analysis API v1",
        "endpoints": {
            "health": "/api/v1/health",
            "classify": "/api/v1/classify",
            "classify_batch": "/api/v1/classify/batch",
            "extract_diseases": "/api/v1/diseases/extract",
            "metrics": "/api/v1/metrics",
            "stats": "/api/v1/stats"
        }
    }


def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the Research Paper Analysis API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--model_path', default='./models', help='Path to the trained model')
    parser.add_argument('--use_quantization', action='store_true', 
                       help='Use quantization for memory efficiency')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["USE_QUANTIZATION"] = str(args.use_quantization).lower()
    
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
