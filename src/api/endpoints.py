"""
FastAPI endpoints for the research paper analysis API.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import logging
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .models import (
    AbstractRequest, BatchAbstractRequest, AbstractAnalysisResult,
    BatchAnalysisResult, PerformanceMetrics, ModelInfo, HealthResponse,
    ErrorResponse, ClassificationResult, DiseaseExtractionResult,
    AnalysisMetadata, ConfidenceScores
)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from inference import ResearchPaperAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer: ResearchPaperAnalyzer = None

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=4)

# Create router
router = APIRouter()


def get_analyzer() -> ResearchPaperAnalyzer:
    """Dependency to get the analyzer instance."""
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is properly initialized."
        )
    return analyzer


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global analyzer
    
    return HealthResponse(
        status="healthy" if analyzer is not None else "unhealthy",
        model_loaded=analyzer is not None,
        device=str(analyzer.device) if analyzer else "unknown",
        timestamp=datetime.now().isoformat()
    )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info(analyzer: ResearchPaperAnalyzer = Depends(get_analyzer)):
    """Get information about the loaded model."""
    return ModelInfo(
        model_name="Cancer Classification Model",
        model_path=analyzer.model_path,
        device=str(analyzer.device),
        quantization_enabled=analyzer.use_quantization
    )


@router.post("/classify", response_model=AbstractAnalysisResult)
async def classify_abstract(
    request: AbstractRequest,
    analyzer: ResearchPaperAnalyzer = Depends(get_analyzer)
):
    """Classify a single abstract and extract diseases."""
    try:
        logger.info(f"Classifying abstract: {request.abstract_id or 'unknown'}")
        
        # Analyze the abstract
        result = analyzer.analyze_abstract(
            text=request.abstract,
            abstract_id=request.abstract_id
        )
        
        # Convert to response model
        response = AbstractAnalysisResult(
            abstract_id=result["abstract_id"],
            original_text=result["original_text"],
            cleaned_text=result["cleaned_text"],
            classification=ClassificationResult(
                predicted_labels=result["classification"]["predicted_labels"],
                confidence_scores=ConfidenceScores(
                    cancer=result["classification"]["confidence_scores"]["Cancer"],
                    non_cancer=result["classification"]["confidence_scores"]["Non-Cancer"]
                )
            ),
            disease_extraction=DiseaseExtractionResult(
                extracted_diseases=result["disease_extraction"]["extracted_diseases"],
                cancer_diseases=result["disease_extraction"]["cancer_diseases"],
                non_cancer_diseases=result["disease_extraction"]["non_cancer_diseases"],
                total_diseases=result["disease_extraction"]["total_diseases"],
                cancer_count=result["disease_extraction"]["cancer_count"],
                non_cancer_count=result["disease_extraction"]["non_cancer_count"]
            ),
            is_cancer_related=result["is_cancer_related"],
            analysis_metadata=AnalysisMetadata(
                text_length=result["analysis_metadata"]["text_length"],
                cleaned_length=result["analysis_metadata"]["cleaned_length"],
                disease_count=result["analysis_metadata"]["disease_count"],
                cancer_disease_count=result["analysis_metadata"]["cancer_disease_count"],
                non_cancer_disease_count=result["analysis_metadata"]["non_cancer_disease_count"]
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error classifying abstract: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing abstract: {str(e)}"
        )


@router.post("/classify/batch", response_model=BatchAnalysisResult)
async def classify_batch(
    request: BatchAbstractRequest,
    background_tasks: BackgroundTasks,
    analyzer: ResearchPaperAnalyzer = Depends(get_analyzer)
):
    """Classify multiple abstracts in batch."""
    try:
        logger.info(f"Processing batch of {len(request.abstracts)} abstracts")
        
        # Extract abstracts and IDs
        abstracts = [item.abstract for item in request.abstracts]
        abstract_ids = [item.abstract_id for item in request.abstracts]
        
        # Process in background to avoid timeout
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            analyzer.analyze_batch,
            abstracts,
            abstract_ids
        )
        
        # Convert results to response models
        response_results = []
        for result in results:
            response_results.append(AbstractAnalysisResult(
                abstract_id=result["abstract_id"],
                original_text=result["original_text"],
                cleaned_text=result["cleaned_text"],
                classification=ClassificationResult(
                    predicted_labels=result["classification"]["predicted_labels"],
                    confidence_scores=ConfidenceScores(
                        cancer=result["classification"]["confidence_scores"]["Cancer"],
                        non_cancer=result["classification"]["confidence_scores"]["Non-Cancer"]
                    )
                ),
                disease_extraction=DiseaseExtractionResult(
                    extracted_diseases=result["disease_extraction"]["extracted_diseases"],
                    cancer_diseases=result["disease_extraction"]["cancer_diseases"],
                    non_cancer_diseases=result["disease_extraction"]["non_cancer_diseases"],
                    total_diseases=result["disease_extraction"]["total_diseases"],
                    cancer_count=result["disease_extraction"]["cancer_count"],
                    non_cancer_count=result["disease_extraction"]["non_cancer_count"]
                ),
                is_cancer_related=result["is_cancer_related"],
                analysis_metadata=AnalysisMetadata(
                    text_length=result["analysis_metadata"]["text_length"],
                    cleaned_length=result["analysis_metadata"]["cleaned_length"],
                    disease_count=result["analysis_metadata"]["disease_count"],
                    cancer_disease_count=result["analysis_metadata"]["cancer_disease_count"],
                    non_cancer_disease_count=result["analysis_metadata"]["non_cancer_disease_count"]
                )
            ))
        
        # Generate summary
        summary = analyzer.get_performance_summary(results)
        
        return BatchAnalysisResult(
            results=response_results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch: {str(e)}"
        )


@router.get("/metrics", response_model=PerformanceMetrics)
async def get_metrics(analyzer: ResearchPaperAnalyzer = Depends(get_analyzer)):
    """Get model performance metrics."""
    try:
        # This would typically load from a saved metrics file
        # For now, return placeholder metrics
        return PerformanceMetrics(
            accuracy=0.92,
            f1_score=0.86,
            precision=0.89,
            recall=0.84,
            confusion_matrix=[[350, 50], [30, 570]]
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics: {str(e)}"
        )


@router.post("/diseases/extract")
async def extract_diseases_only(
    request: AbstractRequest,
    analyzer: ResearchPaperAnalyzer = Depends(get_analyzer)
):
    """Extract only diseases from an abstract without classification."""
    try:
        logger.info(f"Extracting diseases from abstract: {request.abstract_id or 'unknown'}")
        
        # Extract diseases
        disease_result = analyzer.extract_diseases(request.abstract)
        
        return {
            "abstract_id": request.abstract_id or f"abstract_{hash(request.abstract) % 10000}",
            "extracted_diseases": disease_result["extracted_diseases"],
            "cancer_diseases": disease_result["cancer_diseases"],
            "non_cancer_diseases": disease_result["non_cancer_diseases"],
            "total_diseases": disease_result["total_diseases"],
            "cancer_count": disease_result["cancer_count"],
            "non_cancer_count": disease_result["non_cancer_count"]
        }
        
    except Exception as e:
        logger.error(f"Error extracting diseases: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting diseases: {str(e)}"
        )


@router.get("/stats")
async def get_statistics(analyzer: ResearchPaperAnalyzer = Depends(get_analyzer)):
    """Get general statistics about the service."""
    return {
        "service_name": "Research Paper Analysis API",
        "version": "1.0.0",
        "model_loaded": analyzer is not None,
        "device": str(analyzer.device) if analyzer else "unknown",
        "supported_operations": [
            "single_abstract_classification",
            "batch_abstract_classification",
            "disease_extraction",
            "performance_metrics"
        ],
        "timestamp": datetime.now().isoformat()
    }


def initialize_analyzer(model_path: str, use_quantization: bool = True):
    """Initialize the global analyzer instance."""
    global analyzer
    try:
        logger.info(f"Initializing analyzer with model: {model_path}")
        analyzer = ResearchPaperAnalyzer(model_path, use_quantization)
        logger.info("Analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {str(e)}")
        raise
