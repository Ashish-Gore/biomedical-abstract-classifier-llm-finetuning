"""
Pydantic models for FastAPI request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class ClassificationLabels(str, Enum):
    """Enum for classification labels."""
    CANCER = "Cancer"
    NON_CANCER = "Non-Cancer"


class AbstractRequest(BaseModel):
    """Request model for single abstract analysis."""
    abstract: str = Field(..., description="Research paper abstract text", min_length=10)
    abstract_id: Optional[str] = Field(None, description="Unique identifier for the abstract")


class BatchAbstractRequest(BaseModel):
    """Request model for batch abstract analysis."""
    abstracts: List[AbstractRequest] = Field(..., description="List of abstracts to analyze", min_items=1, max_items=100)


class ConfidenceScores(BaseModel):
    """Model for confidence scores."""
    cancer: float = Field(..., ge=0.0, le=1.0, description="Confidence score for cancer classification")
    non_cancer: float = Field(..., ge=0.0, le=1.0, description="Confidence score for non-cancer classification")


class ClassificationResult(BaseModel):
    """Model for classification results."""
    predicted_labels: List[ClassificationLabels] = Field(..., description="Predicted classification labels")
    confidence_scores: ConfidenceScores = Field(..., description="Confidence scores for each label")


class DiseaseExtractionResult(BaseModel):
    """Model for disease extraction results."""
    extracted_diseases: List[str] = Field(..., description="List of extracted diseases")
    cancer_diseases: List[str] = Field(..., description="List of cancer-related diseases")
    non_cancer_diseases: List[str] = Field(..., description="List of non-cancer diseases")
    total_diseases: int = Field(..., description="Total number of diseases extracted")
    cancer_count: int = Field(..., description="Number of cancer-related diseases")
    non_cancer_count: int = Field(..., description="Number of non-cancer diseases")


class AnalysisMetadata(BaseModel):
    """Model for analysis metadata."""
    text_length: int = Field(..., description="Length of original text")
    cleaned_length: int = Field(..., description="Length of cleaned text")
    disease_count: int = Field(..., description="Number of diseases extracted")
    cancer_disease_count: int = Field(..., description="Number of cancer diseases")
    non_cancer_disease_count: int = Field(..., description="Number of non-cancer diseases")


class AbstractAnalysisResult(BaseModel):
    """Complete analysis result for a single abstract."""
    abstract_id: str = Field(..., description="Unique identifier for the abstract")
    original_text: str = Field(..., description="Original abstract text (truncated)")
    cleaned_text: str = Field(..., description="Cleaned abstract text (truncated)")
    classification: ClassificationResult = Field(..., description="Classification results")
    disease_extraction: DiseaseExtractionResult = Field(..., description="Disease extraction results")
    is_cancer_related: bool = Field(..., description="Whether the abstract is cancer-related")
    analysis_metadata: AnalysisMetadata = Field(..., description="Analysis metadata")


class BatchAnalysisResult(BaseModel):
    """Result model for batch analysis."""
    results: List[AbstractAnalysisResult] = Field(..., description="Analysis results for each abstract")
    summary: Dict[str, Any] = Field(..., description="Summary statistics for the batch")


class PerformanceMetrics(BaseModel):
    """Model for performance metrics."""
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")


class ModelInfo(BaseModel):
    """Model information."""
    model_name: str = Field(..., description="Name of the model")
    model_path: str = Field(..., description="Path to the model")
    device: str = Field(..., description="Device the model is running on")
    quantization_enabled: bool = Field(..., description="Whether quantization is enabled")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Device information")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")
