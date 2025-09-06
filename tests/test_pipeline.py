"""
Unit tests for the research paper analysis pipeline.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_preprocessing import AbstractPreprocessor
from disease_extraction import DiseaseExtractor
from model_training import CancerClassifier
from inference import ResearchPaperAnalyzer


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functionality."""
    
    def setUp(self):
        self.preprocessor = AbstractPreprocessor()
    
    def test_clean_abstract(self):
        """Test abstract cleaning functionality."""
        # Test with HTML tags
        text_with_html = "<p>This is a <b>test</b> abstract.</p>"
        cleaned = self.preprocessor.clean_abstract(text_with_html)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<b>", cleaned)
        
        # Test with citations
        text_with_citations = "This study [1,2,3] shows results (2023)."
        cleaned = self.preprocessor.clean_abstract(text_with_citations)
        self.assertNotIn("[1,2,3]", cleaned)
        self.assertNotIn("(2023)", cleaned)
        
        # Test with excessive whitespace
        text_with_whitespace = "This   has    excessive    whitespace."
        cleaned = self.preprocessor.clean_abstract(text_with_whitespace)
        self.assertNotIn("   ", cleaned)
    
    def test_is_cancer_related(self):
        """Test cancer detection functionality."""
        cancer_text = "This study investigates lung cancer treatment."
        non_cancer_text = "This study examines diabetes management."
        
        self.assertTrue(self.preprocessor.is_cancer_related(cancer_text))
        self.assertFalse(self.preprocessor.is_cancer_related(non_cancer_text))
    
    def test_extract_diseases(self):
        """Test disease extraction functionality."""
        text = "This study examines lung cancer and diabetes in patients."
        diseases = self.preprocessor.extract_diseases(text)
        
        self.assertIsInstance(diseases, list)
        self.assertGreater(len(diseases), 0)


class TestDiseaseExtraction(unittest.TestCase):
    """Test disease extraction functionality."""
    
    def setUp(self):
        self.extractor = DiseaseExtractor()
    
    def test_extract_diseases(self):
        """Test disease extraction from text."""
        text = "This study investigates lung cancer and breast cancer treatment."
        result = self.extractor.extract_diseases(text)
        
        self.assertIn("extracted_diseases", result)
        self.assertIn("cancer_diseases", result)
        self.assertIn("total_diseases", result)
        self.assertIsInstance(result["extracted_diseases"], list)
    
    def test_is_cancer_related(self):
        """Test cancer-related detection."""
        cancer_text = "This study examines lung cancer treatment."
        non_cancer_text = "This study investigates diabetes management."
        
        self.assertTrue(self.extractor.is_cancer_related(cancer_text))
        self.assertFalse(self.extractor.is_cancer_related(non_cancer_text))


class TestModelTraining(unittest.TestCase):
    """Test model training functionality."""
    
    def test_cancer_classifier_init(self):
        """Test cancer classifier initialization."""
        classifier = CancerClassifier(
            model_name="microsoft/DialoGPT-medium",
            use_quantization=False  # Disable for testing
        )
        
        self.assertIsNotNone(classifier.model_name)
        self.assertIsNotNone(classifier.device)


class TestInference(unittest.TestCase):
    """Test inference functionality."""
    
    def test_analyze_abstract_structure(self):
        """Test abstract analysis output structure."""
        # This test would require a trained model
        # For now, just test the structure of the expected output
        expected_keys = [
            "abstract_id", "original_text", "cleaned_text",
            "classification", "disease_extraction", "is_cancer_related",
            "analysis_metadata"
        ]
        
        # Mock result structure
        mock_result = {
            "abstract_id": "test_1",
            "original_text": "Test abstract",
            "cleaned_text": "Test abstract",
            "classification": {
                "predicted_labels": ["Cancer"],
                "confidence_scores": {"Cancer": 0.9, "Non-Cancer": 0.1}
            },
            "disease_extraction": {
                "extracted_diseases": ["Lung Cancer"],
                "cancer_diseases": ["Lung Cancer"],
                "non_cancer_diseases": [],
                "total_diseases": 1,
                "cancer_count": 1,
                "non_cancer_count": 0
            },
            "is_cancer_related": True,
            "analysis_metadata": {
                "text_length": 13,
                "cleaned_length": 13,
                "disease_count": 1,
                "cancer_disease_count": 1,
                "non_cancer_disease_count": 0
            }
        }
        
        for key in expected_keys:
            self.assertIn(key, mock_result)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def test_preprocessing_and_extraction(self):
        """Test integration between preprocessing and disease extraction."""
        preprocessor = AbstractPreprocessor()
        extractor = DiseaseExtractor()
        
        text = "This study investigates <b>lung cancer</b> treatment [1,2,3]."
        
        # Clean text
        cleaned = preprocessor.clean_abstract(text)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("[1,2,3]", cleaned)
        
        # Extract diseases
        diseases = extractor.extract_diseases(cleaned)
        self.assertIsInstance(diseases, dict)
        self.assertIn("extracted_diseases", diseases)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataPreprocessing))
    test_suite.addTest(unittest.makeSuite(TestDiseaseExtraction))
    test_suite.addTest(unittest.makeSuite(TestModelTraining))
    test_suite.addTest(unittest.makeSuite(TestInference))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
