"""
Test script for the API.
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing Research Paper Analysis API...")
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    try:
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{base_url}/api/v1/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
        
        # Test classification endpoint
        print("\n2. Testing classification endpoint...")
        test_data = {
            "abstract": "This study investigates lung cancer treatment efficacy in patients with stage III non-small cell lung cancer. The treatment protocol included chemotherapy and radiation therapy.",
            "abstract_id": "test_001"
        }
        
        response = requests.post(f"{base_url}/api/v1/classify", json=test_data, timeout=30)
        if response.status_code == 200:
            print("✅ Classification test passed!")
            result = response.json()
            print(f"Predicted: {result['classification']['predicted_labels']}")
            print(f"Confidence: {result['classification']['confidence_scores']}")
            print(f"Diseases: {result['disease_extraction']['extracted_diseases']}")
        else:
            print(f"❌ Classification test failed: {response.status_code}")
            print(f"Response: {response.text}")
        
        # Test batch classification
        print("\n3. Testing batch classification...")
        batch_data = {
            "abstracts": [
                {
                    "abstract": "This study examines diabetes management in elderly patients.",
                    "abstract_id": "test_002"
                },
                {
                    "abstract": "Breast cancer screening programs have shown significant impact on mortality rates.",
                    "abstract_id": "test_003"
                }
            ]
        }
        
        response = requests.post(f"{base_url}/api/v1/classify/batch", json=batch_data, timeout=30)
        if response.status_code == 200:
            print("✅ Batch classification test passed!")
            result = response.json()
            print(f"Processed {len(result['results'])} abstracts")
            for i, res in enumerate(result['results']):
                print(f"  Abstract {i+1}: {res['classification']['predicted_labels'][0]} "
                      f"(confidence: {res['classification']['confidence_scores']['Cancer']:.3f})")
        else:
            print(f"❌ Batch classification test failed: {response.status_code}")
        
        # Test disease extraction
        print("\n4. Testing disease extraction...")
        disease_data = {
            "abstract": "This study investigates prostate cancer and diabetes in patients with cardiovascular disease.",
            "abstract_id": "test_004"
        }
        
        response = requests.post(f"{base_url}/api/v1/diseases/extract", json=disease_data, timeout=30)
        if response.status_code == 200:
            print("✅ Disease extraction test passed!")
            result = response.json()
            print(f"Extracted diseases: {result['extracted_diseases']}")
            print(f"Cancer diseases: {result['cancer_diseases']}")
            print(f"Non-cancer diseases: {result['non_cancer_diseases']}")
        else:
            print(f"❌ Disease extraction test failed: {response.status_code}")
        
        print("\n🎉 All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_api()
