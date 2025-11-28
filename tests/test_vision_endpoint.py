import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.core.dependencies import get_vision_service, get_image_validator_service, get_vector_store_service
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.services.vector_store import VectorStoreService

class TestVisionEndpoint(unittest.TestCase):

    def setUp(self):
        # Patch the lifespan context manager or the specific function it calls
        self.startup_patcher = patch("app.main.get_vector_store_service")
        self.mock_get_vector_store = self.startup_patcher.start()
        self.mock_vector_store = MagicMock(spec=VectorStoreService)
        self.mock_get_vector_store.return_value = self.mock_vector_store
        
        # Mock vector store methods called in lifespan
        self.mock_vector_store.index_exists.return_value = True
        self.mock_vector_store.load_index.return_value = True
        self.mock_vector_store.chunks = []

        self.client = TestClient(app)
        
        # Mock Services
        self.mock_vision_service = MagicMock(spec=VisionService)
        self.mock_image_validator = MagicMock(spec=ImageValidatorService)
        
        # Override dependencies
        app.dependency_overrides[get_vision_service] = lambda: self.mock_vision_service
        app.dependency_overrides[get_image_validator_service] = lambda: self.mock_image_validator

    def tearDown(self):
        self.startup_patcher.stop()
        app.dependency_overrides = {}

    def test_analyze_screenshot_success(self):
        """Test successful screenshot analysis."""
        # Setup mocks
        self.mock_image_validator.validate_image.return_value = {"is_mcl": True}
        self.mock_vision_service.analyze_image_base64.return_value = "Analysis Result"
        
        # Create dummy file
        files = {'file': ('test.jpg', b'fakeimagedata', 'image/jpeg')}
        data = {'query': 'What is this?'}
        
        response = self.client.post("/api/vision/analyze-screenshot", files=files, data=data)
        
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertTrue(json_response["success"])
        self.assertEqual(json_response["response"], "Analysis Result")
        
        # Verify calls
        self.mock_vision_service.analyze_image_base64.assert_called_once()

    def test_analyze_screenshot_invalid_file_type(self):
        """Test upload with invalid file type."""
        files = {'file': ('test.txt', b'text data', 'text/plain')}
        data = {'query': 'What is this?'}
        
        response = self.client.post("/api/vision/analyze-screenshot", files=files, data=data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid file type", response.json()["detail"])

    def test_analyze_screenshot_validation_failure(self):
        """Test when image validation fails."""
        # Setup mocks
        self.mock_image_validator.validate_image.return_value = {
            "is_mcl": False, 
            "suggestion": "Not an MCL image"
        }
        
        files = {'file': ('test.jpg', b'fakeimagedata', 'image/jpeg')}
        data = {'query': 'What is this?'}
        
        # We need to ensure ENABLE_MCL_IMAGE_VALIDATION is True for this test
        # Since it's imported in the router, we might need to patch it there
        with patch("app.routers.vision.ENABLE_MCL_IMAGE_VALIDATION", True):
            response = self.client.post("/api/vision/analyze-screenshot", files=files, data=data)
            
            self.assertEqual(response.status_code, 200)
            json_response = response.json()
            self.assertFalse(json_response["success"])
            self.assertEqual(json_response["response"], "Not an MCL image")

if __name__ == '__main__':
    unittest.main()
