import unittest
from unittest.mock import MagicMock, patch, mock_open
import json
import os
from app.services.image_validator import ImageValidatorService
from app.services.vision_service import VisionService

class TestImageValidatorService(unittest.TestCase):

    def setUp(self):
        # Ensure clean slate
        if os.path.exists("test_cache.json"):
            os.remove("test_cache.json")
        self.mock_vision_service = MagicMock(spec=VisionService)
        self.validator = ImageValidatorService(self.mock_vision_service, cache_file="test_cache.json")

    def tearDown(self):
        if os.path.exists("test_cache.json"):
            os.remove("test_cache.json")

    def test_validate_image_cache_hit(self):
        """Test that cached results are returned without calling the API."""
        fake_hash = "fakehash123"
        self.validator.cache = {fake_hash: {"is_mcl": True, "confidence": 0.9}}
        
        with patch.object(self.validator, '_get_image_hash', return_value=fake_hash):
            result = self.validator.validate_image("base64data")
            
            self.assertTrue(result["is_mcl"])
            self.assertEqual(result["confidence"], 0.9)
            self.mock_vision_service.analyze_image_base64.assert_not_called()

    def test_validate_image_api_call_success(self):
        """Test full validation flow with a successful API response."""
        # Mock hash to avoid cache hit (cache is empty by default in setUp)
        with patch.object(self.validator, '_get_image_hash', return_value="newhash"):
            # Mock API response
            api_response = json.dumps({
                "is_mcl_app": True,
                "confidence": 0.95,
                "reasoning": "Looks like a checklist",
                "detected_mcl_elements": ["checkboxes"],
                "app_identified": "Task App"
            })
            self.mock_vision_service.analyze_image_base64.return_value = api_response
            
            # Mock file save for cache update
            with patch("builtins.open", mock_open()):
                result = self.validator.validate_image("base64data")
                
                self.assertTrue(result["is_mcl"])
                self.assertEqual(result["confidence"], 0.95)
                self.mock_vision_service.analyze_image_base64.assert_called_once()

    def test_validate_image_json_parse_error(self):
        """Test handling of malformed JSON from API."""
        with patch.object(self.validator, '_get_image_hash', return_value="newhash"):
            self.mock_vision_service.analyze_image_base64.return_value = "Not JSON"
            
            result = self.validator.validate_image("base64data")
            
            self.assertFalse(result["is_mcl"])
            self.assertEqual(result["reason"], "Could not parse validation response")

    def test_validate_image_api_exception(self):
        """Test handling of exceptions from VisionService."""
        with patch.object(self.validator, '_get_image_hash', return_value="newhash"):
            self.mock_vision_service.analyze_image_base64.side_effect = Exception("API Error")
            
            result = self.validator.validate_image("base64data")
            
            self.assertFalse(result["is_mcl"])
            self.assertIn("Validation error", result["reason"])

if __name__ == '__main__':
    unittest.main()
