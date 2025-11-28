import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
from app.services.vision_service import VisionService
from openai import OpenAIError

class TestVisionService(unittest.TestCase):

    def setUp(self):
        self.service = VisionService(api_key="test-key")

    def test_encode_image(self):
        """Test that _encode_image correctly encodes file content to base64."""
        mock_file_content = b"fake_image_data"
        expected_base64 = "ZmFrZV9pbWFnZV9kYXRh"  # base64 of "fake_image_data"
        
        with patch("builtins.open", mock_open(read_data=mock_file_content)):
            with patch("os.path.exists", return_value=True):
                result = self.service._encode_image("dummy.jpg")
                self.assertEqual(result, expected_base64)

    def test_encode_image_file_not_found(self):
        """Test that _encode_image raises FileNotFoundError if file doesn't exist."""
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.service._encode_image("nonexistent.jpg")

    @patch("app.services.vision_service.OpenAI")
    def test_analyze_image_success(self, mock_openai_class):
        """Test analyze_image with a successful API response."""
        # Setup mock client and response
        mock_client = MagicMock()
        self.service.client = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a test description."
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock file operations
        with patch.object(self.service, '_encode_image', return_value="base64string"):
            result = self.service.analyze_image("test.jpg", "Describe this")
            
            self.assertEqual(result, "This is a test description.")
            mock_client.chat.completions.create.assert_called_once()
            
            # Verify arguments passed to API
            call_args = mock_client.chat.completions.create.call_args
            self.assertEqual(call_args.kwargs['model'], "gpt-4o")
            self.assertEqual(call_args.kwargs['messages'][0]['role'], "user")
            self.assertEqual(call_args.kwargs['messages'][0]['content'][0]['text'], "Describe this")

    @patch("app.services.vision_service.OpenAI")
    def test_analyze_image_api_error(self, mock_openai_class):
        """Test analyze_image when OpenAI API raises an error."""
        mock_client = MagicMock()
        self.service.client = mock_client
        
        mock_client.chat.completions.create.side_effect = OpenAIError("API Error")
        
        with patch.object(self.service, '_encode_image', return_value="base64string"):
            with self.assertRaises(OpenAIError):
                self.service.analyze_image("test.jpg", "Describe this")

if __name__ == '__main__':
    unittest.main()
