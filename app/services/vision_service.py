import base64
import os
from typing import Optional, Dict, Any
from openai import OpenAI, OpenAIError

class VisionService:
    """
    Service for analyzing images using OpenAI's GPT-4o vision capabilities.
    Designed to be stateless and modular.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the VisionService.
        
        Args:
            api_key: OpenAI API key. If not provided, defaults to environment variable.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"

    def _encode_image(self, image_path: str) -> str:
        """
        Encodes a local image file to a base64 string.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded string of the image.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image_base64(self, base64_image: str, prompt: str) -> str:
        """
        Analyzes a base64 encoded image using GPT-4o based on the provided prompt.
        
        Args:
            base64_image: Base64 encoded string of the image.
            prompt: The text prompt/question for the model.
            
        Returns:
            The text response from the model.
            
        Raises:
            OpenAIError: If the API call fails.
        """
        try:
            # Ensure the base64 string doesn't have the data prefix if passed raw, 
            # but the API expects the data URL format or just the base64?
            # The API expects a data URL or a URL. 
            # In analyze_image we construct: f"data:image/jpeg;base64,{base64_image}"
            # Let's assume input is raw base64 for consistency with _encode_image output,
            # or handle data URI prefix.
            
            if base64_image.startswith("data:"):
                image_url = base64_image
            else:
                # Default to jpeg if not specified, though GPT-4o is flexible.
                image_url = f"data:image/jpeg;base64,{base64_image}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )
            
            return response.choices[0].message.content
            
        except OpenAIError as e:
            raise e
        except Exception as e:
            raise e

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyzes an image using GPT-4o based on the provided prompt.
        
        Args:
            image_path: Path to the local image file.
            prompt: The text prompt/question for the model.
            
        Returns:
            The text response from the model.
            
        Raises:
            FileNotFoundError: If the image file is missing.
            OpenAIError: If the API call fails.
        """
        base64_image = self._encode_image(image_path)
        return self.analyze_image_base64(base64_image, prompt)
