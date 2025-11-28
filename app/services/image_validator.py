import json
import hashlib
import os
from typing import Dict, Any, Optional, List
from app.services.vision_service import VisionService

class ImageValidatorService:
    """
    Service for validating if an image is a task management app screenshot.
    Uses VisionService for AI analysis and implements caching.
    """

    def __init__(self, vision_service: VisionService, cache_file: str = "mcl_image_cache.json"):
        """
        Initialize the ImageValidatorService.

        Args:
            vision_service: Instance of VisionService.
            cache_file: Path to the JSON cache file.
        """
        self.vision_service = vision_service
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
        self.mcl_characteristics = {
            "common_ui_elements": [
                "checklist items with checkboxes",
                "task lists with status indicators",
                "dashboard with statistics cards",
                "mobile and web app interface",
                "side navigation menu",
                "task completion indicators",
                "colored status cards (blue/red/green)",
                "numbered statistics (task counts)"
            ]
        }

    def _load_cache(self) -> Dict[str, Any]:
        """Load cached validation results."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load validation cache: {e}")
        return {}

    def _save_cache(self):
        """Save validation cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save validation cache: {e}")

    def _get_image_hash(self, image_data: str) -> str:
        """Generate a hash of the image data for caching."""
        # Strip header if present
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        # Hash first 1000 chars for speed
        sample = image_data[:1000]
        return hashlib.md5(sample.encode()).hexdigest()

    def validate_image(self, image_data: str, use_cache: bool = True, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Validate if an image is from a task management application.

        Args:
            image_data: Base64 encoded image string (with or without data URI prefix).
            use_cache: Whether to check and update the cache.
            confidence_threshold: Minimum confidence to consider valid.

        Returns:
            Dictionary with validation results.
        """
        image_hash = self._get_image_hash(image_data)
        
        if use_cache and image_hash in self.cache:
            return self.cache[image_hash]

        prompt = self._build_validation_prompt()
        
        try:
            response_text = self.vision_service.analyze_image_base64(image_data, prompt)
            result = self._parse_response(response_text, confidence_threshold)
            
            if use_cache:
                self.cache[image_hash] = result
                self._save_cache()
                
            return result
            
        except Exception as e:
            return {
                "is_mcl": False,
                "confidence": 0.0,
                "reason": f"Validation error: {str(e)}",
                "detected_elements": [],
                "identified_app": "Unknown",
                "suggestion": "Error validating image."
            }

    def _build_validation_prompt(self) -> str:
        return f"""You are an expert at identifying mobile and web application screenshots.

Your task: Determine if this image is a screenshot from a TASK MANAGEMENT or CHECKLIST application.

Common visual patterns:
- {', '.join(self.mcl_characteristics['common_ui_elements'])}

Instructions:
1. Analyze if this is a screenshot from ANY task management or productivity application.
2. Look for task lists, checklists, dashboards, or project management features.
3. If you see task-related UI, return TRUE.
4. If it's clearly NOT a task/productivity app, return FALSE.

Respond ONLY in JSON format:
{{
    "is_mcl_app": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "detected_mcl_elements": ["element1", "element2"],
    "app_identified": "app name or category"
}}
"""

    def _parse_response(self, response_text: str, threshold: float) -> Dict[str, Any]:
        """Parse the JSON response from the LLM."""
        try:
            # Clean markdown code blocks
            cleaned_text = response_text.strip()
            if "```json" in cleaned_text:
                cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_text:
                cleaned_text = cleaned_text.split("```")[1].split("```")[0].strip()

            data = json.loads(cleaned_text)
            
            is_mcl = data.get('is_mcl_app', False)
            confidence = float(data.get('confidence', 0.0))
            
            return {
                "is_mcl": is_mcl and confidence >= threshold,
                "confidence": confidence,
                "reason": data.get('reasoning', ''),
                "detected_elements": data.get('detected_mcl_elements', []),
                "identified_app": data.get('app_identified', 'Unknown'),
                "suggestion": self._generate_suggestion(is_mcl, confidence, data.get('app_identified', 'Unknown'), threshold)
            }
        except Exception:
            # Fallback if JSON parsing fails
            return {
                "is_mcl": False,
                "confidence": 0.0,
                "reason": "Could not parse validation response",
                "detected_elements": [],
                "identified_app": "Unknown",
                "suggestion": "I couldn't verify this image."
            }

    def _generate_suggestion(self, is_mcl: bool, confidence: float, app: str, threshold: float) -> str:
        if is_mcl and confidence >= threshold:
            return "✅ This looks like a task management app screenshot."
        return f"❌ I don't recognize this as a task management app (identified: {app})."
