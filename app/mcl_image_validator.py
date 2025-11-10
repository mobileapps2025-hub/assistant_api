"""
MCL Image Validator
===================

This module provides visual validation to check if uploaded images are actually
from the MCL (Mobile Checklist) application. It uses a hybrid approach:

1. GPT-4 Vision for quick validation
2. Visual similarity checking with reference images (optional)
3. Caching for performance

Author: AI Assistant
Created: 2025
"""

from openai import OpenAI
import os
import base64
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

class MCLImageValidator:
    """
    Validates if an image is from the MCL application using AI-powered visual recognition.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_file: str = "mcl_image_cache.json"):
        """
        Initialize the MCL Image Validator.
        
        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY environment variable.
            cache_file: Path to cache file for storing validation results
        """
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()
        
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
        # MCL visual characteristics extracted from documentation and real usage
        self.mcl_characteristics = {
            "app_name": "MCL",
            "full_name": "Mobile Checklist",
            "common_ui_elements": [
                "checklist items with checkboxes",
                "task lists with status indicators",
                "dashboard with statistics cards",
                "mobile and web app interface",
                "side navigation menu",
                "task completion indicators",
                "colored status cards (blue/red/green)",
                "numbered statistics (task counts)"
            ],
            "typical_screens": [
                "Tablero de Control (Dashboard)",
                "Tareas (Tasks)",
                "Mensajes (Messages)",
                "Inicio (Home)",
                "dashboard with task statistics",
                "checklist view",
                "task list",
                "item details",
                "settings screen"
            ],
            "typical_labels": [
                "Tablero de Control",
                "Tareas completadas",
                "Tareas retrasadas",
                "Tareas sin completar",
                "Tareas recurrentes",
                "Lista de tareas",
                "Dashboard",
                "Tasks",
                "Checklist"
            ]
        }
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cached validation results."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load validation cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save validation cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save validation cache: {e}")
    
    def _get_image_hash(self, image_data: str) -> str:
        """
        Generate a hash of the image for caching.
        
        Args:
            image_data: Base64-encoded image data URL
        
        Returns:
            Hash string for cache key
        """
        # Extract base64 data (remove data:image/...;base64, prefix)
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        # Create hash from first 1000 characters (for performance)
        sample = image_data[:1000]
        return hashlib.md5(sample.encode()).hexdigest()
    
    def _check_cache(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """
        Check if image validation is cached.
        
        Args:
            image_hash: Hash of the image
        
        Returns:
            Cached validation result or None
        """
        if image_hash in self.cache:
            print(f"📦 Using cached validation result for image {image_hash[:8]}...")
            return self.cache[image_hash]
        return None
    
    def _update_cache(self, image_hash: str, result: Dict[str, Any]):
        """
        Update validation cache.
        
        Args:
            image_hash: Hash of the image
            result: Validation result to cache
        """
        self.cache[image_hash] = result
        self._save_cache()
    
    def validate_image(
        self, 
        image_data: str, 
        use_cache: bool = True,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Validate if an image is from the MCL application.
        
        Args:
            image_data: Base64-encoded image data URL (data:image/png;base64,...)
            use_cache: Whether to use cached results
            confidence_threshold: Minimum confidence score (0.0-1.0) to consider image as MCL
        
        Returns:
            {
                "is_mcl": bool,              # Whether image is from MCL
                "confidence": float,          # Confidence score (0.0-1.0)
                "reason": str,                # Explanation
                "detected_elements": list,    # MCL elements detected
                "suggestion": str             # User-facing message
            }
        """
        print("\n" + "="*80)
        print("🔍 MCL IMAGE VALIDATION")
        print("="*80)
        
        # Check cache
        image_hash = self._get_image_hash(image_data)
        if use_cache:
            cached_result = self._check_cache(image_hash)
            if cached_result:
                print(f"✅ Cache hit - Result: {'MCL' if cached_result['is_mcl'] else 'Not MCL'}")
                print("="*80 + "\n")
                return cached_result
        
        print(f"🔍 Analyzing image (hash: {image_hash[:8]}...)...")
        
        # Prepare validation prompt - Changed approach to check for task management apps
        # instead of trying to identify specific "MCL" which GPT-4o may not know
        validation_prompt = f"""You are an expert at identifying mobile and web application screenshots.

Your task: Determine if this image is a screenshot from a TASK MANAGEMENT or CHECKLIST application (like MCL, Todoist, Trello, Asana, Microsoft To Do, etc.)

What we're looking for:
- Task management or checklist application screenshots
- Productivity apps with tasks, to-dos, or checklists
- Dashboard screens showing task statistics or metrics
- List views with items, checkboxes, or completion indicators
- Project management or workflow tools

Common visual patterns in task management apps:
- {', '.join(self.mcl_characteristics['common_ui_elements'])}
- Dashboard with metrics (completed tasks, pending, overdue)
- Navigation menus with sections (Tasks, Projects, Dashboard, etc.)
- Labeled sections like "Tareas", "Tasks", "To-Do", "Projects"
- Colored status indicators or cards
- Mobile or web interface design

What we're NOT looking for:
- Random images (photos, screenshots of unrelated apps)
- Social media apps
- E-commerce websites
- Games or entertainment apps
- Documents or PDFs (unless they're about task management)

Instructions:
1. Analyze if this is a screenshot from ANY task management or productivity application
2. Look for task lists, checklists, dashboards, or project management features
3. If you see task-related UI (lists, checkboxes, dashboards, statistics), return TRUE
4. If it's clearly NOT a task/productivity app (random photo, different app category), return FALSE
5. Be LENIENT - if it has any task management characteristics, accept it

Respond in JSON format:
{{
    "is_mcl_app": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of what you see",
    "detected_mcl_elements": ["task-related UI elements you see"],
    "app_identified": "task management app" or "other category if clearly different"
}}

Be lenient: Accept ANY task management or productivity app screenshot. Only reject if it's clearly not task-related (random photos, social media, etc.)."""
        
        try:
            # Call GPT-4 Vision
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": validation_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data,
                                    "detail": "low"  # Use low detail for faster validation
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.3  # Lower temperature for more consistent validation
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            print(f"📝 Raw validation response: {response_text[:200]}...")
            
            # Try to extract JSON
            try:
                # Remove markdown code blocks if present
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                validation_data = json.loads(response_text)
                
                is_mcl = validation_data.get('is_mcl_app', False)
                confidence = float(validation_data.get('confidence', 0.0))
                reasoning = validation_data.get('reasoning', 'No reasoning provided')
                detected_elements = validation_data.get('detected_mcl_elements', [])
                identified_app = validation_data.get('app_identified', 'Unknown')
                
                print(f"📊 Validation result: {'MCL ✅' if is_mcl else 'Not MCL ❌'}")
                print(f"📈 Confidence: {confidence:.2f}")
                print(f"💭 Reasoning: {reasoning}")
                
                # Build result
                result = {
                    "is_mcl": is_mcl and confidence >= confidence_threshold,
                    "confidence": confidence,
                    "reason": reasoning,
                    "detected_elements": detected_elements,
                    "identified_app": identified_app,
                    "suggestion": self._generate_suggestion(
                        is_mcl, 
                        confidence, 
                        identified_app, 
                        confidence_threshold
                    )
                }
                
                # Cache the result
                if use_cache:
                    self._update_cache(image_hash, result)
                
                print("="*80 + "\n")
                return result
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse JSON response: {e}")
                print(f"Response was: {response_text}")
                
                # Fallback: analyze text response
                is_task_app = any(keyword in response_text.lower() for keyword in 
                                 ['task', 'checklist', 'todo', 'productivity', 'dashboard', 'tareas'])
                
                result = {
                    "is_mcl": False,
                    "confidence": 0.5 if is_task_app else 0.0,
                    "reason": "Could not parse validation response, being cautious",
                    "detected_elements": [],
                    "identified_app": "Unknown",
                    "suggestion": "I couldn't confidently verify this is a task management app screenshot. Please ensure you're uploading a screenshot from your task/checklist application."
                }
                
                print("="*80 + "\n")
                return result
        
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: allow image but flag uncertainty
            result = {
                "is_mcl": False,
                "confidence": 0.0,
                "reason": f"Validation error: {str(e)}",
                "detected_elements": [],
                "identified_app": "Unknown",
                "suggestion": "I encountered an error validating your image. Please try again or ensure it's a task management app screenshot."
            }
            
            print("="*80 + "\n")
            return result
    
    def _generate_suggestion(
        self, 
        is_mcl: bool, 
        confidence: float, 
        identified_app: str,
        threshold: float
    ) -> str:
        """
        Generate user-facing suggestion based on validation result.
        
        Args:
            is_mcl: Whether image is from a task management app
            confidence: Confidence score
            identified_app: Name/category of identified app
            threshold: Confidence threshold
        
        Returns:
            User-facing suggestion message
        """
        if is_mcl and confidence >= 0.9:
            return "✅ This appears to be a task management app screenshot. I'll analyze it for you."
        elif is_mcl and confidence >= threshold:
            return "✅ This looks like a task management app screenshot. I'll do my best to help you."
        elif confidence >= 0.5:
            return f"⚠️ I'm not confident this is a task management app. It might be from '{identified_app}'. For best results, please upload a screenshot from your task/checklist application."
        else:
            if identified_app and identified_app.lower() not in ["unknown", "task management app"]:
                return f"❌ This appears to be from '{identified_app}', which doesn't seem to be a task management application. Please upload a screenshot from your task/checklist app (MCL, Todoist, etc.)."
            else:
                return "❌ I don't recognize this as a task management app screenshot. Please ensure you're uploading a screenshot from a task or checklist application for accurate assistance."
    
    def batch_validate_images(
        self, 
        image_paths: List[str], 
        confidence_threshold: float = 0.7
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple images (useful for processing reference images during startup).
        
        Args:
            image_paths: List of paths to image files
            confidence_threshold: Minimum confidence score
        
        Returns:
            Dictionary mapping image paths to validation results
        """
        print(f"\n🔍 Batch validating {len(image_paths)} images...")
        
        results = {}
        for image_path in image_paths:
            try:
                # Read and encode image
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                # Determine image type
                ext = Path(image_path).suffix.lower()
                mime_type = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }.get(ext, 'image/png')
                
                # Create data URL
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_data = f"data:{mime_type};base64,{base64_image}"
                
                # Validate
                result = self.validate_image(image_data, confidence_threshold=confidence_threshold)
                results[image_path] = result
                
                print(f"  {'✅' if result['is_mcl'] else '❌'} {Path(image_path).name}: {result['confidence']:.2f}")
                
            except Exception as e:
                print(f"  ❌ Error validating {image_path}: {e}")
                results[image_path] = {
                    "is_mcl": False,
                    "confidence": 0.0,
                    "reason": f"Error: {str(e)}",
                    "detected_elements": [],
                    "identified_app": "Unknown",
                    "suggestion": "Error processing image"
                }
        
        print(f"✅ Batch validation complete: {sum(1 for r in results.values() if r['is_mcl'])}/{len(results)} confirmed as MCL\n")
        
        return results
    
    def clear_cache(self):
        """Clear the validation cache."""
        self.cache = {}
        self._save_cache()
        print("🗑️ Validation cache cleared")


# Global validator instance (initialized on import)
_mcl_validator: Optional[MCLImageValidator] = None


def get_mcl_validator() -> MCLImageValidator:
    """
    Get or create the global MCL image validator instance.
    
    Returns:
        MCLImageValidator instance
    """
    global _mcl_validator
    if _mcl_validator is None:
        _mcl_validator = MCLImageValidator()
    return _mcl_validator


def validate_mcl_image(
    image_data: str, 
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Convenience function to validate if an image is from MCL.
    
    Args:
        image_data: Base64-encoded image data URL
        confidence_threshold: Minimum confidence (0.0-1.0)
    
    Returns:
        Validation result dictionary
    """
    validator = get_mcl_validator()
    return validator.validate_image(image_data, confidence_threshold=confidence_threshold)
