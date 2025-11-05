"""
MCL App Screenshot Analysis Assistant
======================================

This module provides vision-enabled AI assistance for the MCL (Mobile Checklist) application.
It uses OpenAI's Assistants API with GPT-4o vision capabilities to analyze screenshots
and provide contextual help to users.

Author: AI Assistant
Created: 2025
"""

from openai import OpenAI
import os
import time
from typing import Optional, Dict, Any, List
from pathlib import Path


class MCLVisionAssistant:
    """
    AI Assistant for analyzing MCL App screenshots using OpenAI's vision capabilities.
    
    This class implements a complete workflow for:
    - Creating/retrieving vision-enabled assistants
    - Uploading images for analysis
    - Running conversations with multimodal inputs
    - Retrieving and formatting responses
    """
    
    # Default assistant configuration
    DEFAULT_ASSISTANT_NAME = "MCL App Vision Assistant"
    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_INSTRUCTIONS = """You are a specialist assistant for the 'MCL App'. Your primary role is to analyze screenshots provided by users. When a user uploads an image, your tasks are:

1. First, analyze the image to confirm if it is a screenshot from the 'MCL App'.
2. If it is not from the MCL App, politely inform the user you can only analyze MCL App images.
3. If it is from the MCL App, identify which screen, page, or feature is shown in the screenshot.
4. Read the user's text query to understand their question.
5. Based on the user's query and the visual context of the screenshot, provide clear, concise, and step-by-step instructions to help them.
6. If the user's query is vague (e.g., "what is this?"), describe the page's purpose and what they can do on it."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the MCL Vision Assistant.
        
        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY environment variable.
        """
        # Initialize OpenAI client (automatically reads from environment if api_key is None)
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()  # Reads from OPENAI_API_KEY env var
        
        self.assistant_id: Optional[str] = None
        self.thread_id: Optional[str] = None
    
    # ========================================================================
    # STEP 2: Get or Create the Assistant
    # ========================================================================
    
    def get_or_create_assistant(
        self,
        name: str = DEFAULT_ASSISTANT_NAME,
        instructions: str = DEFAULT_INSTRUCTIONS,
        model: str = DEFAULT_MODEL
    ) -> str:
        """
        Get an existing assistant by name or create a new one.
        
        This function first searches for an existing assistant matching the name.
        If found, it returns the existing assistant's ID. Otherwise, it creates
        a new assistant with the specified configuration.
        
        Args:
            name: The name of the assistant
            instructions: The system instructions for the assistant
            model: The OpenAI model to use (must be vision-capable, e.g., gpt-4o)
        
        Returns:
            The assistant ID (string)
        """
        print(f"🔍 Searching for assistant: '{name}'...")
        
        # List existing assistants
        try:
            assistants = self.client.beta.assistants.list(limit=100)
            
            # Check if assistant already exists
            for assistant in assistants.data:
                if assistant.name == name:
                    print(f"✅ Found existing assistant: {assistant.id}")
                    self.assistant_id = assistant.id
                    return assistant.id
            
            # Assistant not found, create new one
            print(f"📝 Creating new assistant: '{name}'...")
            assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                model=model,
                tools=[]  # No additional tools needed for vision analysis
            )
            
            print(f"✅ Assistant created successfully: {assistant.id}")
            self.assistant_id = assistant.id
            return assistant.id
            
        except Exception as e:
            print(f"❌ Error in get_or_create_assistant: {e}")
            raise
    
    # ========================================================================
    # STEP 3: Upload the Image for Vision
    # ========================================================================
    
    def upload_image_for_vision(self, image_path: str) -> str:
        """
        Upload an image file to OpenAI for vision analysis.
        
        The image must be uploaded with purpose="vision" to be used in
        vision-enabled conversations.
        
        Args:
            image_path: Path to the local image file (png, jpg, jpeg, etc.)
        
        Returns:
            The file ID (string) that can be used in messages
        
        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the file is not a valid image format
        """
        # Validate file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Validate file extension
        valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in valid_extensions:
            raise ValueError(f"Invalid image format: {file_ext}. Supported: {valid_extensions}")
        
        print(f"📤 Uploading image: {image_path}...")
        
        try:
            # Upload file with purpose="vision"
            with open(image_path, "rb") as file:
                uploaded_file = self.client.files.create(
                    file=file,
                    purpose="vision"
                )
            
            print(f"✅ Image uploaded successfully: {uploaded_file.id}")
            return uploaded_file.id
            
        except Exception as e:
            print(f"❌ Error uploading image: {e}")
            raise
    
    # ========================================================================
    # STEP 4: Create a Thread and Add the Multimodal Message
    # ========================================================================
    
    def create_thread_and_add_message(
        self,
        user_text_prompt: str,
        file_id: str
    ) -> str:
        """
        Create a new conversation thread and add a multimodal message.
        
        The message contains both text (user's question) and an image file reference.
        
        Args:
            user_text_prompt: The user's text query about the image
            file_id: The file ID of the uploaded image (from upload_image_for_vision)
        
        Returns:
            The thread ID (string) for the conversation
        """
        print(f"💬 Creating new thread...")
        
        try:
            # Create a new thread
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id
            print(f"✅ Thread created: {thread.id}")
            
            # Add multimodal message to the thread
            print(f"📝 Adding message with text and image...")
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": user_text_prompt
                    },
                    {
                        "type": "image_file",
                        "image_file": {
                            "file_id": file_id
                        }
                    }
                ]
            )
            
            print(f"✅ Message added to thread: {message.id}")
            return thread.id
            
        except Exception as e:
            print(f"❌ Error creating thread/message: {e}")
            raise
    
    # ========================================================================
    # STEP 5: Execute the Run and Poll for Completion
    # ========================================================================
    
    def run_assistant_and_wait(
        self,
        thread_id: str,
        assistant_id: str,
        poll_interval: float = 2.0,
        max_wait_time: float = 300.0
    ) -> Any:
        """
        Create a run and wait for it to complete by polling.
        
        This function creates a run (executes the assistant on the thread)
        and polls for completion. It handles various run statuses including
        queued, in_progress, completed, failed, and cancelled.
        
        Args:
            thread_id: The thread ID to run the assistant on
            assistant_id: The assistant ID to use
            poll_interval: Seconds to wait between status checks (default: 2.0)
            max_wait_time: Maximum seconds to wait (default: 300.0)
        
        Returns:
            The completed run object
        
        Raises:
            TimeoutError: If the run doesn't complete within max_wait_time
            RuntimeError: If the run fails or is cancelled
        """
        print(f"🚀 Starting assistant run...")
        
        try:
            # Create the run
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
            
            print(f"✅ Run created: {run.id}")
            start_time = time.time()
            
            # Poll for completion
            while True:
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    raise TimeoutError(f"Run exceeded maximum wait time of {max_wait_time}s")
                
                # Retrieve current run status
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                
                status = run.status
                print(f"⏳ Run status: {status} (elapsed: {elapsed_time:.1f}s)")
                
                # Check terminal states
                if status == "completed":
                    print(f"✅ Run completed successfully!")
                    return run
                
                elif status == "failed":
                    error_msg = f"Run failed. Error: {run.last_error}"
                    print(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)
                
                elif status == "cancelled":
                    print(f"❌ Run was cancelled")
                    raise RuntimeError("Run was cancelled")
                
                elif status == "expired":
                    print(f"❌ Run expired")
                    raise RuntimeError("Run expired")
                
                elif status in ["queued", "in_progress"]:
                    # Continue waiting
                    time.sleep(poll_interval)
                
                else:
                    # Unknown status
                    print(f"⚠️ Unknown run status: {status}")
                    time.sleep(poll_interval)
                    
        except Exception as e:
            print(f"❌ Error during run execution: {e}")
            raise
    
    # ========================================================================
    # STEP 6: Retrieve and Display the Response
    # ========================================================================
    
    def get_assistant_response(self, thread_id: str) -> str:
        """
        Retrieve the assistant's response from the thread.
        
        This function fetches all messages from the thread and extracts
        the latest response from the assistant.
        
        Args:
            thread_id: The thread ID to retrieve messages from
        
        Returns:
            The assistant's response text (string)
        
        Raises:
            ValueError: If no assistant response is found
        """
        print(f"📥 Retrieving assistant response...")
        
        try:
            # List messages in ascending order (oldest first)
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id,
                order="asc"
            )
            
            # Find the last assistant message
            assistant_message = None
            for message in reversed(messages.data):
                if message.role == "assistant":
                    assistant_message = message
                    break
            
            if not assistant_message:
                raise ValueError("No assistant response found in thread")
            
            # Extract text content from the message
            response_text = ""
            for content_block in assistant_message.content:
                if content_block.type == "text":
                    response_text += content_block.text.value
            
            if not response_text:
                raise ValueError("Assistant message contains no text content")
            
            print(f"✅ Response retrieved ({len(response_text)} characters)")
            return response_text
            
        except Exception as e:
            print(f"❌ Error retrieving response: {e}")
            raise
    
    # ========================================================================
    # Convenience Method: Complete Workflow
    # ========================================================================
    
    def analyze_screenshot(
        self,
        image_path: str,
        user_query: str,
        assistant_name: str = DEFAULT_ASSISTANT_NAME,
        instructions: str = DEFAULT_INSTRUCTIONS,
        model: str = DEFAULT_MODEL
    ) -> Dict[str, Any]:
        """
        Complete workflow to analyze an MCL App screenshot.
        
        This is a convenience method that combines all steps into one call:
        1. Get/create assistant
        2. Upload image
        3. Create thread with multimodal message
        4. Run assistant
        5. Get response
        
        Args:
            image_path: Path to the screenshot image
            user_query: User's question about the screenshot
            assistant_name: Name for the assistant (optional)
            instructions: System instructions (optional)
            model: OpenAI model to use (optional)
        
        Returns:
            Dictionary containing:
                - response: The assistant's response text
                - assistant_id: The assistant ID used
                - thread_id: The thread ID created
                - file_id: The uploaded file ID
                - image_path: The original image path
                - query: The user's query
        """
        print("\n" + "=" * 80)
        print("🔍 MCL APP SCREENSHOT ANALYSIS - COMPLETE WORKFLOW")
        print("=" * 80)
        
        try:
            # Step 1: Get or create assistant
            assistant_id = self.get_or_create_assistant(
                name=assistant_name,
                instructions=instructions,
                model=model
            )
            
            # Step 2: Upload image
            file_id = self.upload_image_for_vision(image_path)
            
            # Step 3: Create thread and add message
            thread_id = self.create_thread_and_add_message(
                user_text_prompt=user_query,
                file_id=file_id
            )
            
            # Step 4: Run assistant and wait
            run = self.run_assistant_and_wait(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
            
            # Step 5: Get response
            response_text = self.get_assistant_response(thread_id)
            
            print("=" * 80)
            print("✅ ANALYSIS COMPLETE")
            print("=" * 80)
            
            return {
                "response": response_text,
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "file_id": file_id,
                "image_path": image_path,
                "query": user_query,
                "success": True
            }
            
        except Exception as e:
            print("=" * 80)
            print(f"❌ ANALYSIS FAILED: {e}")
            print("=" * 80)
            
            return {
                "response": None,
                "error": str(e),
                "image_path": image_path,
                "query": user_query,
                "success": False
            }


# ============================================================================
# STEP 7: Main Execution Block (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of the MCL Vision Assistant.
    
    This block shows how to use the assistant in standalone mode.
    Replace IMAGE_TO_ANALYZE with an actual screenshot path.
    """
    
    print("\n🚀 MCL Vision Assistant - Standalone Demo")
    print("=" * 80)
    
    # Configuration
    ASSISTANT_NAME = "MCL App Vision Assistant"
    MODEL = "gpt-4o"
    INSTRUCTIONS = MCLVisionAssistant.DEFAULT_INSTRUCTIONS
    
    # User inputs (modify these for your use case)
    IMAGE_TO_ANALYZE = "./path/to/mcl-screenshot.png"  # ⚠️ UPDATE THIS PATH
    USER_QUERY = "How do I find my profile settings on this page?"
    
    # Check if image path needs to be updated
    if IMAGE_TO_ANALYZE == "./path/to/mcl-screenshot.png":
        print("\n⚠️ WARNING: Please update IMAGE_TO_ANALYZE with an actual screenshot path")
        print("Example usage:")
        print("  IMAGE_TO_ANALYZE = './screenshots/mcl-dashboard.png'")
        print("  USER_QUERY = 'What can I do on this screen?'")
        print("\nExiting demo...")
        exit(1)
    
    try:
        # Initialize the assistant
        assistant = MCLVisionAssistant()
        
        # Analyze the screenshot (complete workflow)
        result = assistant.analyze_screenshot(
            image_path=IMAGE_TO_ANALYZE,
            user_query=USER_QUERY,
            assistant_name=ASSISTANT_NAME,
            instructions=INSTRUCTIONS,
            model=MODEL
        )
        
        # Display results
        if result["success"]:
            print("\n" + "=" * 80)
            print("📋 ASSISTANT RESPONSE")
            print("=" * 80)
            print(result["response"])
            print("=" * 80)
            print(f"\n📊 Metadata:")
            print(f"  - Image: {result['image_path']}")
            print(f"  - Query: {result['query']}")
            print(f"  - Assistant ID: {result['assistant_id']}")
            print(f"  - Thread ID: {result['thread_id']}")
            print(f"  - File ID: {result['file_id']}")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
