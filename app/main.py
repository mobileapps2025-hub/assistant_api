import json
import uvicorn

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel # Added BaseModel import

from app.models import ChatRequest, Message, ContentItem
from app.services import (
    get_stores,
    get_week_events,
    get_events_between_weeks,
    get_events_by_name,
    get_event_details,
    get_unplanned_events_between_weeks,
    get_store_unplanned_events,
    get_company_unplanned_events,
    get_store_sales_areas,
    get_unplanned_sales_areas_on_week,
    start_knowledge_base,
    get_ai_response,
    set_api_client_token 
)
from app.knowledge_base import query_knowledge_base

AVAILABLE_FUNCTIONS = {
    "get_stores": get_stores,
    "get_week_events": get_week_events,
    "get_events_between_weeks": get_events_between_weeks,
    "get_events_by_name": get_events_by_name,
    "get_event_details": get_event_details,
    "get_store_unplanned_events": get_store_unplanned_events,
    "get_company_unplanned_events": get_company_unplanned_events,
    "get_store_sales_areas": get_store_sales_areas,
    "get_unplanned_sales_areas_on_week": get_unplanned_sales_areas_on_week,
    "get_unplanned_events_between_weeks": get_unplanned_events_between_weeks,
}
KNOWLEDGE_BASE_RETRIEVER = None # This global variable is declared but not used in the chat endpoint
VECTOR_STORE_ID = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global VECTOR_STORE_ID
    print("Application startup sequence initiated...")
    try:
        VECTOR_STORE_ID = start_knowledge_base()

        if not VECTOR_STORE_ID:
            print("CRITICAL: Failed to initialize the knowledge base. The vector store ID is missing.")
        else:
            print(f"Knowledge base loaded successfully. Vector Store ID: {VECTOR_STORE_ID}")
        print("Application startup sequence completed.")
        yield
    except Exception as e:
        print(f"FATAL ERROR during application startup: {e}")
        yield # Ensure yield is called even on exception for proper shutdown handling
    finally:
        print("Application will now shut down") # Corrected typo "hust down"

# --- FastAPI Application Initialization ---
app = FastAPI(lifespan=lifespan)
security = HTTPBearer()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for the request body of /api/chat
class ChatBody(BaseModel):
    messages_str: str

def function_needs_request(function_name: str) -> bool:
    functions = ["get_events_between_weeks"]
    return function_name in functions

async def get_spotplan_api_data(tool_call):
    function_name = tool_call.function.name
    function_to_call = AVAILABLE_FUNCTIONS.get(function_name)

    if not function_to_call:
        raise HTTPException(status_code=500, detail=f"Function '{function_name}' not found.")

    function_args = json.loads(tool_call.function.arguments)
    print(f"Function parameters: {function_args}")
    function_result = await function_to_call(**function_args)
    return function_result

def convert_messages_to_objects(messages_str: str) -> ChatRequest:
    try:
        request_data = json.loads(messages_str) # Corrected: was json.loads(chat_request)
        
        messages_list = []
        for msg_data in request_data.get("messages", []):
            content_input = msg_data.get("content")
            processed_content_list = []

            if isinstance(content_input, str):
                processed_content_list.append(ContentItem(text=content_input, type="text"))
            elif isinstance(content_input, list):
                for item in content_input:
                    if isinstance(item, dict) and "text" in item and "type" in item:
                        processed_content_list.append(ContentItem(text=item["text"], type=item["type"]))
            
            messages_list.append(Message(role=msg_data.get("role"), content=processed_content_list))
        
        chat_request_obj = ChatRequest(messages=messages_list)
        return chat_request_obj
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in messages_str: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field in messages_str content: {str(e)}")
    except Exception as e:
        # Log the exception for more detailed debugging if needed
        # import traceback; traceback.print_exc();
        raise HTTPException(status_code=400, detail=f"Error parsing messages_str: {str(e)}")

# --- API Endpoint ---
@app.post("/api/chat")
async def chat(
    body: ChatRequest, # Changed: Expect messages_str in the body
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    set_api_client_token(token)

    chat_request_obj = body

    fetching_data = True
    count = 0
    try:
        while fetching_data:
            count += 1
            if count > 5: # Max iterations to prevent infinite loops
                raise HTTPException(status_code=500, detail="Too many iterations, something went wrong.")
            
            # Convert Pydantic messages to dicts for the AI service
            messages_for_ai = [msg.model_dump(exclude_none=True) for msg in chat_request_obj.messages]
            print(f"Messages about to be send to AI: {messages_for_ai}")
            ai_response_obj = get_ai_response(messages_for_ai)
            print(f"AI response object: {ai_response_obj.model_dump()}")
            response_message_from_ai = ai_response_obj.choices[0].message

            # Convert AI response message to our Pydantic Message model
            ai_message_content_list = []
            if response_message_from_ai.content:
                ai_message_content_list.append(ContentItem(text=response_message_from_ai.content, type="text"))
            
            pydantic_ai_msg = Message(
                role=response_message_from_ai.role,
                content=ai_message_content_list
            )
            chat_request_obj.messages.append(response_message_from_ai)

            if response_message_from_ai.tool_calls:
                fetching_data = True # Continue loop if there are tool calls
                for tool_call in response_message_from_ai.tool_calls:
                    print(f"Response message before tool call: {response_message_from_ai}")
                    function_result = await get_spotplan_api_data(tool_call)

                    # Convert tool response to Pydantic Message model
                    tool_response_pydantic_msg = Message(
                        role="tool",
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name, # Storing function name for context
                        content=[ContentItem(text=json.dumps(function_result), type="text")]
                    )
                    chat_request_obj.messages.append(tool_response_pydantic_msg)
            else:
                fetching_data = False

        print(f"Final messages: {[msg.model_dump(exclude_none=True) for msg in chat_request_obj.messages]}")
        return {"messages": [msg.model_dump(exclude_none=True) for msg in chat_request_obj.messages]}

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)