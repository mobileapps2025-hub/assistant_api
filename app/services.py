import json
import requests
from io import BytesIO
import os
from app.config import client 
from app.models import EventsBetweenWeeksRequest
from app.utils import FUNCTION_TOOLS 
from app.clients.api_client import APIClient
from pydantic import BaseModel
from datetime import datetime

# Global API client instance - will be set by the middleware
_api_client: APIClient = None

def set_api_client_token(token: str):
    """Set the global API client with the provided token."""
    global _api_client
    _api_client = APIClient(token)

async def get_stores():
    return await _api_client.make_request("GET", "Store/GetUserStores")

async def get_events_between_weeks(store_id: str, starting_week: int, ending_week: int, year: int):
    json_body = {
        "StoreId": store_id,
        "StartingWeek": starting_week,
        "EndingWeek": ending_week,
        "Year": year
    }
    print(f"Requesting events between weeks with body: {json_body}")
    return await _api_client.make_request("POST", "Event/GetEventsBetweenWeeks", json_body=json_body)

async def get_unplanned_events_between_weeks(store_id: str, starting_week: int, ending_week: int, year: int):
    json_body = {
        "StoreId": store_id,
        "StartingWeek": starting_week,
        "EndingWeek": ending_week,
        "Year": year
    }
    print(f"Requesting unplanned events between weeks with body: {json_body}")
    return await _api_client.make_request("POST", "Event/GetUnplannedEventsBetweenWeeks", json_body=json_body)

async def get_week_events(store_id: str, week: int, year: int):
    query_params = {
        "idStore": store_id,
        "week": week,
        "year": year
    }
    print(f"Requesting week events with body: {query_params}")
    return await _api_client.make_request("GET", "Event/GetWeekEvents", query_params=query_params)

async def get_events_by_name(event_name):
    query_params = {"name": event_name}
    return await _api_client.make_request("GET", "Event/GetEventsByName", query_params=query_params)
        
async def get_event_details(event_id):
    query_params = {"id": event_id}
    return await _api_client.make_request("GET", "Event/GetEvent", query_params=query_params)

async def get_store_unplanned_events(store_id):
    query_params = {"idStore": store_id}
    return await _api_client.make_request("GET", "Event/GetStoreUnplannedEvents", query_params=query_params)

async def get_company_unplanned_events():
    return await _api_client.make_request("POST", "Event/GetCompanyUnplannedEvents")

async def get_store_sales_areas(store_id):
    query_params = {"storeId": store_id}
    return await _api_client.make_request("GET", "Store/GetStoreSalesAreas", query_params=query_params)

async def get_unplanned_sales_areas_on_week(store_id, year, week):
    query_params = {"storeId": store_id, "year": year, "week": week}
    return await _api_client.make_request("GET", "Store/GetUnplannedSalesAreasOnWeek", query_params=query_params)

# --- Knowledge Base Functions ---
def create_file(openai_client, file_path):
    print(f"Attempting to create OpenAI file from: {file_path}")
    if file_path.startswith("http://") or file_path.startswith("https://"):
        try:
            response = requests.get(file_path, timeout=30)
            response.raise_for_status()  

            file_content = BytesIO(response.content)
            file_name = os.path.basename(file_path) or "downloaded_knowledge_file.pdf"
            
            created_file = openai_client.files.create(
                file=(file_name, file_content),
                purpose="assistants"
            )

        except requests.RequestException as e:
            print(f"Error downloading file from URL {file_path}: {e}")
            raise
    else:
        if not os.path.exists(file_path):
            print(f"Local file {file_path} not found.")
            raise FileNotFoundError(f"Local file {file_path} not found.")
        with open(file_path, "rb") as file_content_stream:
            created_file = openai_client.files.create(
                file=file_content_stream,
                purpose="assistants"
            )
    print(f"File created successfully with ID: {created_file.id}")
    return created_file.id

def start_knowledge_base():
    print("Starting knowledge base initialization...")
    try:
        file_path = "spotplan_guide.md" 
        print(f"Creating file object for: {file_path}")
        
        with open(file_path, "rb") as file_content_stream:
            knowledge_file = client.files.create(
                file=file_content_stream,
                purpose="assistants"
            )
        
        print(f"File created successfully with ID: {knowledge_file.id}")

        print(f"Creating vector store named 'spotplan_knowledge_base'...")
        vector_store = client.vector_stores.create(
            name="spotplan_knowledge_base",
            file_ids=[knowledge_file.id]
        )
        
        if not vector_store.id:
            raise ValueError("Failed to create a vector store with a valid ID.")

        print(f"Knowledge base setup complete. Vector Store ID: {vector_store.id} is ready.")
        
        return vector_store.id

    except Exception as e:
        print(f"FATAL: An error occurred during knowledge base setup: {e}")
        # Ensure the function returns None on failure so the main app knows it failed.
        return None


def get_ai_response(messages_input):
    current_year = datetime.now().year
    current_week = datetime.now().isocalendar()[1]  # Get the current week number

    system_prompt = f"""You are "Spotplan Assistant," an expert AI partner for the Spotplan application. 
    Your goal is to help users by calling the available API function tools. Follow the workflow instructions in your function descriptions.

    - **The Golden Rule of Clarification:** If a user's data request is ambiguous, you MUST default to the `get_stores()` workflow to ask the user for clarification.
    - **Use Your Memory:** Before asking the user for information (like a `store_id`).
    - **If user does not specify the year, default to the current year that is {current_year}**
    - **If user does not specify a week, default to the current week of the year that is {current_week}.**
    """
    
    final_messages = [{"role": "system", "content": system_prompt}] + messages_input

    print(f"Sending messages to AI for function-calling: {final_messages}")
    
    response = client.chat.completions.create( 
        model="gpt-4o",
        messages=final_messages, 
        tools=FUNCTION_TOOLS,
        tool_choice="auto"
    )
    print(f"Received AI response object: {type(response)}") 
    return response

def get_ai_format(messages_input, request: BaseModel):
    final_messages = [messages_input]

    print(f"Sending messages to AI for function-calling: {final_messages}")

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=final_messages,
        response_format=request.model_dump()
    )
    print(f"Received AI response object: {type(response)}")
    return response
