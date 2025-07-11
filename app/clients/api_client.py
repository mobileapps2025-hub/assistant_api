import httpx
import json
from typing import Optional, Dict, Any, Callable

class APIClient:
    """HTTP client with token middleware functionality for Spotplan API."""
    
    def __init__(self, token: str, base_url: str = "https://spotplanapi-hjhmgufjduhza6h0.westeurope-01.azurewebsites.net/api"):
        """
        Initialize the API client with authentication token.
        
        Args:
            token: JWT authentication token
            base_url: Base URL for the API endpoints
        """
        self.token = token
        self.base_url = base_url.rstrip('/')
    
    async def make_request(
        self,
        method: str,
        endpoint: str,
        json_body: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        custom_error_handler: Optional[Callable] = None
    ) -> Any:
        """
        Make an authenticated API request.
        
        Args:
            method: HTTP method ('GET' or 'POST')
            endpoint: API endpoint path (without base URL)
            json_body: JSON body for POST requests
            query_params: Query parameters for GET requests
            custom_error_handler: Optional custom error handler function
        
        Returns:
            API response data or error message
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {'Authorization': f'Bearer {self.token}'}
        
        # Add query parameters to URL if provided
        if query_params:
            query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
            url = f"{url}?{query_string}"
        
        print(f"Making {method} request to: {url}")
        if json_body:
            print(f"Request body: {json_body}")
        
        async with httpx.AsyncClient(verify=False) as http_client:
            try:
                # Make the HTTP request
                if method.upper() == "GET":
                    response = await http_client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await http_client.post(url, headers=headers, json=json_body)
                else:
                    return f"Error: Unsupported HTTP method {method}"
                
                print(f"Response: {response.status_code}, {response.text}")
                
                # Use custom error handler if provided
                if custom_error_handler:
                    return custom_error_handler(response)
                
                # Default error handling
                return self._handle_response(response)
                    
            except httpx.RequestError as e:
                return f"Request Error: {e}"
    
    def _handle_response(self, response: httpx.Response) -> Any:
        """
        Handle HTTP response with standard error codes.
        
        Args:
            response: The HTTP response object
            
        Returns:
            Parsed response data or error message
        """
        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text
        elif response.status_code == 401:
            return "Unauthorized: Session expired, please login again"
        elif response.status_code == 400:
            return f"Bad Request: {response.text}"
        elif response.status_code == 404:
            return f"Not Found: {response.text}"
        else:
            return f"Error: {response.status_code}, {response.text}"
