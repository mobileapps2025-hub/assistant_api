# --- Function Tool Definitions (V3 - Corrected Based on Actual API) ---
FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stores",
            "description": "Retrieves a list of all stores the current user is authorized to access. This is the essential first step for any operation that is specific to a store. Call this when you need a 'store_id' for another function. Maps to: GET /api/Store/GetUserStores",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_event_details",
            "description": "Retrieves all detailed information for a single event by its unique ID. **Prerequisite:** This function requires a specific `event_id`. You can get an `event_id` from other functions like `get_event_by_name` or `get_events_between_weeks`. Maps to: GET /api/Event/GetEvent",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The unique identifier (GUID) of the event to retrieve details for."}
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events_by_name",
            "description": "Searches for events across all accessible stores that match a specific name. **Follow-up:** If this function returns multiple events, list them for the user to select from. If the user wants more details after selecting one, use the returned `event_id` to call `get_event_details`. Maps to: GET /api/Event/GetEventsByName",
            "parameters": {
                "type": "object",
                "properties": {"event_name": {"type": "string", "description": "The name of the event to search for."}},
                "required": ["event_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_store_unplanned_events",
            "description": "Fetches all unplanned events for a single, specific store. **Prerequisite:** This function requires a `store_id`. If the user has not specified a store, you must call `get_stores()` first and ask them to select one. Maps to: GET /api/Event/GetStoreUnplannedEvents",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "The unique identifier (GUID) of the store."}
                },
                "required": ["store_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_unplanned_events",
            "description": "Provides a high-level overview by fetching all unplanned events across every store in the user's company. Use this when the user asks for a company-wide summary of unscheduled items. Maps to: GET /api/Event/GetCompanyUnplannedEvents",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_week_events",
            "description": "Fetches all scheduled events for a specific store during a particular week of a specific year. **Prerequisites:** This function requires a `store_id`, `year`, and `week` number. If the `store_id` is missing, call `get_stores()` first and ask the user to select one. Maps to: GET /api/Event/GetWeekEvents",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "The unique identifier (GUID) of the store."},
                    "year": {"type": "integer", "description": "The year (e.g., 2024)."},
                    "week": {"type": "integer", "description": "The week number from 1 to 52."}
                },
                "required": ["store_id", "year", "week"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events_between_weeks",
            "description": "Fetches all scheduled events for a specific store within a given range of calendar weeks and a specific year. **Prerequisites:** This function requires a `store_id`, `starting_week`, `ending_week`, and `year`. If the `store_id` is missing, call `get_stores()` first and ask the user to select one. Maps to: POST /api/Event/GetEventsBetweenWeeks",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "The unique identifier (GUID) of the store."},
                    "starting_week": {"type": "integer", "description": "The starting week of the range as an integer from 1 to 52."},
                    "ending_week": {"type": "integer", "description": "The ending week of the range as an integer from 1 to 52."},
                    "year": {"type": "integer", "description": "The year (e.g., 2024)."}
                },
                "required": ["store_id", "starting_week", "ending_week", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unplanned_events_between_weeks",
            "description": "Fetches all unplanned events for a specific store within a given range of calendar weeks and a specific year. **Prerequisites:** This function requires a `store_id`, `starting_week`, `ending_week`, and `year`. If the `store_id` is missing, call `get_stores()` first and ask the user to select one. Maps to: POST /api/Event/GetUnplannedEventsBetweenWeeks",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "The unique identifier (GUID) of the store."},
                    "starting_week": {"type": "integer", "description": "The starting week of the range as an integer from 1 to 52."},
                    "ending_week": {"type": "integer", "description": "The ending week of the range as an integer from 1 to 52."},
                    "year": {"type": "integer", "description": "The year (e.g., 2024)."}
                },
                "required": ["store_id", "starting_week", "ending_week", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_store_sales_areas",
            "description": "Retrieves all sales areas for a single, specific store. This is a crucial intermediate step for many workflows. **Prerequisite:** This function requires a `store_id`. If the user has not specified a store, call `get_stores()` first and ask them to select one. Maps to: GET /api/SalesArea/GetStoreSalesAreas",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "The unique identifier (GUID) of the store for which to retrieve sales areas."}
                },
                "required": ["store_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unplanned_sales_areas_on_week",
            "description": "Retrieves all sales areas that have no planned events for a specific week in a specific year for a given store. This helps identify which sales areas are available for planning new events. **Prerequisites:** This function requires a `store_id`, `year`, and `week` number. Maps to: GET /api/SalesArea/GetUnplannedSalesAreasOnWeek",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "The unique identifier (GUID) of the store."},
                    "year": {"type": "integer", "description": "The year (e.g., 2024)."},
                    "week": {"type": "integer", "description": "The week number from 1 to 52."}
                },
                "required": ["store_id", "year", "week"],
            },
        },
    }
]