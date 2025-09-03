# --- MCL Assistant Utility Functions ---

def format_mcl_response(response_text: str) -> str:
    """Format MCL assistant response for better readability."""
    # Add any specific formatting logic for MCL responses
    return response_text.strip()

def extract_mcl_topic(query: str) -> str:
    """Extract the main topic from a user query about MCL."""
    # Simple topic extraction - can be enhanced with NLP
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['checklist', 'check', 'list']):
        return 'checklist'
    elif any(word in query_lower for word in ['dashboard', 'web', 'browser']):
        return 'dashboard'
    elif any(word in query_lower for word in ['mobile', 'phone', 'app']):
        return 'mobile_app'
    elif any(word in query_lower for word in ['tablet', 'ipad']):
        return 'tablet'
    elif any(word in query_lower for word in ['role', 'permission', 'access']):
        return 'roles'
    elif any(word in query_lower for word in ['question', 'quiz', 'test']):
        return 'questions'
    elif any(word in query_lower for word in ['setup', 'install', 'config']):
        return 'setup'
    elif any(word in query_lower for word in ['error', 'problem', 'issue', 'bug']):
        return 'troubleshooting'
    elif any(word in query_lower for word in ['update', 'release', 'new']):
        return 'updates'
    else:
        return 'general'

# MCL topic categories for better organization
MCL_CATEGORIES = {
    'checklist': 'Creating and Managing Checklists',
    'dashboard': 'Web Dashboard Usage',
    'mobile_app': 'Mobile Application',
    'tablet': 'Tablet Application',
    'roles': 'User Roles and Permissions',
    'questions': 'Questions and Quiz Features',
    'setup': 'Setup and Configuration',
    'troubleshooting': 'Common Issues and Solutions',
    'updates': 'Updates and Release Notes',
    'general': 'General MCL Information'
}