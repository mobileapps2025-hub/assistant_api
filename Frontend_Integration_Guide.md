# Unified Assistant API - Frontend Integration Guide

## Overview

The Unified Assistant API now supports both **Spotplan** and **MCL** applications in a single endpoint. This guide provides frontend developers with the information needed to integrate with both functionalities.

## Key Changes

### 1. Unified Endpoint

- **Single Endpoint**: `/api/chat` now handles both Spotplan and MCL requests
- **Automatic Detection**: The API can automatically detect which app is being used
- **Explicit App Type**: Frontend can explicitly specify the app type for better accuracy

### 2. Request Format Changes

#### Basic Request Structure (Unchanged)
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "Your question here",
          "type": "text"
        }
      ]
    }
  ]
}
```

#### New Optional Parameter
```json
{
  "messages": [...],
  "app_type": "spotplan" // or "mcl"
}
```

### 3. Response Format Changes

#### New Response Structure
```json
{
  "response": "The AI assistant's response text",
  "response_id": "resp_abc12345",
  "sources": ["document1.pdf", "document2.pdf"],
  "app_type": "mcl"
}
```

**Response Fields:**
- `response`: The main AI response text
- `response_id`: Unique identifier for feedback tracking (MCL only)
- `sources`: Array of source documents used (MCL only)
- `app_type`: Which app was used ("spotplan" or "mcl")

## Implementation Guide

### 1. Determining App Type

#### Option A: Explicit App Type (Recommended)
```javascript
const requestBody = {
  messages: chatMessages,
  app_type: "spotplan" // or "mcl"
};
```

#### Option B: Let API Auto-Detect
If you don't specify `app_type`, the API will attempt to detect it based on keywords in the user's message:

**MCL Keywords:** mcl, mobile checklist, checklist, quiz, question, dashboard, tablet
**Spotplan Keywords:** spotplan, store, event, sales area, week, planning, unplanned

**Note:** Auto-detection defaults to "spotplan" if uncertain, so explicit specification is recommended.

### 2. Frontend Implementation Examples

#### React/TypeScript Example
```typescript
interface ChatMessage {
  role: 'user' | 'assistant';
  content: Array<{text: string, type: string}>;
}

interface ChatRequest {
  messages: ChatMessage[];
  app_type?: 'spotplan' | 'mcl';
}

interface ChatResponse {
  response: string;
  response_id: string;
  sources?: string[];
  app_type: 'spotplan' | 'mcl';
}

const sendChatMessage = async (
  messages: ChatMessage[], 
  appType: 'spotplan' | 'mcl',
  authToken?: string  // Optional for MCL requests
): Promise<ChatResponse> => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };
  
  // Only add authentication for Spotplan or when token is provided
  if (authToken && (appType === 'spotplan' || authToken)) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }

  const response = await fetch('/api/chat', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      messages,
      app_type: appType
    })
  });
  
  return response.json();
};
```

#### JavaScript Example
```javascript
async function sendMessage(userMessage, appType, authToken = null) {
  const requestData = {
    messages: [
      {
        role: "user",
        content: [
          {
            text: userMessage,
            type: "text"
          }
        ]
      }
    ],
    app_type: appType // "spotplan" or "mcl"
  };

  const headers = {
    'Content-Type': 'application/json'
  };

  // Only add Authorization header for Spotplan or if authToken is provided
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }

  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(requestData)
  });

  const result = await response.json();
  return result;
}
```

### 3. Authentication Requirements

#### Spotplan Requests
- **Required**: Bearer token authentication
- **Purpose**: Token is used for API calls to Spotplan backend services
- **Behavior**: Without valid token, Spotplan-related API calls will fail

#### MCL Requests
- **Required**: No authentication required
- **Purpose**: MCL is purely document-based and doesn't require API access
- **Behavior**: MCL functionality works completely without authentication tokens

**Note**: While MCL doesn't require authentication, you can still provide a Bearer token for consistency if desired. The system will automatically handle both authenticated and non-authenticated MCL requests.

### 4. Error Handling

#### Common Error Responses
```json
{
  "detail": "Error message describing the issue"
}
```

#### Specific Error Cases

**MCL Knowledge Base Unavailable:**
```json
{
  "detail": "MCL knowledge base is not available. Please try again later."
}
```

**Spotplan API Issues:**
```json
{
  "detail": "Function 'get_stores' not found."
}
```

**Authentication Issues:**
```json
{
  "detail": "Invalid authentication credentials"
}
```

### 5. MCL-Specific Features

#### Feedback System
MCL responses include a `response_id` that can be used for user feedback:

```javascript
// Submit positive feedback
const submitFeedback = async (responseId, feedbackType, comment) => {
  await fetch('/api/feedback', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authToken}`
    },
    body: JSON.stringify({
      response_id: responseId,
      feedback_type: feedbackType, // "positive" or "negative"
      user_comment: comment
    })
  });
};
```

#### Document Search
MCL also provides document search capabilities:

```javascript
// Search MCL documents
const searchDocuments = async (query) => {
  const response = await fetch('/api/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      query: query,
      max_results: 5
    })
  });
  
  return response.json();
};
```

### 6. Migration from Separate Systems

#### If Currently Using Separate MCL and Spotplan APIs:

1. **Update Endpoint**: Change both frontend apps to use the same `/api/chat` endpoint
2. **Add App Type**: Include `app_type` parameter in requests
3. **Update Response Handling**: Adapt to new response format
4. **Test Both Flows**: Ensure both Spotplan and MCL functionality works correctly

#### Backward Compatibility Notes:
- The API maintains backward compatibility for existing Spotplan requests
- MCL-specific features (feedback, sources) are only available in the new response format

### 7. Testing Recommendations

#### Test Cases to Verify:

1. **Spotplan with explicit app_type:**
   ```json
   {"messages": [...], "app_type": "spotplan"}
   ```

2. **MCL with explicit app_type:**
   ```json
   {"messages": [...], "app_type": "mcl"}
   ```

3. **Auto-detection with MCL keywords:**
   ```json
   {"messages": [{"role": "user", "content": [{"text": "How do I use MCL dashboard?", "type": "text"}]}]}
   ```

4. **Auto-detection with Spotplan keywords:**
   ```json
   {"messages": [{"role": "user", "content": [{"text": "Show me store events", "type": "text"}]}]}
   ```

5. **Error handling for invalid tokens**

6. **MCL feedback submission**

### 8. Health Check and Monitoring

#### Health Check Endpoint: `GET /health`
```json
{
  "status": "healthy",
  "spotplan_knowledge_base_loaded": true,
  "mcl_knowledge_base_loaded": true,
  "spotplan_vector_store_id": "vs_abc123",
  "mcl_vector_store_id": "vs_def456",
  "total_mcl_document_chunks": 245
}
```

Use this endpoint to verify both knowledge bases are loaded correctly.

### 9. Additional Endpoints

#### MCL Document Information: `GET /api/chunks`
Returns information about available MCL document chunks.

#### Debug Endpoints:
- `GET /api/debug/response-cache` - Check response cache status
- `GET /api/debug/response-cache/{response_id}` - Get specific cached response

### 10. Performance Considerations

- **MCL responses** may be slightly slower due to document processing
- **Spotplan responses** depend on external API response times
- Consider implementing loading states for both app types
- MCL responses include source attribution which may make responses longer

## Summary

The unified API provides a seamless experience for both Spotplan and MCL functionality while maintaining the unique capabilities of each system. Frontend developers should:

1. Use explicit `app_type` specification for best results
2. Handle the new response format appropriately
3. Implement proper error handling for both app types
4. Consider implementing MCL-specific features like feedback when applicable

For any questions or issues, please refer to the API documentation or contact the development team.