# MCL Assistant API Documentation

## Overview

The MCL Assistant API is a FastAPI-based service that provides AI-powered assistance for the MCL (Mobile Checklist) application. It uses OpenAI's GPT-4o with a comprehensive knowledge base built from MCL documentation to answer user questions about the application.

## Base URL

```
http://localhost:8000
```

## Authentication

**No authentication required.** This is a purely informational service.

## API Endpoints

### 1. Root Endpoint

**GET /**

Returns basic API information and available endpoints.

**Response:**
```json
{
  "message": "MCL Assistant API",
  "version": "1.0.0",
  "description": "AI-powered assistant for the MCL (Mobile Checklist) application",
  "endpoints": {
    "chat": "/api/chat",
    "health": "/health"
  }
}
```

### 2. Health Check

**GET /health**

Returns the service health status and knowledge base availability.

**Response:**
```json
{
  "status": "healthy",
  "knowledge_base_loaded": true,
  "vector_store_id": "vs_68b097a4bcc88191a893e70488584c00"
}
```

**Response Fields:**
- `status`: Service status ("healthy" or "unhealthy")
- `knowledge_base_loaded`: Boolean indicating if the knowledge base is available
- `vector_store_id`: OpenAI vector store identifier (or "Not loaded" if unavailable)

### 3. Chat Endpoint (Main)

**POST /api/chat**

Main endpoint for interacting with the MCL Assistant.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "How do I create a checklist in MCL?",
          "type": "text"
        }
      ]
    }
  ]
}
```

**Request Schema:**
```typescript
interface ChatRequest {
  messages: Message[]
}

interface Message {
  role: "user" | "assistant" | "system"
  content: ContentItem[] | string | null
  tool_call_id?: string
  name?: string
  tool_calls?: any[]
  annotations?: string
}

interface ContentItem {
  text: string
  type: "text"
}
```

**Response:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "How do I create a checklist in MCL?",
          "type": "text"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "text": "To create a checklist in MCL, follow these steps:\n\n1. Open the MCL application\n2. Navigate to the 'Checklists' section\n3. Click on 'Create New Checklist'\n4. Enter the checklist name and description\n5. Add items to your checklist by clicking 'Add Item'\n6. Configure each item with appropriate settings\n7. Save your checklist\n\nFor more detailed instructions, refer to the MCL user guide in your dashboard.",
          "type": "text"
        }
      ]
    }
  ]
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid JSON format in messages_str: Expecting value: line 1 column 1 (char 0)"
}
```

### 500 Internal Server Error
```json
{
  "detail": "An unexpected error occurred: [error message]"
}
```

### 503 Service Unavailable
```json
{
  "detail": "MCL knowledge base is not available. Please try again later."
}
```

## Usage Examples

### Example 1: Basic Question

**Request:**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "text": "What are the different user roles in MCL?",
            "type": "text"
          }
        ]
      }
    ]
  }'
```

### Example 2: Follow-up Question

**Request:**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "text": "How do I use MCL on my phone?",
            "type": "text"
          }
        ]
      },
      {
        "role": "assistant", 
        "content": [
          {
            "text": "To use MCL on your phone, you need to download the MCL mobile app...",
            "type": "text"
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "text": "Can you give me more details about the mobile app features?",
            "type": "text"
          }
        ]
      }
    ]
  }'
```

### Example 3: JavaScript/TypeScript Frontend Implementation

```typescript
interface MCLAssistantAPI {
  sendMessage(messages: Message[]): Promise<ChatResponse>
  checkHealth(): Promise<HealthResponse>
}

class MCLAssistant implements MCLAssistantAPI {
  private baseUrl = 'http://localhost:8000'

  async sendMessage(messages: Message[]): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages })
    })

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }

    return await response.json()
  }

  async checkHealth(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`)
    return await response.json()
  }
}

// Usage example
const mclAssistant = new MCLAssistant()

// Send a question
const response = await mclAssistant.sendMessage([
  {
    role: 'user',
    content: [
      {
        text: 'How do I troubleshoot login issues in MCL?',
        type: 'text'
      }
    ]
  }
])

console.log(response.messages[response.messages.length - 1].content[0].text)
```

### Example 4: React Hook Implementation

```typescript
import { useState, useCallback } from 'react'

interface UseMLCAssistant {
  messages: Message[]
  sendMessage: (text: string) => Promise<void>
  isLoading: boolean
  error: string | null
}

export const useMCLAssistant = (): UseMLCAssistant => {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const sendMessage = useCallback(async (text: string) => {
    setIsLoading(true)
    setError(null)

    const userMessage: Message = {
      role: 'user',
      content: [{ text, type: 'text' }]
    }

    const newMessages = [...messages, userMessage]
    setMessages(newMessages)

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages })
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      setMessages(data.messages)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }, [messages])

  return { messages, sendMessage, isLoading, error }
}
```

## What the Assistant Can Help With

The MCL Assistant has comprehensive knowledge about:

### Core Features
- **Mobile App Usage**: How to use MCL on phones and mobile devices
- **Dashboard Operations**: Web dashboard functionality and navigation  
- **Tablet Interface**: Tablet-specific features and usage
- **Checklist Management**: Creating, managing, and using checklists
- **Questions and Quizzes**: Understanding the quiz and question features

### Administration
- **User Roles**: Role profiles and permission management
- **Setup and Configuration**: Installation and configuration guidance
- **Dropbox Integration**: Setup and important instructions

### Support
- **Troubleshooting**: Common issues and solutions
- **Release Updates**: Information about new features and updates
- **Best Practices**: Guidelines for effective MCL usage
- **Testing Procedures**: How to properly test MCL functionality

### Business Information
- **Business Cases**: Understanding MCL's business applications
- **Technical Updates**: Information about technical changes and improvements

## Sample Questions

Here are example questions you can ask the MCL Assistant:

- "How do I create a checklist in MCL?"
- "What are the different user roles in MCL?"
- "How do I use MCL on my mobile phone?"
- "What are common mistakes when using the MCL app?"
- "How do I set up Dropbox integration?"
- "What's new in the latest MCL release?"
- "How do I troubleshoot login issues?"
- "How do I create questions for a checklist?"
- "What are the tablet-specific features?"
- "How do I configure user permissions?"

## Rate Limits

No rate limits are currently implemented, but for production use consider:
- Implementing rate limiting on the client side
- Adding request timeouts (recommended: 30-60 seconds)
- Implementing retry logic for failed requests

## CORS

CORS is enabled for all origins (`*`) for development. For production, configure specific allowed origins.

## Error Handling Best Practices

1. **Always check response status** before processing the response
2. **Implement retry logic** for 5xx errors
3. **Handle network errors** gracefully
4. **Provide user-friendly error messages** instead of raw API errors
5. **Check health endpoint** before making chat requests if the service was down

## Development Notes

- The API runs on port 8000 by default
- The knowledge base is loaded at startup and may take a few seconds
- PDF processing happens automatically when the service starts
- The service uses OpenAI's GPT-4o model for responses
- All MCL documentation is processed and indexed for optimal search

## Production Considerations

When deploying to production:

1. **Environment Variables**: Ensure `OPENAI_API_KEY` is properly set
2. **CORS Configuration**: Restrict origins to your domain
3. **Rate Limiting**: Implement appropriate rate limits
4. **Logging**: Configure proper logging for monitoring
5. **Health Monitoring**: Use the `/health` endpoint for service monitoring
6. **SSL/HTTPS**: Ensure secure connections in production
7. **Error Handling**: Implement comprehensive error tracking
