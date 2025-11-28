# API Contracts (Backend)

## Overview
- **Base URL**: `/`
- **Framework**: FastAPI
- **Version**: 1.0.0 (Inferred)

## Endpoints

### General

#### `GET /`
- **Description**: Root endpoint providing API information.
- **Response**: JSON
  ```json
  {
    "message": "MCL AI Assistant API",
    "docs": "/docs",
    "version": "1.0.0"
  }
  ```

#### `GET /health`
- **Description**: Health check endpoint.
- **Response**: JSON
  ```json
  {
    "status": "healthy"
  }
  ```

### Knowledge Base

#### `GET /api/chunks`
- **Description**: List available knowledge chunks.
- **Query Parameters**:
  - `limit` (int, optional, default=10)
- **Response**: JSON (List of chunks)

#### `POST /api/search`
- **Description**: Search knowledge base chunks.
- **Request Body**: JSON
  ```json
  {
    "query": "string",
    "limit": 5
  }
  ```
- **Response**: JSON (Search results)

### Chat & Vision

#### `POST /api/chat`
- **Description**: Main chat endpoint supporting text and vision inputs.
- **Request Body**: `ChatRequest`
  ```json
  {
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "..." },
          { "type": "image_url", "image_url": { "url": "..." } }
        ]
      }
    ],
    "stream": false
  }
  ```
- **Response**: `ChatResponse`
  ```json
  {
    "id": "string",
    "choices": [...],
    "created": 1234567890,
    "model": "gpt-4o",
    "object": "chat.completion"
  }
  ```

#### `POST /api/vision/analyze-screenshot`
- **Description**: Analyze a screenshot for context.
- **Request Body**: JSON
  ```json
  {
    "image_base64": "string"
  }
  ```
- **Response**: JSON (Analysis result)

### Feedback

#### `POST /api/feedback`
- **Description**: Submit user feedback for a response.
- **Request Body**: `FeedbackRequest`
  ```json
  {
    "response_id": "string",
    "feedback_type": "positive|negative",
    "comment": "string"
  }
  ```
- **Response**: `FeedbackResponse`
  ```json
  {
    "status": "success",
    "message": "Feedback received"
  }
  ```
