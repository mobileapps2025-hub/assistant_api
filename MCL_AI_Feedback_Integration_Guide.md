# MCL AI Assistant - Feedback System Integration Guide

## Overview

The MCL AI Assistant now includes a comprehensive feedback and learning system that allows users to rate AI responses and helps improve the AI's performance over time. This guide provides all the necessary information for frontend developers to integrate these new features.

**⚠️ IMPORTANT UPDATE:** The system now properly stores the actual user questions and AI responses when feedback is submitted, enabling full learning capabilities.

## New API Endpoints

### 1. Enhanced Chat Endpoint

**Endpoint:** `POST /api/chat`

**Changes:** The chat endpoint now returns a tracking ID with each response for feedback purposes.

**Request Format:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "How do I use the MCL app?",
          "type": "text"
        }
      ]
    }
  ]
}
```

**Response Format:**
```json
{
  "response": "MCL is a mobile application for creating and managing checklists...",
  "response_id": "resp_a1b2c3d4",
  "sources": [
    "MCL How-to-use-Guide Tablet.pdf (Chunk 1/5)",
    "MCL How-to-use Phone.pdf (Chunk 2/3)"
  ]
}
```

**Key Changes:**
- `response_id`: Unique identifier for tracking this specific response
- `sources`: Array of document sources used to generate the response
- Response format changed from message array to single response object
- **CRITICAL:** The system now caches the full conversation data (question + response) in memory for feedback tracking

### 2. Submit Feedback

**Endpoint:** `POST /api/feedback`

**Purpose:** Allow users to rate AI responses and provide comments.

**IMPORTANT:** The system now validates that the response_id exists in the server cache before accepting feedback. This ensures data integrity and proper tracking of actual conversations.

**Request Format:**
```json
{
  "response_id": "resp_a1b2c3d4",
  "feedback_type": "positive",
  "user_comment": "Very helpful answer!"
}
```

**Request Fields:**
- `response_id` (required): The response ID from the chat endpoint
- `feedback_type` (required): Either "positive" or "negative"
- `user_comment` (optional): Additional user feedback text

**Response Format:**
```json
{
  "id": 1,
  "response_id": "resp_a1b2c3d4",
  "feedback_type": "positive",
  "user_comment": "Very helpful answer!",
  "created_at": "2025-09-18T10:30:00Z",
  "processed": false
}
```

**Error Responses:**
```json
// If response_id not found in cache
{
  "detail": "Response not found. Cannot submit feedback for this response.",
  "status_code": 404
}

// If feedback already submitted
{
  "detail": "Feedback already submitted for this response",
  "status_code": 400
}
```

### 3. Admin Endpoints (Optional Integration)

#### Get Feedback Statistics
**Endpoint:** `GET /api/admin/feedback/stats`

**Response:**
```json
{
  "total_feedback": 150,
  "positive_feedback": 120,
  "negative_feedback": 30,
  "unprocessed_feedback": 25,
  "satisfaction_rate": 80.0
}
```

#### Get Unprocessed Feedback
**Endpoint:** `GET /api/admin/feedback/unprocessed?limit=50`

**Response:**
```json
{
  "total": 25,
  "feedback": [
    {
      "id": 1,
      "response_id": "resp_a1b2c3d4",
      "feedback_type": "negative",
      "user_comment": "Answer was not helpful",
      "created_at": "2025-09-18T10:30:00Z",
      "user_question": "How do I create a new checklist in MCL?",
      "ai_response": "To create a new checklist in MCL, you need to..."
    }
  ]
}
```

#### Manage Curated Q&A
**Endpoint:** `GET /api/admin/curated-qa`
**Endpoint:** `POST /api/admin/curated-qa`
**Endpoint:** `DELETE /api/admin/curated-qa/{qa_id}`

#### Debug Endpoints (Development Only)
**Endpoint:** `GET /api/debug/response-cache`
**Purpose:** Check response cache status and size
**Response:**
```json
{
  "cache_size": 150,
  "recent_responses": ["resp_a1b2c3d4", "resp_e5f6g7h8"],
  "oldest_timestamp": "2025-09-18T10:00:00",
  "newest_timestamp": "2025-09-18T17:30:00"
}
```

**Endpoint:** `GET /api/debug/response-cache/{response_id}`
**Purpose:** Check specific response data in cache
**Response:**
```json
{
  "user_question": "How do I use MCL?",
  "ai_response": "MCL is a mobile application...",
  "retrieved_chunks": [],
  "timestamp": "2025-09-18T17:30:00"
}
```

## Frontend Implementation Guide

### 1. Update Chat Interface

#### Step 1: Modify Chat Service
Update your chat service to handle the new response format:

```javascript
// Before
async function sendMessage(messages) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages })
  });
  
  const data = await response.json();
  return data.messages; // Old format
}

// After
async function sendMessage(messages) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages })
  });
  
  const data = await response.json();
  return {
    message: data.response,
    responseId: data.response_id,
    sources: data.sources || []
  };
}
```

#### Step 2: Store Response IDs
Store the response ID with each AI message for feedback tracking:

```javascript
function addMessageToChat(message, responseId = null, sources = []) {
  const messageElement = {
    text: message,
    isAI: true,
    responseId: responseId,
    sources: sources,
    timestamp: new Date()
  };
  
  chatMessages.push(messageElement);
  renderMessage(messageElement);
}
```

### 2. Add Feedback UI Components

#### Basic Feedback Component (React Example)
```jsx
import React, { useState } from 'react';

const FeedbackComponent = ({ responseId, onFeedbackSubmitted }) => {
  const [feedbackGiven, setFeedbackGiven] = useState(false);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState('');

  const submitFeedback = async (feedbackType) => {
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          response_id: responseId,
          feedback_type: feedbackType,
          user_comment: comment || null
        })
      });

      if (response.ok) {
        setFeedbackGiven(true);
        onFeedbackSubmitted?.(feedbackType);
      } else {
        // Handle error responses
        const errorData = await response.json();
        console.error('Feedback submission failed:', errorData.detail);
        // Show user-friendly error message
        alert(errorData.detail || 'Failed to submit feedback. Please try again.');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Network error. Please check your connection and try again.');
    }
  };

  if (feedbackGiven) {
    return <div className="feedback-thanks">Thank you for your feedback!</div>;
  }

  return (
    <div className="feedback-container">
      <div className="feedback-buttons">
        <button 
          className="feedback-btn positive" 
          onClick={() => submitFeedback('positive')}
          title="This answer was helpful"
        >
          👍
        </button>
        <button 
          className="feedback-btn negative" 
          onClick={() => setShowComment(true)}
          title="This answer was not helpful"
        >
          👎
        </button>
      </div>
      
      {showComment && (
        <div className="feedback-comment">
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="How could this answer be improved?"
            rows="3"
          />
          <div className="comment-actions">
            <button onClick={() => submitFeedback('negative')}>
              Submit Feedback
            </button>
            <button onClick={() => setShowComment(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FeedbackComponent;
```

#### Vanilla JavaScript Example
```javascript
function createFeedbackUI(responseId) {
  const feedbackContainer = document.createElement('div');
  feedbackContainer.className = 'feedback-container';
  feedbackContainer.innerHTML = `
    <div class="feedback-buttons">
      <button class="feedback-btn positive" title="Helpful">👍</button>
      <button class="feedback-btn negative" title="Not helpful">👎</button>
    </div>
    <div class="feedback-comment" style="display: none;">
      <textarea placeholder="How could this answer be improved?" rows="3"></textarea>
      <div class="comment-actions">
        <button class="submit-feedback">Submit</button>
        <button class="cancel-feedback">Cancel</button>
      </div>
    </div>
  `;

  // Add event listeners
  const positiveBtn = feedbackContainer.querySelector('.positive');
  const negativeBtn = feedbackContainer.querySelector('.negative');
  const commentDiv = feedbackContainer.querySelector('.feedback-comment');
  const submitBtn = feedbackContainer.querySelector('.submit-feedback');
  const cancelBtn = feedbackContainer.querySelector('.cancel-feedback');
  const textarea = feedbackContainer.querySelector('textarea');

  positiveBtn.addEventListener('click', () => {
    submitFeedback(responseId, 'positive', null);
  });

  negativeBtn.addEventListener('click', () => {
    commentDiv.style.display = 'block';
  });

  submitBtn.addEventListener('click', () => {
    submitFeedback(responseId, 'negative', textarea.value);
  });

  cancelBtn.addEventListener('click', () => {
    commentDiv.style.display = 'none';
    textarea.value = '';
  });

  return feedbackContainer;
}

async function submitFeedback(responseId, feedbackType, comment) {
  try {
    const response = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        response_id: responseId,
        feedback_type: feedbackType,
        user_comment: comment
      })
    });

    if (response.ok) {
      // Replace feedback UI with thank you message
      const feedbackContainer = document.querySelector(`[data-response-id="${responseId}"]`);
      feedbackContainer.innerHTML = '<div class="feedback-thanks">Thank you for your feedback!</div>';
    } else {
      // Handle error responses
      const errorData = await response.json();
      console.error('Feedback submission failed:', errorData.detail);
      alert(errorData.detail || 'Failed to submit feedback. Please try again.');
    }
  } catch (error) {
    console.error('Error submitting feedback:', error);
    alert('Network error. Please check your connection and try again.');
  }
}
```

### 3. CSS Styling Example

```css
.feedback-container {
  margin-top: 8px;
  padding: 8px;
  border-top: 1px solid #e0e0e0;
}

.feedback-buttons {
  display: flex;
  gap: 8px;
  align-items: center;
}

.feedback-btn {
  background: none;
  border: 1px solid #ddd;
  border-radius: 20px;
  padding: 4px 8px;
  cursor: pointer;
  font-size: 16px;
  transition: all 0.2s ease;
}

.feedback-btn:hover {
  background-color: #f5f5f5;
  border-color: #ccc;
}

.feedback-btn.positive:hover {
  background-color: #e8f5e8;
  border-color: #4caf50;
}

.feedback-btn.negative:hover {
  background-color: #fce8e8;
  border-color: #f44336;
}

.feedback-comment {
  margin-top: 12px;
  padding: 12px;
  background-color: #f9f9f9;
  border-radius: 8px;
}

.feedback-comment textarea {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  resize: vertical;
  font-family: inherit;
}

.comment-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  justify-content: flex-end;
}

.comment-actions button {
  padding: 6px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
  background: white;
}

.comment-actions .submit-feedback {
  background-color: #2196f3;
  color: white;
  border-color: #2196f3;
}

.feedback-thanks {
  color: #4caf50;
  font-style: italic;
  font-size: 14px;
}
```

## Implementation Checklist

### Essential Features
- [ ] Update chat service to handle new response format
- [ ] Store response IDs with AI messages
- [ ] Add thumbs up/down buttons to AI responses
- [ ] Implement feedback submission functionality
- [ ] Add thank you message after feedback submission

### Optional Features
- [ ] Show comment input for negative feedback
- [ ] Display source documents used for each response
- [ ] Add admin dashboard for feedback statistics
- [ ] Implement feedback analytics

### Error Handling
- [ ] Handle network errors when submitting feedback
- [ ] Prevent duplicate feedback submission
- [ ] Show user-friendly error messages
- [ ] Graceful degradation if feedback service is unavailable

## Testing

### Test Scenarios

1. **Basic Feedback Flow**
   - Send a message to the chat API
   - Verify response contains `response_id`
   - Submit positive feedback
   - Verify feedback is stored correctly

2. **Negative Feedback with Comment**
   - Submit negative feedback with a comment
   - Verify comment is stored in the database

3. **Duplicate Feedback Prevention**
   - Try to submit feedback twice for the same response
   - Verify the second submission is rejected

4. **Error Handling**
   - Test with invalid response IDs (should return 404)
   - Test network failure scenarios
   - Test duplicate feedback submission (should return 400)
   - Test feedback submission for expired cache entries

### Sample Test Data

```javascript
// Test positive feedback
const testFeedback = {
  response_id: "resp_12345678",
  feedback_type: "positive",
  user_comment: "Very helpful explanation!"
};

// Test negative feedback
const testNegativeFeedback = {
  response_id: "resp_87654321",
  feedback_type: "negative",
  user_comment: "The answer was not specific enough."
};

// Test invalid response ID (should return 404)
const testInvalidFeedback = {
  response_id: "resp_invalid123",
  feedback_type: "positive",
  user_comment: "This should fail"
};
```

## Important Implementation Notes

### Response Cache Management

The system now uses an **in-memory cache** to store response data for feedback tracking:

- **Cache Size Limit:** 1000 responses (automatically cleaned)
- **Cache Persistence:** Data exists only while server is running
- **Cache Cleanup:** Oldest 100 entries removed when limit exceeded
- **Response Availability:** Responses are available for feedback immediately after generation

### Error Scenarios You Must Handle

1. **Response Not Found (404):** 
   - User tries to submit feedback for a response_id not in cache
   - Can happen if server was restarted or cache was cleaned
   - **Frontend Action:** Show user-friendly message, disable feedback buttons

2. **Duplicate Feedback (400):**
   - User tries to submit feedback twice for the same response
   - **Frontend Action:** Show "already submitted" message

3. **Network Errors:**
   - Connection issues, server down, etc.
   - **Frontend Action:** Show retry option, don't disable feedback permanently

## Database Schema Reference

For backend developers who need to understand the data structure:

### AIFeedback Table
- `id`: Primary key
- `response_id`: Unique response identifier
- `user_question`: The user's original question
- `ai_response`: The AI's response
- `feedback_type`: 'positive' or 'negative'
- `user_comment`: Optional user comment
- `retrieved_chunks`: JSON of document chunks used
- `created_at`: Timestamp
- `processed`: Boolean flag for admin review

### AICurated_Qa Table
- `id`: Primary key
- `question`: Curated question  
- `answer`: Curated answer
- `source_feedback_id`: Reference to original feedback
- `created_at`: Timestamp
- `active`: Boolean flag

## Next Steps

1. **Implement the basic feedback UI** with thumbs up/down buttons
2. **Test the integration** with the provided endpoints
3. **Add analytics** to track user satisfaction over time
4. **Consider adding** batch feedback operations for admin users
5. **Implement** feedback-driven content improvements

## Support

If you encounter any issues during integration:

1. Check the API response format matches the documentation
2. Verify database connectivity (check server logs)
3. Test endpoints individually using tools like Postman
4. Review network tab in browser developer tools

For technical questions, refer to the backend developer or system administrator.