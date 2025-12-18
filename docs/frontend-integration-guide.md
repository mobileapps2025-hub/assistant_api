# Frontend Integration Guide: Self-Improving Agent Features

This guide outlines the changes required in the frontend application to support the new "Self-Improving" capabilities of the MCL Assistant.

## Overview

The goal is to allow users (or subject matter experts) to "teach" the agent when it makes a mistake. This is achieved by capturing the **correct answer** during the negative feedback flow and sending it to the backend to be used for training.

## Workflow: Negative Feedback & Correction

When a user indicates that an answer was incorrect (Thumbs Down), we want to capture the *correction* immediately.

### 1. UI Changes

*   **Current State**: User clicks "Thumbs Down" -> (Maybe) a generic comment box appears.
*   **New State**:
    1.  User clicks "Thumbs Down".
    2.  Display a form asking: **"What should have been the correct answer?"** or **"Please provide the missing information."**
    3.  This input field should be mandatory or strongly encouraged for negative feedback if we want to use it for training.

### 2. API Integration Sequence

When the user submits the correction form, the frontend should perform the following actions:

#### Step A: Submit Standard Feedback (Existing)
First, log the feedback event as usual.

*   **Endpoint**: `POST /api/feedback`
*   **Payload**:
    ```json
    {
      "response_id": "resp_12345",  // ID from the chat response
      "feedback_type": "negative",
      "user_comment": "The answer was outdated." // Optional generic comment
    }
    ```
*   **Response**:
    ```json
    {
      "id": 42,  // <--- Capture this ID
      "response_id": "resp_12345",
      ...
    }
    ```

#### Step B: Submit Correction for Training (New)
Immediately after Step A, send the correction to the curated knowledge base.

*   **Endpoint**: `POST /admin/curated-qa`
*   **Description**: Adds the corrected Q&A pair to the training dataset.
*   **Payload**:
    ```json
    {
      "question": "What is the secret codename?", // The original user query from chat history
      "answer": "The codename is Project Chimera.", // The USER'S CORRECTION from the text box
      "source_feedback_id": 42 // The ID returned from Step A (links the correction to the feedback)
    }
    ```
*   **Response**: `200 OK` with the created record.

---

## Admin Dashboard Features (Optional)

If you are building an Admin Panel for the MCL Assistant, you can include a button to trigger the training process manually.

### Trigger Training
This forces the agent to re-optimize its prompts using the newly collected data from the steps above.

*   **Endpoint**: `POST /admin/train`
*   **Method**: `POST`
*   **Payload**: `{}` (Empty JSON object)
*   **UI Behavior**:
    *   Show a loading spinner (Process takes 30-60 seconds).
    *   Display "Training Complete" upon success.
    *   **Note**: This does not need to be called after every feedback. It can be a daily or weekly batch job, or triggered manually by an admin when enough corrections have accumulated.

## Data Requirements

To successfully implement this, the Frontend must maintain context of:
1.  **The Original Question**: The text the user sent that triggered the response.
2.  **The Response ID**: The `id` field from the assistant's message (if available) or the transaction ID.

## Example Scenario

1.  **User**: "How do I reset the router?"
2.  **Agent**: "I don't know."
3.  **User**: Clicks **Thumbs Down**.
4.  **Frontend**: Shows input "What is the correct answer?".
5.  **User**: Types "Hold the reset button for 10 seconds." and clicks **Submit**.
6.  **Frontend**:
    *   Calls `POST /api/feedback` (logs the thumbs down).
    *   Gets `id: 101`.
    *   Calls `POST /admin/curated-qa` with:
        *   `question`: "How do I reset the router?"
        *   `answer`: "Hold the reset button for 10 seconds."
        *   `source_feedback_id`: 101
7.  **Result**: The system now has a "Golden Record" for this question. Next time `/admin/train` runs, the agent will learn this fact.
