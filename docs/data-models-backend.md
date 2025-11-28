# Data Models (Backend)

## Pydantic Models (API Layer)

### Chat Models
- **`ChatRequest`**
  - `messages`: List[`Message`]
  - `stream`: bool (default: False)

- **`Message`**
  - `role`: str
  - `content`: Union[str, List[`ContentItem`]]

- **`ContentItem`**
  - `type`: str ("text" or "image_url")
  - `text`: Optional[str]
  - `image_url`: Optional[dict]

- **`ChatResponse`**
  - `id`: str
  - `choices`: List[dict]
  - `created`: int
  - `model`: str
  - `object`: str

### Feedback Models
- **`FeedbackRequest`**
  - `response_id`: str
  - `feedback_type`: str ("positive" or "negative")
  - `comment`: Optional[str]

- **`FeedbackResponse`**
  - `status`: str
  - `message`: str

### Query Models
- **`MCLQuery`**
  - `query`: str
  - `context`: Optional[str]

## SQLAlchemy Models (Database Layer)

### `Feedback`
- **Table**: `feedback`
- **Columns**:
  - `id`: Integer (PK)
  - `response_id`: String(100) (Unique, Index)
  - `feedback_type`: String(20)
  - `user_comment`: Text (Nullable)
  - `created_at`: DateTime
  - `processed`: Boolean (default: False)

### `CuratedQA`
- **Table**: `curated_qa`
- **Columns**:
  - `id`: Integer (PK)
  - `question`: Text
  - `answer`: Text
  - `source_feedback_id`: Integer (Nullable)
  - `created_at`: DateTime
  - `active`: Boolean (default: True)
