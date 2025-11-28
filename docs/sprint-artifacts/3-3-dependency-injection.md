# Story 3.3: Dependency Injection & Main Refactor

## Status
**Completed**

## Description
Refactored the application to use dependency injection for services, improving modularity and testability. Replaced legacy service calls in `main.py` with a new `ChatService`.

## Changes
- Created `app/core/dependencies.py`:
  - Defines dependency providers for `VectorStoreService`, `VisionService`, `ImageValidatorService`, and `ChatService`.
  - Implements singleton pattern for `VectorStoreService`.
- Created `app/services/chat_service.py`:
  - Encapsulates chat logic (text and vision).
  - Implements RAG and prompt construction.
- Updated `app/main.py`:
  - Removed `app.legacy_services` imports.
  - Uses `ChatService` for the `/api/chat` endpoint.
  - Uses `VectorStoreService` for `/health`, `/api/chunks`, and `/api/search`.
  - Initializes `VectorStoreService` in `lifespan`.
- Updated `app/routers/vision.py`:
  - Uses centralized dependencies.
- Updated `tests/test_vision_endpoint.py`:
  - Adapted to new dependency injection structure.

## Verification
- **Tests**: All 20 tests passed (`test_vision_endpoint.py`, `test_vector_store.py`, `test_vision_service.py`, `test_image_validator.py`).
- **Manual Check**: Verified code structure and imports.

## Next Steps
- Proceed to Story 3.4: Cleanup Legacy Code.
