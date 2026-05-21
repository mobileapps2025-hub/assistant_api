from fastapi.testclient import TestClient

from app.main import app


def test_memory_save_no_messages_is_noop_success():
    client = TestClient(app)

    response = client.post(
        "/api/memory/save",
        json={"session_id": "empty-regression-session", "messages": []},
    )

    assert response.status_code == 200
    assert response.json() == {"saved": [], "updated": [], "deleted": []}
