"""Stage 4 tray — only web-surface gaps are surfaced to the local walker."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "codebase_agent"))
import web_gaps


def test_only_web_gaps_are_listed():
    payload = {"gaps": [
        {"id": 1, "surface": "web", "times_asked": 2, "role": "TeamMember", "question": "sensors?"},
        {"id": 2, "surface": "app", "times_asked": 1, "role": None, "question": "sync?"},
        {"id": 3, "surface": None, "times_asked": 1, "role": None, "question": "old gap"},
    ]}
    picked = web_gaps.web_gaps(payload)
    assert [g["id"] for g in picked] == [1]
