"""Eval the in-house KB retriever against the spike question set.

    cd assistant_api && python spikes/kb_eval.py

For each question: retrieve top-k, print the source docs + whether a screenshot chunk came
back when one was expected. Topic->doc hit judged by keyword match on the document name.
Compare with spikes/ragie_eval_report.md (the Ragie baseline) before switching.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.retrieval import retrieve  # noqa: E402
from spikes.eval_questions import EVAL_QUESTIONS  # noqa: E402

TOPIC_DOC_KEYWORDS = {
    "checklists": ["checklist"],
    "navigation": ["dashboard", "entry"],
    "tasks": ["task"],
    "inspections": ["checklist", "faq", "user guide"],
    "terminology": ["faq", "user guide", "checklist"],
    "troubleshooting": ["faq", "user guide"],
    "roles": ["security", "checklist"],
    "reports": ["report", "data analysis"],
    "markets": ["market", "user guide"],
    "photos": ["photo"],
    "data_analysis": ["data analysis"],
    "notifications": ["notification"],
    "security": ["security"],
}


def main() -> None:
    hits = 0
    visual_expected = visual_got = 0
    for case in EVAL_QUESTIONS:
        chunks = retrieve(case["q"])
        docs = [c.document_name for c in chunks]
        keywords = TOPIC_DOC_KEYWORDS.get(case["topic"], [])
        hit = any(k in d.lower() for d in docs for k in keywords)
        hits += hit
        has_image = any(c.images for c in chunks)
        if case["expects_visual"]:
            visual_expected += 1
            visual_got += has_image
        mark = "ok " if hit else "MISS"
        img = " [img]" if has_image else ""
        print(f"{mark} {case['topic']:14} {case['q'][:52]:54} -> {sorted(set(docs))[:3]}{img}")

    print(f"\nTopic-doc hit: {hits}/{len(EVAL_QUESTIONS)}")
    print(f"Visual coverage (screenshot retrieved when expected): {visual_got}/{visual_expected}")


if __name__ == "__main__":
    main()
