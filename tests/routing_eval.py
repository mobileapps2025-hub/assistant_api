"""Real-LLM routing accuracy + determinism eval (run on demand, not in CI).

    cd assistant_api && python tests/routing_eval.py

Hits the live model via classify_route. Each misroute we decide is wrong becomes a new
labeled case here. Determinism: temp 0 means the same input must return the same route.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.routing import classify_route


def U(text):
    return {"role": "user", "content": text}


def A(text):
    return {"role": "assistant", "content": text}


LABELED = [
    # CHAT
    {"messages": [U("Hi, how are you?")], "expected": "CHAT"},
    {"messages": [U("thanks!")], "expected": "CHAT"},
    {"messages": [U("what can you do?")], "expected": "CHAT"},
    {"messages": [U("are you a real person?")], "expected": "CHAT"},
    {"messages": [U("Hallo, wie geht's?")], "expected": "CHAT"},

    # KNOWLEDGE
    {"messages": [U("How do I create a checklist?")], "expected": "KNOWLEDGE"},
    {"messages": [U("What is the Checklist Wizard?")], "expected": "KNOWLEDGE"},
    {"messages": [U("What's the difference between a routine and a special inspection?")], "expected": "KNOWLEDGE"},
    {"messages": [U("How do I complete a task on a tablet?")], "expected": "KNOWLEDGE"},
    {"messages": [U("How do notifications work in the app?")], "expected": "KNOWLEDGE"},
    {"messages": [U("Why isn't the app syncing?")], "expected": "KNOWLEDGE"},
    {"messages": [U("Wie erstelle ich eine Checkliste?")], "expected": "KNOWLEDGE"},

    # PERSONAL
    {"messages": [U("How many open tasks do I have?")], "expected": "PERSONAL"},
    {"messages": [U("Who am I?")], "expected": "PERSONAL"},
    {"messages": [U("Which markets are assigned to me?")], "expected": "PERSONAL"},
    {"messages": [U("Show me my checklists")], "expected": "PERSONAL"},
    {"messages": [U("What's my role and company?")], "expected": "PERSONAL"},
    {"messages": [U("Wie viele offene Aufgaben habe ich?")], "expected": "PERSONAL"},

    # Follow-ups that need conversation context
    {"messages": [U("How do I create a checklist?"), A("Use the Checklist Wizard."), U("and how do I delete one?")],
     "expected": "KNOWLEDGE", "note": "follow-up, general"},
    {"messages": [U("How many tasks do I have?"), A("You have 5 open tasks."), U("and how many are overdue?")],
     "expected": "PERSONAL", "note": "follow-up, personal"},
    {"messages": [U("What is a market in MCL?"), A("A market is a store/location."), U("how many do I have?")],
     "expected": "PERSONAL", "note": "follow-up flips general->personal"},
]

DETERMINISM_SUBSET = [
    [U("How many open tasks do I have?")],
    [U("How do I create a checklist?")],
    [U("Hi, how are you?")],
    [U("Wie viele offene Aufgaben habe ich?")],
    [U("How do I create a checklist?"), A("Use the Checklist Wizard."), U("and how do I delete one?")],
]


def run_accuracy():
    print("=== Routing accuracy ===")
    passed = 0
    misroutes = []
    for case in LABELED:
        decision = classify_route(case["messages"])
        ok = decision.route == case["expected"]
        passed += ok
        latest = case["messages"][-1]["content"]
        note = f"  [{case['note']}]" if case.get("note") else ""
        mark = "ok " if ok else "MISS"
        print(f"  {mark}  {case['expected']:9} got {decision.route:9} | {latest[:48]}{note}")
        if not ok:
            misroutes.append((latest, case["expected"], decision.route, decision.reason))
    print(f"\nAccuracy: {passed}/{len(LABELED)} = {100*passed/len(LABELED):.0f}%")
    if misroutes:
        print("\nMisroutes:")
        for latest, exp, got, reason in misroutes:
            print(f"  - '{latest}' expected {exp}, got {got} — reason: {reason[:80]}")
    return passed, len(LABELED)


def run_determinism(repeats=3):
    print("\n=== Determinism (same input must give same route) ===")
    all_stable = True
    for messages in DETERMINISM_SUBSET:
        routes = [classify_route(messages).route for _ in range(repeats)]
        stable = len(set(routes)) == 1
        all_stable = all_stable and stable
        latest = messages[-1]["content"]
        print(f"  {'stable ' if stable else 'UNSTABLE'} {Counter(routes)} | {latest[:48]}")
    print("All stable." if all_stable else "DETERMINISM ISSUES above.")
    return all_stable


if __name__ == "__main__":
    run_accuracy()
    run_determinism()
