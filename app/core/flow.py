"""Human-readable flow trace for manual testing.

Prints each step of the request flow followed by a down-arrow connector, to stderr and
separate from the JSON logs, so you can watch which path the agent takes. Toggle with the
`FLOW_TRACE` env var (default on).
"""
import os
import sys

_ENABLED = os.getenv("FLOW_TRACE", "true").lower() == "true"


def flow(step: str) -> None:
    if _ENABLED:
        print(f"  {step}\n   ↓", file=sys.stderr, flush=True)
