"""Ad-hoc live test for the MCL user-data integration (no OpenAI required).

Usage (from the assistant_api directory, with the venv active):
    python test_checklists_tool.py <MCL_ACCESS_TOKEN>
or set the token via env:
    $env:MCL_TOKEN="..."; python test_checklists_tool.py

It verifies the two steps that the chat tool relies on:
  1. Resolve identity from the shared token  (GET /api/Account/UserInfo)
  2. Fetch the user's checklists             (GET /v8/CompanyCheckLists)
"""
import asyncio
import json
import os
import sys

# Allow running from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.clients.mcl_service_client import MCLServiceClient


async def main() -> None:
    token = (sys.argv[1] if len(sys.argv) > 1 else None) or os.environ.get("MCL_TOKEN")
    if not token:
        print("ERROR: provide the token as an argument or via the MCL_TOKEN env var.")
        sys.exit(1)

    client = MCLServiceClient()
    print(f"Base URL: {client.base_url}\n")

    print("[1/2] Resolving identity via /api/Account/UserInfo ...")
    info = await client.get_user_info(token)
    user_id = info.get("id", "")
    company_id = info.get("companyId", "")
    print(f"      fullName  : {info.get('fullName')}")
    print(f"      user_id   : {user_id}")
    print(f"      company_id: {company_id}")
    print(f"      company   : {info.get('companyName')}\n")

    print("[2/2] Fetching checklists via /v8/CompanyCheckLists ...")
    checklists = await client.get_user_checklists(token, company_id, user_id)
    print(f"      returned {len(checklists)} checklist(s).")
    print("      first few:")
    print(json.dumps(checklists[:5], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
