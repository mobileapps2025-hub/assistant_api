"""Live test for the working MCL user tools (no OpenAI required).

Usage (from the assistant_api directory, with the venv active):
    python test_user_tools.py <MCL_ACCESS_TOKEN>

Exercises the exact client methods the chat tools dispatch to:
  get_user_info, get_markets_by_username, get_checklists_by_date,
  get_open_task_count.
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.clients.mcl_service_client import MCLServiceClient


async def main() -> None:
    token = (sys.argv[1] if len(sys.argv) > 1 else None) or os.environ.get("MCL_TOKEN")
    if not token:
        print("ERROR: provide the token as an argument or via MCL_TOKEN.")
        sys.exit(1)

    client = MCLServiceClient()

    print("[get_user_info]")
    info = await client.get_user_info(token)
    user_id = info.get("id", "")
    email = info.get("email", "")
    print(f"   name={info.get('fullName')!r} email={email!r} "
          f"company={info.get('companyName')!r} role={info.get('roleId')!r}\n")

    print("[get_open_task_count]")
    count = await client.get_open_task_count(token, user_id)
    print(f"   open tasks = {count}\n")

    print("[get_user_markets -> GetMarketsUser]")
    markets = await client.get_markets_by_username(token, email)
    print(f"   {len(markets)} market(s): "
          f"{[m.get('Name') or m.get('name') for m in markets]}\n")

    print("[get_user_checklists -> CheckListDate]")
    now = datetime.utcnow()
    date_from = (now - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
    date_to = (now + timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S")
    checklists = await client.get_checklists_by_date(token, user_id, date_from, date_to)
    print(f"   window {date_from} .. {date_to}")
    print(f"   {len(checklists)} checklist(s)")
    print(json.dumps(checklists[:5], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
