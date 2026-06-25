"""End-to-end test against a running backend on localhost:8000.

Maintains conversation history across turns (like the real UI) to reproduce the
"second time it falls through to RAG" bug and confirm the routing fix.
"""
import asyncio
import sys
import httpx

API = "http://127.0.0.1:8000"


async def main():
    token = sys.argv[1]
    history = []
    async with httpx.AsyncClient(timeout=60) as client:
        s = await client.post(f"{API}/api/auth/session", json={"access_token": token})
        s.raise_for_status()
        auth = s.json()
        print(f"[session] {auth.get('full_name')} / {auth.get('company_name')}\n")

        async def ask(question):
            history.append({"role": "user", "content": question})
            resp = await client.post(
                f"{API}/api/chat",
                json={"messages": history, "session_id": "e2e", "auth_context": auth},
            )
            resp.raise_for_status()
            answer = resp.json().get("response")
            history.append({"role": "assistant", "content": answer})
            print(f"Q: {question}\nA: {answer}\n")

        # Personal — asked twice in the SAME session (the regression case).
        await ask("What markets are assigned to me?")
        await ask("What markets are assigned to me?")
        # General — should go to RAG, NOT a tool.
        await ask("How do I create a checklist in MCL?")
        # Personal again, after a general turn.
        await ask("How many open tasks do I have?")
        await ask("Who am I?")


if __name__ == "__main__":
    asyncio.run(main())
