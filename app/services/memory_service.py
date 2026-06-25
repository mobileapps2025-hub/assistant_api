import os
import re
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from app.core.config import client
from app.core.logging import get_logger

logger = get_logger(__name__)

MEMORIES_DIR = os.path.join(os.path.dirname(__file__), "..", "memories")
ANONYMOUS_SCOPE = "_anonymous"
RECALL_MAX_ITEMS = int(os.getenv("MEMORY_RECALL_MAX_ITEMS", "20"))
RECALL_MAX_CHARS = int(os.getenv("MEMORY_RECALL_MAX_CHARS", "4000"))
_IMPORTANCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _generate_id() -> str:
    return f"mem_{uuid.uuid4().hex[:8]}"


def _sanitize_scope(user_id: Optional[str]) -> str:
    if not user_id:
        return ANONYMOUS_SCOPE
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(user_id))[:64] or ANONYMOUS_SCOPE


def _parse_frontmatter(content: str) -> Dict[str, Any]:
    meta = {}
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm = content[3:end].strip()
            for line in fm.split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip()
                    if val.startswith("[") and val.endswith("]"):
                        inner = val[1:-1]
                        if inner.strip():
                            val = [t.strip().strip('"').strip("'") for t in inner.split(",")]
                            val = [t for t in val if t]
                        else:
                            val = []
                    meta[key] = val
    return meta


def _get_body(content: str) -> str:
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:].strip()
    return content.strip()


def _format_frontmatter(meta: Dict[str, Any]) -> str:
    lines = ["---"]
    for key in ["id", "title", "category", "importance", "tags", "created", "updated"]:
        val = meta.get(key, "")
        if isinstance(val, list):
            lines.append(f"{key}: [{', '.join(val)}]")
        elif val:
            lines.append(f"{key}: {val}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


EXTRACTION_PROMPT = """You are MarieClaire's memory extraction system. Review the conversation below and extract ONLY persistent, important information worth remembering for future interactions.

SAVE (create/update) -- information that is:
- User preferences (format style, language, detail level, how they want things)
- Ongoing projects or tasks the user is working on
- Decisions the user has made (tech choices, approaches)
- Personal context (role, department, workflow habits)
- Pain points, recurring issues, frustrations
- Facts the user explicitly asked you to remember
- Contradictions to previously stored facts (update the old memory)

DO NOT SAVE (skip):
- Small talk, greetings, pleasantries
- One-off general MCL questions that are answered by docs
- Generic information lookups
- Temporary debugging steps
- Anything a stranger wouldn't need to know

IMPORTANCE GUIDE:
- high: Core identity, critical preferences, active projects -- these directly change how you should respond
- medium: Useful recurring context, noted patterns, nice-to-know
- low: Minor preferences, single-mention context that might be useful

ACTION GUIDE:
- "create": New information not in existing memories
- "update": Existing memory needs to be modified (provide its id). Use when facts change or more detail emerges.
- "delete": Existing memory is now wrong or irrelevant. HIGH BAR -- only when the user explicitly contradicts it or the fact is clearly obsolete.

Output ONLY valid JSON with no markdown wrapper, no explanation:

{
  "memories": [
    {"action": "create", "title": "...", "category": "...", "importance": "high|medium|low", "tags": [...], "content": "..."},
    {"action": "update", "id": "mem_xxx", "title": "...", "category": "...", "importance": "...", "tags": [...], "content": "..."},
    {"action": "delete", "id": "mem_xxx"}
  ]
}"""


class MemoryService:
    """Durable memory, scoped to a single user. Without a user_id, durable memory is
    disabled (recall returns nothing) so one user never sees another's memories."""

    def __init__(self, user_id: Optional[str] = None):
        self.scoped = bool(user_id)
        self.dir = os.path.join(MEMORIES_DIR, _sanitize_scope(user_id))
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, memory_id: str) -> str:
        return os.path.join(self.dir, f"{memory_id}.md")

    def list_memories(self) -> List[Dict[str, Any]]:
        results = []
        for fname in os.listdir(self.dir):
            if not fname.endswith(".md"):
                continue
            with open(os.path.join(self.dir, fname), "r", encoding="utf-8") as f:
                raw = f.read()
            meta = _parse_frontmatter(raw)
            results.append({
                "id": meta.get("id", fname[:-3]),
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "importance": meta.get("importance", "low"),
                "tags": meta.get("tags", []),
                "content": _get_body(raw),
                "created": meta.get("created", ""),
                "updated": meta.get("updated", ""),
            })
        results.sort(key=lambda m: m.get("created", ""), reverse=True)
        return results

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        fpath = self._path(memory_id)
        if not os.path.exists(fpath):
            return None
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()
        meta = _parse_frontmatter(raw)
        return {
            "id": meta.get("id", memory_id),
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "importance": meta.get("importance", "low"),
            "tags": meta.get("tags", []),
            "content": _get_body(raw),
            "created": meta.get("created", ""),
            "updated": meta.get("updated", ""),
        }

    def save_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        memory_id = memory.get("id") or _generate_id()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        existing = self.get_memory(memory_id)
        created = memory.get("created") or (existing.get("created") if existing else now)
        meta = {
            "id": memory_id,
            "title": memory.get("title", ""),
            "category": memory.get("category", ""),
            "importance": memory.get("importance", "low"),
            "tags": memory.get("tags", []),
            "created": created,
            "updated": now,
        }
        content = memory.get("content", "")
        with open(self._path(memory_id), "w", encoding="utf-8") as f:
            f.write(_format_frontmatter(meta))
            f.write(content)
        return {**meta, "content": content}

    def delete_memory(self, memory_id: str) -> bool:
        fpath = self._path(memory_id)
        if os.path.exists(fpath):
            os.remove(fpath)
            return True
        return False

    def update_memory_content(self, memory_id: str, content: str) -> Optional[Dict[str, Any]]:
        existing = self.get_memory(memory_id)
        if not existing:
            return None
        existing["content"] = content
        return self.save_memory(existing)

    def recall_context(self) -> str:
        if not self.scoped:
            return ""
        memories = self.list_memories()
        if not memories:
            return ""
        memories.sort(key=lambda m: (_IMPORTANCE_RANK.get(m.get("importance", "low"), 2),))
        lines = ["## User Context (from previous conversations)\n"]
        used = 0
        for m in memories[:RECALL_MAX_ITEMS]:
            prefix = {"high": "IMPORTANT:", "medium": "Noted:", "low": "Info:"}.get(m.get("importance", "low"), "")
            block = (
                f"### {prefix} {m.get('title', 'Memory')}\n"
                f"_Category: {m.get('category', 'general')} | Tags: {', '.join(m.get('tags', []))}_\n\n"
                f"{m.get('content', '')}\n"
            )
            if used + len(block) > RECALL_MAX_CHARS:
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines).strip()

    def store_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        pending_dir = os.path.join(self.dir, "pending")
        os.makedirs(pending_dir, exist_ok=True)
        with open(os.path.join(pending_dir, f"{session_id}.json"), "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False)

    def get_stored_messages(self, session_id: str) -> List[Dict[str, Any]]:
        fpath = os.path.join(self.dir, "pending", f"{session_id}.json")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def clear_stored_messages(self, session_id: str):
        fpath = os.path.join(self.dir, "pending", f"{session_id}.json")
        if os.path.exists(fpath):
            os.remove(fpath)

    async def extract_memories(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        existing = self.list_memories()
        existing_json = json.dumps(existing, ensure_ascii=False, default=str)

        rendered = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
            if content:
                rendered.append(f"[{msg.get('role', '').upper()}] {content}")

        conversation_text = "\n".join(rendered)
        if len(conversation_text) > 12000:
            conversation_text = conversation_text[-12000:]

        user_prompt = (
            "Existing memories:\n" + existing_json
            + "\n\nConversation to review:\n" + conversation_text
            + "\n\nExtract important memories as JSON:"
        )

        try:
            logger.info(f"[MEMORY] Extracting from {len(rendered)} messages, {len(existing)} existing")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=60,
            )
            data = json.loads(response.choices[0].message.content.strip())
            logger.info(f"[MEMORY] Extracted {len(data.get('memories', []))} actions")
            return data
        except Exception as e:
            logger.error(f"[MEMORY] Extraction failed: {e}")
            return {"memories": []}

    async def process_and_save(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        extracted = await self.extract_memories(messages)
        saved, updated, deleted = [], [], []

        for mem in extracted.get("memories", []):
            action = mem.get("action", "create")
            try:
                if action == "delete":
                    mem_id = mem.get("id")
                    if mem_id and self.delete_memory(mem_id):
                        deleted.append(mem_id)
                elif action == "update" and mem.get("id") and self.get_memory(mem.get("id")):
                    existing = self.get_memory(mem["id"])
                    for field in ("title", "category", "importance", "tags", "content"):
                        existing[field] = mem.get(field, existing.get(field))
                    updated.append(self.save_memory(existing))
                else:
                    saved.append(self.save_memory(mem))
            except Exception as e:
                logger.error(f"[MEMORY] Error processing action '{action}': {e}")

        logger.info(f"[MEMORY] Processed: {len(saved)} created, {len(updated)} updated, {len(deleted)} deleted")
        return {"saved": saved, "updated": updated, "deleted": deleted}
