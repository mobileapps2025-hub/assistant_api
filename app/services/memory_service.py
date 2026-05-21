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


def _ensure_dir():
    os.makedirs(MEMORIES_DIR, exist_ok=True)


def _generate_id() -> str:
    return f"mem_{uuid.uuid4().hex[:8]}"


def _file_path(memory_id: str) -> str:
    return os.path.join(MEMORIES_DIR, f"{memory_id}.md")


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
            tags_str = ", ".join(val)
            lines.append(f"{key}: [{tags_str}]")
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

    def __init__(self):
        _ensure_dir()

    def list_memories(self) -> List[Dict[str, Any]]:
        _ensure_dir()
        results = []
        for fname in os.listdir(MEMORIES_DIR):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(MEMORIES_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                raw = f.read()
            meta = _parse_frontmatter(raw)
            body = _get_body(raw)
            results.append({
                "id": meta.get("id", fname[:-3]),
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "importance": meta.get("importance", "low"),
                "tags": meta.get("tags", []),
                "content": body,
                "created": meta.get("created", ""),
                "updated": meta.get("updated", ""),
            })
        results.sort(key=lambda m: m.get("created", ""), reverse=True)
        return results

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        fpath = _file_path(memory_id)
        if not os.path.exists(fpath):
            return None
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()
        meta = _parse_frontmatter(raw)
        body = _get_body(raw)
        return {
            "id": meta.get("id", memory_id),
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "importance": meta.get("importance", "low"),
            "tags": meta.get("tags", []),
            "content": body,
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
        fpath = _file_path(memory_id)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(_format_frontmatter(meta))
            f.write(content)
        return {**meta, "content": content}

    def delete_memory(self, memory_id: str) -> bool:
        fpath = _file_path(memory_id)
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
        memories = self.list_memories()
        if not memories:
            return ""
        lines = ["## User Context (from previous conversations)\n"]
        for m in memories:
            imp = m.get("importance", "low")
            prefix_map = {"high": "IMPORTANT:", "medium": "Noted:", "low": "Info:"}
            prefix = prefix_map.get(imp, "")
            title = m.get("title", "Memory")
            category = m.get("category", "general")
            tags_list = m.get("tags", [])
            tags_str = ", ".join(tags_list)
            lines.append(f"### {prefix} {title}")
            lines.append(f"_Category: {category} | Tags: {tags_str}_")
            lines.append("")
            lines.append(m.get("content", ""))
            lines.append("")
        return "\n".join(lines).strip()

    def store_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        pending_dir = os.path.join(MEMORIES_DIR, "pending")
        os.makedirs(pending_dir, exist_ok=True)
        fpath = os.path.join(pending_dir, f"{session_id}.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False)

    def get_stored_messages(self, session_id: str) -> List[Dict[str, Any]]:
        pending_dir = os.path.join(MEMORIES_DIR, "pending")
        fpath = os.path.join(pending_dir, f"{session_id}.json")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def clear_stored_messages(self, session_id: str):
        pending_dir = os.path.join(MEMORIES_DIR, "pending")
        fpath = os.path.join(pending_dir, f"{session_id}.json")
        if os.path.exists(fpath):
            os.remove(fpath)

    async def extract_memories(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        existing = self.list_memories()
        existing_json = json.dumps(existing, ensure_ascii=False, default=str)

        user_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
            if content:
                user_messages.append(f"[{role.upper()}] {content}")

        conversation_text = "\n".join(user_messages)
        if len(conversation_text) > 12000:
            conversation_text = conversation_text[-12000:]

        user_prompt = "Existing memories:\n" + existing_json + "\n\nConversation to review:\n" + conversation_text + "\n\nExtract important memories as JSON:"

        try:
            logger.info(f"[MEMORY] Extracting from {len(user_messages)} messages, {len(existing)} existing memories")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=60
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            count = len(data.get("memories", []))
            logger.info(f"[MEMORY] Extracted {count} memory actions")
            return data
        except Exception as e:
            logger.error(f"[MEMORY] Extraction failed: {e}")
            return {"memories": []}

    async def process_and_save(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        extracted = await self.extract_memories(messages)
        saved = []
        updated = []
        deleted = []

        for mem in extracted.get("memories", []):
            action = mem.get("action", "create")
            try:
                if action == "delete":
                    mem_id = mem.get("id")
                    if mem_id and self.delete_memory(mem_id):
                        deleted.append(mem_id)
                elif action == "update":
                    mem_id = mem.get("id")
                    if mem_id:
                        existing = self.get_memory(mem_id)
                        if existing:
                            existing["title"] = mem.get("title", existing.get("title", ""))
                            existing["category"] = mem.get("category", existing.get("category", ""))
                            existing["importance"] = mem.get("importance", existing.get("importance", "low"))
                            existing["tags"] = mem.get("tags", existing.get("tags", []))
                            existing["content"] = mem.get("content", existing.get("content", ""))
                            result = self.save_memory(existing)
                            updated.append(result)
                        else:
                            result = self.save_memory(mem)
                            saved.append(result)
                    else:
                        result = self.save_memory(mem)
                        saved.append(result)
                else:
                    result = self.save_memory(mem)
                    saved.append(result)
            except Exception as e:
                logger.error(f"[MEMORY] Error processing memory action '{action}': {e}")

        logger.info(f"[MEMORY] Processed: {len(saved)} created, {len(updated)} updated, {len(deleted)} deleted")
        return {"saved": saved, "updated": updated, "deleted": deleted}
