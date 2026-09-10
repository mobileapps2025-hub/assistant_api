import hmac

from fastapi import HTTPException, Request

from app.core.config import ASSISTANT_SERVICE_SECRET


def require_service_secret(request: Request) -> None:
    if not ASSISTANT_SERVICE_SECRET:
        raise HTTPException(503, "The platform surface is not configured on this server.")
    supplied = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, ASSISTANT_SERVICE_SECRET):
        raise HTTPException(401, "Invalid or missing service secret.")
