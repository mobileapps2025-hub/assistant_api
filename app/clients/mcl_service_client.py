import os
import httpx
from typing import Optional, Dict, Any, List
from app.core.logging import get_logger

logger = get_logger(__name__)

MCL_SERVICES_BASE_URL = os.getenv(
    "MCL_SERVICES_BASE_URL",
    "https://mcl-dev-services.azurewebsites.net"
)


class MCLServiceClient:
    """Client for the MCL CheckList Service API.

    Authentication is delegated to the MCL app: the user's bearer token is
    obtained there and shared with this app (there is no login here). Given
    that token, ``get_user_info`` resolves the user's identity — most
    importantly ``user_id`` and ``company_id``, which the data endpoints
    require as query parameters.
    """

    def __init__(self, base_url: str = MCL_SERVICES_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def _auth_headers(self, access_token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {access_token}"}

    async def _get(
        self,
        path: str,
        access_token: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Perform an authenticated GET against the MCL service."""
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                url, params=params, headers=self._auth_headers(access_token)
            )
            resp.raise_for_status()
            return resp.json()

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Resolve the user's identity from a shared MCL bearer token.

        Returns the raw ``/api/Account/UserInfo`` payload, which includes
        ``id`` (user id), ``companyId``, ``companyName``, ``fullName`` and
        ``email`` among other fields.
        """
        return await self._get("/api/Account/UserInfo", access_token)

    async def get_user_markets(
        self, access_token: str, company_id: str, user_id: str
    ) -> List[Dict[str, Any]]:
        return await self._get(
            "/v8/CompanyMarkets",
            access_token,
            params={"CompanyId": company_id, "UserId": user_id},
        )

    async def get_user_checklists(
        self, access_token: str, company_id: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Get the checklists available to the user for their company."""
        return await self._get(
            "/v8/CompanyCheckLists",
            access_token,
            params={"CompanyId": company_id, "UserId": user_id},
        )

    async def get_breaklist_markets(
        self, access_token: str, company_id: str, user_id: str
    ) -> List[Dict[str, Any]]:
        return await self._get(
            "/v8/api/BreakList/GetMarkets",
            access_token,
            params={"CompanyId": company_id, "UserId": user_id},
        )

    # --- Working endpoints (verified against mcl-dev-services) -------------
    # CompanyMarkets / CompanyCheckLists currently 500 upstream, so the tools
    # are backed by these equivalent, working endpoints for now.

    async def get_markets_by_username(
        self, access_token: str, username: str
    ) -> List[Dict[str, Any]]:
        """Markets assigned to the user, looked up by username/email.

        ``/v8/GetMarketsUser`` wraps the list in an envelope:
        ``{"data": [...], "message": null, "success": true}``.
        """
        result = await self._get(
            "/v8/GetMarketsUser",
            access_token,
            params={"username": username},
        )
        if isinstance(result, dict):
            return result.get("data") or []
        return result or []

    async def get_checklists_by_date(
        self, access_token: str, user_id: str, date_from: str, date_to: str
    ) -> List[Dict[str, Any]]:
        """Checklists for the user within a date range (CheckListViewModel[])."""
        return await self._get(
            "/v8/CheckListDate",
            access_token,
            params={"UserId": user_id, "From": date_from, "To": date_to},
        )

    async def get_open_task_count(
        self, access_token: str, user_id: str
    ) -> int:
        """Number of open tasks assigned to the user."""
        result = await self._get(
            "/v8/GetOpenTaskNumber",
            access_token,
            params={"userId": user_id},
        )
        try:
            return int(result)
        except (TypeError, ValueError):
            return 0
