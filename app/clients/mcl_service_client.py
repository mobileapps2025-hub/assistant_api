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
    def __init__(self, base_url: str = MCL_SERVICES_BASE_URL):
        self.base_url = base_url.rstrip("/")

    async def login(self, user_name: str, password: str) -> Dict[str, Any]:
        url = f"{self.base_url}/Token"
        payload = {
            "userName": user_name,
            "password": password,
            "grant_type": "password"
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, data=payload)
            resp.raise_for_status()
            token_data = resp.json()
            logger.info(f"Login response keys: {list(token_data.keys())}")

            access_token = token_data.get("access_token")
            if not access_token:
                raise ValueError("No access_token in login response")

            user_info = await self._get_user_info(access_token)
            return {
                "access_token": access_token,
                "user_id": user_info.get("id", ""),
                "company_id": user_info.get("companyId", ""),
                "company_name": user_info.get("companyName", ""),
                "full_name": user_info.get("fullName", ""),
                "email": user_info.get("email", ""),
                "role_id": user_info.get("roleId", ""),
            }

    async def _get_user_info(self, access_token: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/Account/UserInfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def get_user_markets(
        self, access_token: str, company_id: str, user_id: str
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/v8/CompanyMarkets"
        params = {"CompanyId": company_id, "UserId": user_id}
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def get_breaklist_markets(
        self, access_token: str, company_id: str, user_id: str
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/v8/api/BreakList/GetMarkets"
        params = {"CompanyId": company_id, "UserId": user_id}
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()
