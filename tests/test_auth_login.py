"""Temporary username/password login (app/main.login)."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app import main
from app.models import LoginRequest


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_login_exchanges_credentials_for_session():
    with patch("app.main.MCLServiceClient") as MC:
        inst = MC.return_value
        inst.login = AsyncMock(return_value="tok123")
        inst.get_user_info = AsyncMock(return_value={
            "id": "u1", "companyId": "c1", "companyName": "Co", "fullName": "N", "email": "e@x.com",
        })
        res = _run(main.login(LoginRequest(user_name="a", password="b")))
    assert res.access_token == "tok123"
    assert res.user_id == "u1" and res.company_id == "c1"


def test_login_rejects_empty_token():
    with patch("app.main.MCLServiceClient") as MC:
        MC.return_value.login = AsyncMock(return_value="")
        with pytest.raises(Exception):
            _run(main.login(LoginRequest(user_name="a", password="bad")))
