import pytest, pytest_asyncio
from httpx import AsyncClient
from stream_rec.api.main import app

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_home_feed(client):
    r = await client.post("/v1/scenarios/home_feed",
                          json={"user_id": "u1", "ctx_timestamp": 0})
    assert r.status_code == 200
    assert r.json()["streams"], "no streams returned"
