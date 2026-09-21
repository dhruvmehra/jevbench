from __future__ import annotations

import asyncio

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    headers: dict,
    retries: int = 3,
    backoff: float = 1.0,
) -> dict:
    """POST JSON, retrying on 429/5xx with exponential backoff. Raises on other errors."""
    for attempt in range(retries + 1):
        r = await client.post(url, json=body, headers=headers, timeout=60)
        if r.status_code in RETRY_STATUSES and attempt < retries:
            await asyncio.sleep(backoff * (2**attempt))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")
