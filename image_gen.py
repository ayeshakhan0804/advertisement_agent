"""Ad image generation via Pollinations. Cooldown + caching so you never see the 'Free limit reached' picture."""
import os
import time
import urllib.parse

import requests

COOLDOWN_SECONDS = 50  # anonymous tier allows ~1 image / 45 s


def generate_image(prompt: str, last_call: float = 0.0) -> bytes:
    key = os.getenv("POLLINATIONS_API_KEY")  # free key from enter.pollinations.ai removes the limit
    wait = COOLDOWN_SECONDS - (time.time() - last_call)
    if not key and wait > 0:
        raise RuntimeError(f"Free image limit: please wait {int(wait) + 1}s and click again.")
    url = "https://gen.pollinations.ai/image/" + urllib.parse.quote(prompt[:400])
    params = {"model": "flux", "width": 1024, "height": 1024}
    headers = {}
    if key:
        params["key"] = key
        headers["Authorization"] = f"Bearer {key}"
    r = requests.get(url, params=params, headers=headers, timeout=90)
    if r.status_code == 429:
        raise RuntimeError("Image service is rate-limiting you. Wait a minute or add POLLINATIONS_API_KEY.")
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
        raise RuntimeError(f"Image service returned status {r.status_code}.")
    return r.content