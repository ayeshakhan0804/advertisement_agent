"""Tools the advertisement agent can call. Plain functions first, wrapped as LangChain tools at the bottom."""
import json
import os
import re
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "campaigns.db")

PLATFORM_SPECS = {
    "google_ads": {"headline_max": 30, "body_max": 90, "cta_max": 20,
                   "tips": ["Lead with the keyword", "Include one concrete benefit or number", "No emojis"]},
    "facebook": {"headline_max": 40, "body_max": 125, "cta_max": 25,
                 "tips": ["Hook in the first line", "Speak to one person", "Keep it conversational"]},
    "instagram": {"headline_max": 60, "body_max": 300, "cta_max": 25,
                  "tips": ["Emotional hook first", "3-5 relevant hashtags", "Visual-first, short caption"]},
    "twitter": {"headline_max": 60, "body_max": 220, "cta_max": 25,
                "tips": ["Punchy and witty", "1-2 hashtags max", "Total post must be under 280 chars"]},
    "linkedin": {"headline_max": 70, "body_max": 400, "cta_max": 25,
                 "tips": ["Professional tone", "Lead with a business outcome", "Use a stat if available"]},
}

PLATFORM_ALIASES = {
    "google": "google_ads", "google_ad": "google_ads", "googleads": "google_ads",
    "ig": "instagram", "insta": "instagram",
    "fb": "facebook",
    "x": "twitter", "tweet": "twitter"
}

BANNED_PHRASES = ["guaranteed", "100% free", "miracle", "no risk", "get rich", "cure", "best in the world"]


def web_search(query: str) -> str:
    """Search the web for trends, competitor ads, or audience insights. Input: a short search query."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    results = DDGS().text(query, max_results=5)
    if not results:
        raise RuntimeError("search returned no results - try a different, shorter query")
    return "\n".join(f"- {r.get('title')}: {r.get('body')}" for r in results)


def get_platform_specs(platform: str) -> str:
    """Get character limits and best practices for an ad platform.
    Valid platforms: google_ads, facebook, instagram, twitter, linkedin."""
    key = platform.lower().strip().replace(" ", "_")
    key = PLATFORM_ALIASES.get(key, key)
    if key not in PLATFORM_SPECS:
        raise ValueError(f"Unknown platform '{platform}'. Valid options: {', '.join(PLATFORM_SPECS)}")
    return json.dumps({"platform": key, **PLATFORM_SPECS[key]})


def estimate_campaign_budget(budget: float, cost_per_click: float,
                             conversion_rate_pct: float, price_per_sale: float) -> str:
    """Estimate clicks, sales, revenue and ROAS for a campaign.
    conversion_rate_pct is a percentage (e.g. 2 for 2%)."""
    budget = float(budget)
    cost_per_click = float(cost_per_click)
    conversion_rate_pct = float(conversion_rate_pct)
    price_per_sale = float(price_per_sale)
    if budget <= 0 or cost_per_click <= 0 or price_per_sale <= 0:
        raise ValueError("budget, cost_per_click and price_per_sale must all be greater than 0")
    if not 0 < conversion_rate_pct <= 100:
        raise ValueError("conversion_rate_pct must be between 0 and 100")
    clicks = budget / cost_per_click
    sales = clicks * conversion_rate_pct / 100
    revenue = sales * price_per_sale
    return json.dumps({"clicks": round(clicks), "sales": round(sales, 1),
                       "revenue": round(revenue, 2), "roas": round(revenue / budget, 2),
                       "cost_per_sale": round(budget / sales, 2) if sales else None})


def generate_hashtags(product: str, keywords: str = "") -> str:
    """Generate hashtags from a product name and comma-separated keywords."""
    if isinstance(keywords, list):
        keywords = " ".join(str(k) for k in keywords)
    words = re.findall(r"[A-Za-z0-9]+", f"{product} {str(keywords).replace(',', ' ')}")
    seen, tags = set(), []
    for w in words:
        if len(w) > 2 and w.lower() not in seen:
            seen.add(w.lower())
            tags.append("#" + w.capitalize())
    joined = "".join(w.capitalize() for w in re.findall(r"[A-Za-z0-9]+", product))
    if joined and "#" + joined.capitalize() not in tags:
        tags.insert(0, "#" + joined)
    return " ".join(tags[:8])


def get_past_campaigns(product: str) -> str:
    """Look up previously saved campaigns for a product in the local database."""
    _init_db()
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT created, platform, headline, body FROM campaigns WHERE product LIKE ? "
            "ORDER BY id DESC LIMIT 3", (f"%{product}%",)).fetchall()
    if not rows:
        return "No past campaigns found for this product."
    return "\n".join(f"[{r[0]}] {r[1]}: {r[2]} | {r[3]}" for r in rows)


def check_ad_copy(platform: str, headline: str, body: str, cta: str) -> str:
    """Validate ad copy against platform limits and policy. Returns JSON {ok, issues}."""
    key = platform.lower().strip().replace(" ", "_")
    key = PLATFORM_ALIASES.get(key, key)
    if key not in PLATFORM_SPECS:
        return json.dumps({"ok": False, "issues": [f"Unknown platform {platform}"]})
    s, issues = PLATFORM_SPECS[key], []
    if not headline.strip():
        issues.append("Headline is empty")
    if len(headline) > s["headline_max"]:
        issues.append(f"Headline is {len(headline)} chars; max {s['headline_max']}")
    if not body.strip():
        issues.append("Body is empty")
    if len(body) > s["body_max"]:
        issues.append(f"Body is {len(body)} chars; max {s['body_max']}")
    if not cta.strip():
        issues.append("Missing call-to-action")
    elif len(cta) > s["cta_max"]:
        issues.append(f"CTA is {len(cta)} chars; max {s['cta_max']}")
    text = f"{headline} {body} {cta}".lower()
    for p in BANNED_PHRASES:
        if p in text:
            issues.append(f"Contains risky claim '{p}' - remove or soften it")
    if text.count("!") > 2:
        issues.append("Too many exclamation marks (max 2)")
    if key == "google_ads" and re.search(r"[\U0001F300-\U0001FAFF]", text):
        issues.append("Emojis are not allowed in Google Ads")
    return json.dumps({"ok": not issues, "issues": issues})


def generate_ad_image(prompt: str, platform: str = "instagram", seed: int = None) -> str:
    """Generate an advertisement image URL matching the prompt and platform aspect ratio.
    Uses Pollinations.ai with FLUX/SDXL models (free, fast, no API key required)."""
    if not prompt or not prompt.strip():
        return ""
    dimensions = {
        "instagram": (1024, 1024),   # Square 1:1
        "facebook": (1200, 628),     # Landscape ~1.91:1
        "linkedin": (1200, 628),     # Landscape ~1.91:1
        "twitter": (1200, 675),      # Landscape 16:9
        "google_ads": (1200, 628),   # Display banner
    }
    key = platform.lower().strip().replace(" ", "_")
    width, height = dimensions.get(key, (1024, 1024))
    
    if seed is None:
        import random
        seed = random.randint(1, 1000000)
    
    import urllib.parse
    clean_prompt = prompt.strip()
    encoded = urllib.parse.quote(clean_prompt)
    return f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&seed={seed}&nologo=true"


def _init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, product TEXT, platform TEXT,
            headline TEXT, body TEXT, cta TEXT, image_url TEXT)""")
        try:
            con.execute("ALTER TABLE campaigns ADD COLUMN image_url TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists


def save_campaign(product: str, platform: str, ad: dict) -> None:
    """Persist the final ad (not an LLM tool - called by the finalizer node)."""
    _init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO campaigns (created, product, platform, headline, body, cta, image_url) VALUES (?,?,?,?,?,?,?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M"), product, platform,
                     ad.get("headline", ""), ad.get("body", ""), ad.get("cta", ""),
                     ad.get("image_url", "")))


# ---- LangChain wrappers (kept separate so the logic above is testable without LangChain) ----
try:
    from langchain_core.tools import tool
    RESEARCH_TOOLS = [tool(f) for f in (web_search, get_platform_specs, get_past_campaigns,
                                        estimate_campaign_budget, generate_hashtags)]
    check_ad_copy_tool = tool(check_ad_copy)
except ImportError:  # pragma: no cover
    RESEARCH_TOOLS, check_ad_copy_tool = [], None
