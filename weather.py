"""
weather.py — Weather awareness for Viora.

Uses Open-Meteo API (free, no API key required) to fetch current weather
for the user's location. Weather context is injected into AI prompts so
Viora can make weather-aware caring messages.

Example: "今天降温了，记得多穿点"
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

try:
    from flask import has_request_context, session as flask_session
except Exception:  # pragma: no cover
    has_request_context = lambda: False
    flask_session = None

import requests

logger = logging.getLogger(__name__)

# Cache: avoid repeated API calls
_weather_cache: dict = {}
_CACHE_DURATION = 1800  # 30 minutes

# Path to store user's location preference
LOCATION_FILE = os.path.join(os.path.dirname(__file__), "data", "location.json")


def _location_file_for_user(user_id: str) -> str:
    if user_id == "default":
        return LOCATION_FILE
    base, ext = os.path.splitext(LOCATION_FILE)
    return f"{base}_{user_id}{ext}"

# Open-Meteo API
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather_context(location: Optional[dict] = None) -> Optional[str]:
    """
    Get a human-readable weather context string for AI prompts.

    Args:
        location: {"lat": float, "lon": float, "city": str} or None

    Returns:
        A string like "当前天气：晴，温度：22°C，空气质量：良好"
        or None if location not set / API failed
    """
    if not location:
        location = get_user_location()

    if not location:
        return None

    cache_key = f"{location.get('lat')}_{location.get('lon')}"
    now = time.time()

    if (cache_key in _weather_cache
            and now - _weather_cache[cache_key]["time"] < _CACHE_DURATION):
        data = _weather_cache[cache_key]["data"]
    else:
        data = _fetch_weather(location.get("lat"), location.get("lon"))
        if data:
            _weather_cache[cache_key] = {"time": now, "data": data}

    if not data:
        return None

    weather_code = data.get("weathercode", 0)
    temp = data.get("temperature_2m", "?")
    humidity = data.get("relative_humidity_2m", "?")
    wind = data.get("wind_speed_10m", "?")
    is_day = data.get("is_day", 1)

    weather_desc = _weather_code_to_desc(weather_code, is_day)

    parts = [
        f"当前天气：{weather_desc}",
        f"温度：{temp}°C",
        f"湿度：{humidity}%",
        f"风速：{wind} km/h",
    ]

    # Note: No caring hints. Weather is purely informational context.
    # The LLM should only mention weather when it naturally fits the conversation.
    return "，".join(parts)


def _weather_code_to_desc(code: int, is_day: int) -> str:
    """Convert WMO weather code to Chinese description."""
    if code == 0:
        return "晴" if is_day else "晴夜"
    elif code in (1, 2):
        return "多云"
    elif code == 3:
        return "阴"
    elif code in (45, 48):
        return "雾"
    elif code in (51, 53, 55):
        return "小雨"
    elif code in (61, 63, 65):
        return "雨"
    elif code in (66, 67):
        return "冻雨"
    elif code in (71, 73, 75):
        return "雪"
    elif code in (80, 81, 82):
        return "阵雨"
    elif code in (95, 96, 99):
        return "雷暴"
    else:
        return "未知"


def _fetch_weather(lat: float, lon: float) -> Optional[dict]:
    """Fetch current weather from Open-Meteo API."""
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m", "relative_humidity_2m",
                "wind_speed_10m", "weathercode", "is_day",
            ],
            "timezone": "Asia/Shanghai",
        }
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        # Flatten current weather into a single dict
        return {
            "weathercode": current.get("weathercode"),
            "temperature_2m": current.get("temperature_2m"),
            "relative_humidity_2m": current.get("relative_humidity_2m"),
            "wind_speed_10m": current.get("wind_speed_10m"),
            "is_day": current.get("is_day"),
        }
    except requests.RequestException as e:
        logger.warning("Weather API error: %s", e)
        return None
    except (KeyError, ValueError) as e:
        logger.warning("Weather API parse error: %s", e)
        return None


# ── Location management ─────────────────────────────────────────────────────

def _current_user_id() -> str:
    if has_request_context() and flask_session is not None:
        return flask_session.get("user_id") or "default"
    return "default"


def get_user_location(user_id: Optional[str] = None) -> Optional[dict]:
    """Load user's saved location preference."""
    location_file = _location_file_for_user(user_id or _current_user_id())
    try:
        if os.path.exists(location_file):
            with open(location_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load location config: %s", e)
    return None


def set_user_location(lat: float, lon: float, city: str = "", user_id: Optional[str] = None) -> dict:
    """Save user's location preference."""
    location = {"lat": lat, "lon": lon, "city": city}
    location_file = _location_file_for_user(user_id or _current_user_id())
    os.makedirs(os.path.dirname(location_file), exist_ok=True)
    with open(location_file, "w", encoding="utf-8") as f:
        json.dump(location, f, ensure_ascii=False, indent=2)
    try:
        from accounts import update_account
        update_account(
            user_id or _current_user_id(),
            location=location,
            location_city=city,
            location_lat=lat,
            location_lon=lon,
            location_updated_at=datetime.now(timezone.utc).isoformat(),
        )
    except ImportError:
        pass
    # Clear cache for this location
    _weather_cache.clear()
    logger.info("Location set: %s (%.4f, %.4f)", city, lat, lon)
    return location


def clear_location(user_id: Optional[str] = None) -> None:
    """Remove user's location preference."""
    uid = user_id or _current_user_id()
    location_file = _location_file_for_user(uid)
    if os.path.exists(location_file):
        os.remove(location_file)
    try:
        from accounts import update_account
        update_account(
            uid,
            location=None,
            location_city="",
            location_lat=None,
            location_lon=None,
            location_updated_at=None,
        )
    except ImportError:
        pass
    _weather_cache.clear()


# ── Prompt injection helper ─────────────────────────────────────────────────

def build_weather_prompt_context(user_id: Optional[str] = None) -> str:
    """
    Build a weather context string to inject into the AI system prompt.
    Returns empty string if no location set or API failed.
    """
    loc = get_user_location(user_id)
    weather_str = get_weather_context(loc)
    if not weather_str:
        return ""

    city = loc.get("city", "") if loc else ""
    return f"[背景参考，不需要主动提及]{city}天气：{weather_str}"
