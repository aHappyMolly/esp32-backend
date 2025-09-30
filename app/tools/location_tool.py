# /app/tools/location_tool.py
# 方案 A：優先吃前端傳來的座標（最準）
# 方案 B：退而求其次，用 IP Geo（可接免費服務或乾脆略過，只做反解） 
import math

def format_latlon(lat, lon):
    return f"{lat:.5f}, {lon:.5f}"

def run_location_tool(device_ctx: dict | None):
    if not device_ctx or "lat" not in device_ctx or "lon" not in device_ctx:
        return None
    lat, lon = float(device_ctx["lat"]), float(device_ctx["lon"])
    tz  = device_ctx.get("tz", "Asia/Taipei")
    place_hint = device_ctx.get("place_hint", "")

    text = f"使用者座標：約 {format_latlon(lat, lon)}（{tz}）。"
    if place_hint:
        text += f" 地點提示：{place_hint}"

    return dict(
        text=text,
        sources=[{
            "type": "location",
            "title": "Device Context",
            "url": None,
            "meta": {"lat": lat, "lon": lon, "tz": tz}
        }]
    )
