# /app/tools/location_tool.py
# 方案 A：優先吃前端傳來的座標（最準）
# 方案 B：退而求其次，用 IP Geo（可接免費服務或乾脆略過，只做反解） 
import math

def format_latlon(lat, lon):
    return f"{lat:.5f}, {lon:.5f}"

def run_location_tool(device_ctx: dict|None):
    # device_ctx 由前端附帶：{"lat":..., "lon":..., "tz":"Asia/Taipei", "place_hint":"NCHU"}
    if not device_ctx or "lat" not in device_ctx or "lon" not in device_ctx:
        return None
    lat, lon = float(device_ctx["lat"]), float(device_ctx["lon"])
    text = f"使用者座標：約 {format_latlon(lat, lon)}。"
    # 你也可以加：行政區反查（呼叫 Nominatim 或你的地理 API）
    return dict(
        text=text,
        sources=[{"type":"location","title":"Device Context","url":None,"meta":{"lat":lat,"lon":lon}}]
    )
