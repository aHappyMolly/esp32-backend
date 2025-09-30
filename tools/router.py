# /app/tools/router.py
import re
from .time_tool import run_time_tool
from .location_tool import run_location_tool
from .web_tool import run_web_tool

TIME_PAT = re.compile(r"(現在|幾點|今天|日期|星期幾)")
LOC_PAT  = re.compile(r"(附近|哪裡|位置|多遠|距離|座標)")
WEB_PAT  = re.compile(r"(搜尋|查一下|最新|新聞|誰是|是誰|維基|Wikipedia)")

def plan_tools(user_text: str, device_ctx: dict|None):
    plan = []
    if TIME_PAT.search(user_text):
        plan.append(("time", {}))
    if device_ctx and ("lat" in (device_ctx or {})):
        if LOC_PAT.search(user_text):
            plan.append(("location", {"ctx": device_ctx}))
    if WEB_PAT.search(user_text):
        plan.append(("web", {"q": user_text}))
    return plan

def run_tools(user_text: str, device_ctx: dict|None):
    chunks, sources = [], []
    for name, kw in plan_tools(user_text, device_ctx):
        if name == "time":
            r = run_time_tool(device_ctx.get("tz") if device_ctx else "Asia/Taipei")
        elif name == "location":
            r = run_location_tool(kw.get("ctx"))
        elif name == "web":
            r = run_web_tool(kw.get("q"))
        else:
            r = None
        if r and r.get("text"):
            chunks.append(r["text"])
            sources.extend(r.get("sources", []))
    return chunks, sources
