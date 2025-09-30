# /app/tools/time_tool.py
from datetime import datetime
import zoneinfo

def run_time_tool(tz_name="Asia/Taipei"):
    now = datetime.now(zoneinfo.ZoneInfo(tz_name))
    text = f"現在時間：{now:%Y-%m-%d %H:%M:%S}（{tz_name}），今天是星期{now.isoweekday()}。"
    return dict(
        text=text,
        sources=[{
            "type": "time",
            "title": "Server Clock",
            "url": None,
            "meta": {"tz": tz_name}
        }]
    )
