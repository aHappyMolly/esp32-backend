# /app/tools/web_tool.py
import requests

def wiki_summary(query: str, lang="zh"):
    try:
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}"
        s = requests.get(url, timeout=6)
        if s.status_code != 200:
            return None
        j = s.json()
        title = j.get("title")
        extract = j.get("extract")
        page_url = j.get("content_urls", {}).get("desktop", {}).get("page")
        if not extract:
            return None
        return dict(
            text=f"【維基百科】{title}\n{extract}",
            sources=[{
                "type": "web",
                "title": title,
                "url": page_url,
                "meta": {"site": "wikipedia"}
            }]
        )
    except Exception:
        return None

def run_web_tool(query: str):
    return wiki_summary(query)  # 目前只做維基，之後可擴充 Bing/SerpAPI
