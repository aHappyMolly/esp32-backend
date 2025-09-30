# /app/tools/web_tool.py
import requests

def wiki_summary(query: str, lang="zh"):
    # 極簡：用維基 opensearch 找頁，再抓 summary
    s = requests.get(
        f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}",
        timeout=6
    )
    if s.status_code != 200:
        return None
    j = s.json()
    title = j.get("title")
    extract = j.get("extract")
    url = j.get("content_urls",{}).get("desktop",{}).get("page")
    if not extract:
        return None
    return dict(text=f"【維基百科】{title}\n{extract}", 
                sources=[{"type":"web","title":title,"url":url,"meta":{"site":"wikipedia"}}])

def run_web_tool(query: str):
    hit = wiki_summary(query)
    if hit: 
        return hit
    # 之後可再擴充：bing_search(query) → 選前2-3條，抽摘要
    return None
