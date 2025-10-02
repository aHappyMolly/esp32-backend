# /app/rag/loaders.py
from pathlib import Path
from typing import Optional
import re

# ---- TXT / MD / HTML 純文字讀取 ----
def read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

# ---- PDF → text ----
def extract_pdf_text(path: Path) -> Optional[str]:
    try:
        import pdfplumber
    except Exception:
        return None
    try:
        txts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    txts.append(t)
        return "\n".join(txts).strip() or None
    except Exception:
        return None

# ---- DOCX → text ----
def extract_docx_text(path: Path) -> Optional[str]:
    try:
        from docx import Document
    except Exception:
        return None
    try:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip() or None
    except Exception:
        return None

# ---- URL → text（優先 trafilatura，退而求其次 bs4） ----
def extract_url_text(url: str) -> Optional[str]:
    """
    盡量從 URL 擷取正文：
    1) trafilatura（最好）
    2) 直接 PDF → 交給 extract_pdf_text
    3) requests + bs4（常見網站）
    4) 最後退路：簡單去除標籤的 <body> 文字
    回傳 None 表示真的抓不到。
    """
    # --- 簡單辨識 PDF 直鏈 ---
    try:
        if re.search(r"\.pdf($|\?)", url, re.I):
            import requests, tempfile, os
            r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and r.content:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(r.content)
                    tmp_path = Path(tmp.name)
                try:
                    txt = extract_pdf_text(tmp_path)  # 你已經有這個函式
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                if txt and txt.strip():
                    return txt.strip()
    except Exception:
        pass

    # --- 優先 trafilatura ---
    try:
        import trafilatura
        html = trafilatura.fetch_url(url, timeout=12)
        if html:
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_links=False,
                favor_precision=True,
                with_metadata=False,
            )
            if text and text.strip():
                return text.strip()
    except Exception:
        pass

    # --- requests + bs4（加強 headers，處理常見擋爬） ---
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get(
            url,
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        if r.status_code != 200 or not r.text:
            return None
        soup = BeautifulSoup(r.text, "html.parser")

        # 先試「主要內文」選擇器（常見 CMS）
        main = soup.select_one("article") or soup.select_one("main")
        container = main if main else soup

        ps = [p.get_text(" ", strip=True) for p in container.find_all("p")]
        txt = "\n".join(p for p in ps if p)
        # 簡單清洗：去除 3+ 連續空行
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        if txt:
            return txt
    except Exception:
        pass

    # --- 最後退路：把所有標籤去掉，抓 <body> 文字 ---
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and r.text:
            soup = BeautifulSoup(r.text, "html.parser")
            body = soup.body.get_text(" ", strip=True) if soup.body else soup.get_text(" ", strip=True)
            body = re.sub(r"\s{3,}", "  ", body).strip()
            return body or None
    except Exception:
        pass

    return None