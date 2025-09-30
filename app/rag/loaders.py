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
    try:
        import trafilatura
        html = trafilatura.fetch_url(url, timeout=10)
        if html:
            text = trafilatura.extract(html, include_comments=False, include_links=False)
            if text and text.strip():
                return text.strip()
    except Exception:
        pass
    # fallback: requests + bs4
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        # 抓主要段落文字
        ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        txt = "\n".join(p for p in ps if p)
        # 簡單清洗：去除多空白
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        return txt or None
    except Exception:
        return None
