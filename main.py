# main.py —— 可切換 provider 的後端骨架
# 路由：/stt（語音→文字）、/chat（文字→RAG→LLM）、/health
# 用法：設定環境變數決定 STT / RAG / LLM 供應商，然後啟動：
#   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

import os
import io
import glob
import json
import math
import tempfile
import datetime
import json, re, requests
import zipfile, time
from typing import List, Optional
from bs4 import BeautifulSoup
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, HTMLResponse, JSONResponse
from app.tools.router import run_tools  # ← 新增 import
from pathlib import Path
from app.rag.loaders import (
    read_text_file, extract_pdf_text, extract_docx_text, extract_url_text
)
from app.ui.rag_admin import mount_rag_admin
from urllib.parse import urljoin




# ============= 環境參數（可在 PowerShell/cmd 或 .env 設定） =============
# 供應商選項：
#   STT:   "local"（faster-whisper）| "openai"
#   RAG:   "local"（極簡檢索）| "openai"（OpenAI Embeddings）| "none"
#   LLM:   "openai"（建議；本地 LLM 之後可再擴充）
PROVIDER_STT  = os.getenv("PROVIDER_STT",  "local")     # local | openai
PROVIDER_RAG  = os.getenv("PROVIDER_RAG",  "none")      # local | openai | none
PROVIDER_LLM  = os.getenv("PROVIDER_LLM",  "openai")    # openai


# Whisper（本地 STT）設定
WHISPER_MODEL     = os.getenv("WHISPER_MODEL", "small") # tiny/base/small/medium/large-v3 ...
WHISPER_DEVICE    = os.getenv("WHISPER_DEVICE", "cuda") # cuda | cpu
WHISPER_COMPUTE   = os.getenv("WHISPER_COMPUTE", "float16")  # float16 | int8_float16 | float32

# OpenAI 設定
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL= os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_STT_MODEL  = os.getenv("OPENAI_STT_MODEL", "whisper-1")  # 或 gpt-4o-mini-transcribe
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# RAG 知識來源（會讀這個資料夾下的 .txt/.md/.mdx/.html 檔）
KNOWLEDGE_DIR     = os.getenv("KNOWLEDGE_DIR", "./knowledge")
RAG_TOP_K         = int(os.getenv("RAG_TOP_K", "4"))

# 語言偏好（繁體）
LANG_HINT         = os.getenv("LANG_HINT", "zh")     # STT 語言提示
REPLY_LANG        = os.getenv("REPLY_LANG", "zh-TW") # LLM 回覆語言偏好

# TTS 設定（NEW）
PROVIDER_TTS   = os.getenv("PROVIDER_TTS", "openai")  # openai | piper
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")  # OpenAI TTS 型號
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")            # 聲音樣式
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "")  # 例：/models/zh_CN-piper-medium.onnx


# ============= FastAPI 初始化與 CORS =============
app = FastAPI(title="ESP32 Voice Backend (Pluggable)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# 只是註冊一個 /rag 的 GET 路由 ( app 已存在 → rag_admin)
mount_rag_admin(app)


# ============= Provider 介面定義 =============
class STTProvider:
    def transcribe(self, audio_bytes: bytes) -> str:
        raise NotImplementedError

class RAGProvider:
    def retrieve(self, query: str, k: int = 4) -> List[str]:
        return []

class LLMProvider:
    def chat(self, user_text: str, context_chunks: List[str], lang: str = "zh-TW") -> str:
        raise NotImplementedError
    
class TTSProvider:
    def synth(self, text: str, sr: int = 16000) -> bytes:
        """回傳整段 WAV（bytes）。"""
        raise NotImplementedError
    



# ============= STT Providers =============
# A) local: faster-whisper（GPU/CPU）
class LocalWhisperProvider(STTProvider):
    def __init__(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,         # "cuda" or "cpu"
            compute_type=WHISPER_COMPUTE   # "float16" 等
        )

    def transcribe(self, audio_bytes: bytes) -> str:
        # faster-whisper 最穩吃檔案路徑 → 先落地暫存檔
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name
        segments, info = self.model.transcribe(
            path,
            language=LANG_HINT,
            vad_filter=True,
            temperature=0.0,
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            initial_prompt="請使用繁體中文輸出，保留台灣用語。專有名詞：ESP32、伺服馬達、麥克風、相機、韌體、OTA、上傳下載。"
        )
        text = "".join(seg.text for seg in segments).strip()
        return text

# B) openai STT
class OpenAIWhisperProvider(STTProvider):
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing for OpenAI STT")
        from openai import OpenAI
        self.client = OpenAI()

    def transcribe(self, audio_bytes: bytes) -> str:
        from tempfile import NamedTemporaryFile
        # OpenAI SDK 需要 file-like；這裡也走暫存檔
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        # whisper-1：傳統 Whisper；若用 gpt-4o-mini-transcribe 需依官方語音API更新
        r = self.client.audio.transcriptions.create(
            model=OPENAI_STT_MODEL,
            file=open(tmp_path, "rb"),
            # 若要強制 zh，可用 language="zh"
            # language="zh",
            response_format="json"
        )
        # r.text 將是文字
        return r.text.strip()


# ============= RAG Providers =============

def _load_documents_from_dir(path: str) -> List[str]:
    """讀取 knowledge 中的各類文件 → 純文字 → 簡單切塊"""
    chunks: List[str] = []
    exts = [".txt", ".md", ".mdx", ".html", ".htm", ".pdf", ".docx"]
    for p in Path(path).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue

        text = None
        suf = p.suffix.lower()
        if suf in [".txt", ".md", ".mdx", ".html", ".htm"]:
            text = read_text_file(p)
        elif suf == ".pdf":
            text = extract_pdf_text(p)
        elif suf == ".docx":
            text = extract_docx_text(p)

        if not (text and text.strip()):
            continue

        # --- 你的原本切塊策略（每塊 ≤800 字） ---
        for para in text.splitlines():
            t = (para or "").strip()
            if not t:
                continue
            while len(t) > 800:
                chunks.append(t[:800])
                t = t[800:]
            if t:
                chunks.append(t)
    return chunks

    # --- 若未來要 OCR 掃描版 PDF：在 extract_pdf_text 中偵測 page.images，必要時切圖丟 pytesseract


# --- 圖片建議：從工具來源 URL 抓 og:image + 從 knowledge/.meta.json 挑關鍵圖 ---
# 即使目前的 RAG chunk 沒帶「來源檔名」，也可先用「工具來源 URL」與「knowledge 的 .meta.json」來挑圖
def _images_from_tool_sources(sources, max_n=3):
    imgs = []
    for s in sources or []:
        url = (s.get("url") if isinstance(s, dict) else None) or ""
        if not url or not url.startswith("http"): 
            continue
        try:
            r = requests.get(url, timeout=6, headers={"User-Agent":"Mozilla/5.0"})
            html = r.text
            # og:image
            m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
            if m:
                img = m.group(1).strip()
                if not img.startswith("http"):
                    img = urljoin(url, img)
                imgs.append(img)
        except Exception:
            pass
        if len(imgs) >= max_n:
            break
    return imgs

# 取代 main.py 內的 _images_from_meta_by_query
def _images_from_meta_by_query(query: str, max_n=3):
    """掃描 knowledge/*.meta.json，若 title/url/tags 有任何 token 命中，就收集 image/images。"""
    def _tokens(s: str):
        s = (s or "").lower().strip()
        out, buf = [], []
        for ch in s:
            # 中文：逐字；英文數字：聚成詞
            if "\u4e00" <= ch <= "\u9fff":
                if buf: out.append("".join(buf)); buf=[]
                out.append(ch)
            elif ch.isalnum():
                buf.append(ch)
            else:
                if buf: out.append("".join(buf)); buf=[]
        if buf: out.append("".join(buf))
        # 過濾太短的英文 token
        return [t for t in out if (len(t)>=2 or ("\u4e00"<=t<= "\u9fff"))]

    q_tokens = _tokens(query)
    if not q_tokens:
        return []

    out = []
    root = Path(KNOWLEDGE_DIR)
    for m in root.glob("*.meta.json"):
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue

        hay = " ".join([
            str(meta.get("title","")),
            str(meta.get("url","")),
            " ".join(meta.get("tags",[])),
        ]).lower()

        # 只要任一 token 出現在 title/url/tags 就視為命中
        if any(tok in hay for tok in q_tokens):
            if meta.get("image"):
                out.append(meta["image"])
            if isinstance(meta.get("images"), list):
                out.extend([x for x in meta["images"] if isinstance(x, str)])

        if len(out) >= max_n:
            break

    # 去重、限量
    seen, uniq = set(), []
    for u in out:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq[:max_n]




# A) local RAG（免外部依賴的極簡檢索：tf-idf-like + overlap 分數）
class LocalNaiveRAG(RAGProvider):
    def __init__(self, knowledge_dir: str):
        self.chunks = _load_documents_from_dir(knowledge_dir)
        # 建一些輕量索引（word -> df）
        self.N = len(self.chunks)
        self.df = {}
        for c in self.chunks:
            tokens = set(self._tokenize(c))
            for t in tokens:
                self.df[t] = self.df.get(t, 0) + 1

    def _tokenize(self, s: str) -> List[str]:
        # 極簡 token（中文：逐字；英文：小寫拆詞）；可換 jieba 等
        out = []
        buf = []
        for ch in s:
            if "\u4e00" <= ch <= "\u9fff":
                out.append(ch)
            elif ch.isalnum():
                buf.append(ch.lower())
            else:
                if buf:
                    out.append("".join(buf))
                    buf = []
        if buf: out.append("".join(buf))
        return out

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        if not self.chunks:
            return []
        q_tokens = self._tokenize(query)
        # tf-idf like：sum( tf * idf ) + 字元重疊 bonus
        def score(text: str) -> float:
            tokens = self._tokenize(text)
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q_tokens:
                idf = math.log((self.N + 1) / (1 + self.df.get(t, 0) )) + 1.0
                s += tf.get(t, 0) * idf
            # 加一點字元重疊度
            overlap = len(set(query) & set(text)) / (len(set(query)) + 1e-6)
            return s + overlap
        ranked = sorted(self.chunks, key=score, reverse=True)
        return ranked[:k]

# B) openai RAG（用 OpenAI Embeddings 做簡單向量檢索；向量存記憶體）
class OpenAIRAG(RAGProvider):
    def __init__(self, knowledge_dir: str):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing for OpenAI RAG")
        from openai import OpenAI
        self.client = OpenAI()
        self.model = OPENAI_EMBED_MODEL
        self.chunks = _load_documents_from_dir(knowledge_dir)
        self.vecs = self._embed_batch(self.chunks) if self.chunks else []

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # OpenAI embeddings：一次可送多筆
        r = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in r.data]

    def _cos(self, a: List[float], b: List[float]) -> float:
        if not a or not b: return 0.0
        s = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        return s / (na*nb + 1e-9)

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        if not self.chunks:
            return []
        qv = self._embed_batch([query])[0]
        scored = [(self._cos(qv, v), i) for i, v in enumerate(self.vecs)]
        scored.sort(reverse=True)
        top = [self.chunks[i] for _, i in scored[:k]]
        return top


# ============= LLM Provider（OpenAI） =============
class OpenAILLM(LLMProvider):
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing for OpenAI LLM")
        from openai import OpenAI
        self.client = OpenAI()
        self.model  = OPENAI_CHAT_MODEL

    def chat(self, user_text: str, context_chunks: List[str], lang: str = "zh-TW") -> str:
        system_prompt = (
            "你是一位語音助理，所有回覆一律使用繁體中文（臺灣），口語、簡短、直覺。"
            "若有附帶的參考內容，請盡量引用其中資訊回答。"
        )
        context_block = ""
        if context_chunks:
            joined = "\n\n---\n".join(context_chunks)
            context_block = f"以下是可參考的內容片段，請在合適時引用：\n{joined}\n\n"

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": context_block + f"\n使用者說：{user_text}"}
        ]
        r = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=0.5,
            max_tokens=300,
        )
        return r.choices[0].message.content.strip()
    

# ============= TTS Provider（NEW） =============
# A) OpenAI TTS（雲端，最簡單）

class OpenAITTSProvider(TTSProvider):
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing for OpenAI TTS")
        import requests  # 確保 requirements.txt 有 requests
        self.requests = requests
        self.model  = OPENAI_TTS_MODEL or "gpt-4o-mini-tts"
        self.voice  = OPENAI_TTS_VOICE or "alloy"
        self.base   = "https://api.openai.com/v1/audio/speech"
        self.headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

    def synth(self, text: str, sr: int = 16000) -> bytes:
        payload = {"model": self.model, "voice": self.voice, "input": text, "format": "wav", "sample_rate": int(sr)}
        r = self.requests.post(self.base, headers=self.headers, json=payload, timeout=60)
        if r.status_code == 400:
            # 某些模型/版本不接受 sample_rate，移除後重試
            payload.pop("sample_rate", None)
            r = self.requests.post(self.base, headers=self.headers, json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI TTS HTTP {r.status_code}: {r.text[:200]}")
        b = r.content
        if not (len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE"):
            raise RuntimeError("TTS returned non-WAV")
        return b





# B) Piper（本地，離線）
class PiperTTSProvider(TTSProvider):
    def __init__(self):
        if not PIPER_MODEL_PATH or not os.path.exists(PIPER_MODEL_PATH):
            raise RuntimeError("Piper model not found; set PIPER_MODEL_PATH")
        # 依你安裝的封裝而定，以下示意常見 API
        # pip install piper-phonemize piper-tts 或相容套件
        from piper import PiperVoice
        self.voice = PiperVoice.load(PIPER_MODEL_PATH)

    def synth(self, text: str, sr: int = 16000) -> bytes:
        # Piper 預設會回 PCM/或直接寫檔；這裡將其包成 WAV
        import wave, struct
        import numpy as np
        # 產生 16-bit mono PCM（numpy int16）
        pcm = self.voice.synthesize(text, length_scale=1.0)  # 取得 float32 PCM（依套件版本不同）
        if pcm.dtype != np.int16:
            # 轉 16-bit（簡單限幅）
            x = np.clip(pcm, -1.0, 1.0)
            pcm16 = (x * 32767.0).astype(np.int16)
        else:
            pcm16 = pcm

        # 封裝 WAV（16-bit mono）
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()


# ============= 加入一個產生 beep WAV 的小工具 =============
# --- fallback: 產生一段 0.8 秒 1kHz 的 16-bit mono WAV ---
def make_beep_wav(sr: int = 16000, dur_s: float = 0.8, freq: float = 1000.0) -> bytes:
    import math, io, wave, struct
    n = int(sr * dur_s)
    amp = 0.4
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            v = int(amp * 32767 * math.sin(2*math.pi*freq*i/sr))
            wf.writeframes(struct.pack("<h", v))
    return buf.getvalue()



# =============== RAG 上傳相關 ================

# 重建索引 ( 讓相關 API 可以呼叫它重整 RAG)
def _reload_rag_provider():
    """重新載入 RAG（依照目前 PROVIDER_RAG），以便上傳/新增網址後立即生效。"""
    global rag_provider
    if PROVIDER_RAG == "local":
        rag_provider = LocalNaiveRAG(KNOWLEDGE_DIR)
    elif PROVIDER_RAG == "openai":
        rag_provider = OpenAIRAG(KNOWLEDGE_DIR)
    elif PROVIDER_RAG == "none":
        rag_provider = RAGProvider()


# 列出 knowledge 目錄檔案（給 /rag/list 用）
def _list_knowledge_files():
    root = Path(KNOWLEDGE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(root.glob("*")):
        if p.is_file():
            st = p.stat()
            items.append({
                "name": p.name,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "ext": p.suffix.lower(),
            })
    return items



# ============= Provider 載入器 =============
stt_provider: Optional[STTProvider] = None
rag_provider: Optional[RAGProvider] = None
llm_provider: Optional[LLMProvider] = None
tts_provider: Optional[TTSProvider] = None

@app.on_event("startup")
def load_providers():
    global stt_provider, rag_provider, llm_provider, tts_provider

    # STT
    if PROVIDER_STT == "local":
        stt_provider = LocalWhisperProvider()
    elif PROVIDER_STT == "openai":
        stt_provider = OpenAIWhisperProvider()
    else:
        raise RuntimeError(f"Unknown PROVIDER_STT: {PROVIDER_STT}")

    # RAG
    if PROVIDER_RAG == "local":
        rag_provider = LocalNaiveRAG(KNOWLEDGE_DIR)
    elif PROVIDER_RAG == "openai":
        rag_provider = OpenAIRAG(KNOWLEDGE_DIR)
    elif PROVIDER_RAG == "none":
        rag_provider = RAGProvider()  # 空實作，retrieve() 回 []
    else:
        raise RuntimeError(f"Unknown PROVIDER_RAG: {PROVIDER_RAG}")

    # LLM
    if PROVIDER_LLM == "openai":
        llm_provider = OpenAILLM()
    else:
        raise RuntimeError(f"Unknown PROVIDER_LLM: {PROVIDER_LLM}")

    # TTS（NEW）
    if PROVIDER_TTS == "openai":
        tts_provider = OpenAITTSProvider()
    elif PROVIDER_TTS == "piper":
        tts_provider = PiperTTSProvider()
    else:
        raise RuntimeError(f"Unknown PROVIDER_TTS: {PROVIDER_TTS}")
    



# ============= 路由 =============
@app.get("/health")
def health():
    return {
        "ok": True,
        "stt": PROVIDER_STT,
        "rag": PROVIDER_RAG,
        "llm": PROVIDER_LLM,
        "tts": dict(
            provider=PROVIDER_TTS,
            openai=dict(model=OPENAI_TTS_MODEL, voice=OPENAI_TTS_VOICE, has_key=bool(OPENAI_API_KEY)),
            piper=dict(model_path=bool(PIPER_MODEL_PATH)),
        ),
        "whisper": dict(model=WHISPER_MODEL, device=WHISPER_DEVICE, compute=WHISPER_COMPUTE),
        "openai": dict(
            chat_model=OPENAI_CHAT_MODEL,
            embed_model=OPENAI_EMBED_MODEL,
            stt_model=OPENAI_STT_MODEL,
            has_key=bool(OPENAI_API_KEY),
        ),
        "knowledge_loaded": (isinstance(rag_provider, RAGProvider) and
                             hasattr(rag_provider, "chunks") and
                             len(getattr(rag_provider, "chunks", []))),
    }



# --- main.py 追加：輕量預熱 ---
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse

_warm_ready = False

def _do_warm():
    global _warm_ready
    try:
        # 觸發 RAG/LLM/TTS 供應器的 lazy init（皆為輕量呼叫）
        try:
            _ = getattr(rag_provider, "retrieve", lambda q, k=1: [])("hi", 1)
        except Exception as e:
            print("[warm] rag:", e)
        try:
            if llm_provider:
                _ = llm_provider.chat("ping", [], lang="zh-TW")
        except Exception as e:
            print("[warm] llm:", e)
        # 若你想連 TTS 也預熱，放開下面 3 行即可（會稍增冷啟時延）
        # try:
        #     if tts_provider:
        #         _ = tts_provider.synth("hi", sr=16000)
        # except Exception as e:
        #     print("[warm] tts:", e)

        _warm_ready = True
    except Exception as e:
        print("[warm] error:", e)

@app.get("/warm")
def warm(tasks: BackgroundTasks):
    """非阻塞預熱；若尚未 ready，回 202 並在背景跑一次預熱。"""
    global _warm_ready
    if _warm_ready:
        return {"ok": True, "ready": True}
    tasks.add_task(_do_warm)
    return JSONResponse({"ok": True, "ready": False}, status_code=202)



# /stt：支援 multipart "file" 與 raw bytes（Content-Type: audio/wav）
@app.post("/stt")
async def stt(file: Optional[UploadFile] = File(None), request: Request = None):
    if stt_provider is None:
        raise HTTPException(500, "STT provider not loaded")
    if file is not None:
        audio_bytes = await file.read()
    else:
        audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(400, "No audio provided")
    try:
        text = stt_provider.transcribe(audio_bytes)
        return {"ok": True, "text": text}
    except Exception as e:
        raise HTTPException(500, f"STT error: {e}")


class ChatIn(BaseModel):
    text: str
    device_context: Optional[dict] = None   # ← 允許前端附座標/時區

@app.post("/chat")
def chat(body: ChatIn):
    if llm_provider is None or rag_provider is None:
        raise HTTPException(500, "Providers not loaded")
    try:
        tool_chunks, tool_sources = run_tools(body.text, body.device_context)  # ← 新增
        rag_chunks = rag_provider.retrieve(body.text, k=RAG_TOP_K) if hasattr(rag_provider, "retrieve") else []
        chunks = tool_chunks + rag_chunks

        reply  = llm_provider.chat(body.text, chunks, lang=REPLY_LANG)

        images_a = _images_from_tool_sources(tool_sources, max_n=3)
        images_b = _images_from_meta_by_query(body.text, max_n=3)
        images = (images_a + images_b)[:3]

        return {"ok": True, "reply": reply, "ctx_used": len(chunks), "sources": tool_sources, "images": images}

    except Exception as e:
        raise HTTPException(500, f"Chat error: {e}")
    



class TTSIn(BaseModel):
    text: str
    sr: Optional[int] = 16000

@app.post("/tts")
def tts(body: TTSIn):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    sr = int(body.sr or 16000)

    # 先預備一個 resp 物件，方便附加 header
    out_bytes: bytes = b""
    impl = "rest"
    err  = ""

    try:
        if tts_provider is None:
            impl = "beep"
            out_bytes = make_beep_wav(sr=sr, dur_s=0.8, freq=1000.0)
        else:
            out_bytes = tts_provider.synth(text, sr=sr)
    except Exception as e:
        impl = "beep"
        err  = str(e)
        out_bytes = make_beep_wav(sr=sr, dur_s=0.8, freq=600.0)

    resp = StreamingResponse(io.BytesIO(out_bytes), media_type="audio/wav")
    resp.headers["X-TTS-Impl"] = impl
    if err:
        resp.headers["X-TTS-Error"] = err[:300]
    return resp

# （可選）GET /tts 方便瀏覽器測
@app.get("/tts")
def tts_get(text: Optional[str] = "", sr: Optional[int] = 16000):
    t = (text or "").strip() or "系統測試音"
    return tts(TTSIn(text=t, sr=sr))


@app.get("/tts_mp3")
def tts_mp3(text: str = "", voice: str = "alloy"):
    """回 MP3（audio/mpeg），給 ESP32-S3 + ESP8266Audio 直接串流播放。"""
    t = (text or "").strip()
    if not t:
        raise HTTPException(400, "empty text")

    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY missing")

    import requests
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_TTS_MODEL or "gpt-4o-mini-tts",
        "voice": voice or (OPENAI_TTS_VOICE or "alloy"),
        "input": t,
        "format": "mp3",          # ★ 要 MP3
        # MP3 自帶採樣率，通常不指定 sample_rate
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(500, f"TTS failed: {r.status_code} {r.text[:200]}")

    b = r.content
    # 明確告知長度與關閉連線 → ESP32 串流更穩
    return Response(
        content=b,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(b)),
            "Cache-Control": "no-store",
            "Connection": "close",
        },
    )



# 方便把資料丟進 RAG ( 上傳檔案 → 存進 knowledge → 重新索引 )
@app.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    """支援 .txt .md .html .pdf .docx，上傳後存進 knowledge/ 原檔名，並重建索引。"""
    suf = Path(file.filename).suffix.lower()
    if suf not in [".txt", ".md", ".mdx", ".html", ".htm", ".pdf", ".docx"]:
        raise HTTPException(400, "Unsupported file type")

    # 保存到 knowledge/
    dst = Path(KNOWLEDGE_DIR) / Path(file.filename).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    dst.write_bytes(data)

    # 重建索引
    _reload_rag_provider()

    return {"ok": True, "saved": str(dst.name), "chunks": len(getattr(rag_provider, "chunks", []))}


# 方便把資料丟進 RAG (加入網址 → 轉文字 → 存成 .txt → 重新索引)
@app.post("/rag/add_url")
async def rag_add_url(url: str = Form(...)):
    """把網址主要內文抓下來，存成 .txt 到 knowledge，並重建索引。"""
    text = extract_url_text(url)
    if not text or not text.strip():
        # 加強除錯：在伺服器日誌印一次，前端看到 400 但你可在後端查看是哪個 URL 抓不到
        print(f"[rag_add_url] extract failed: {url}")
        raise HTTPException(400, "Failed to extract text from URL (no text)")
    safe = "".join(ch for ch in url if ch.isalnum() or ch in "-_").strip("-_")[:60] or "page"
    dst = Path(KNOWLEDGE_DIR) / f"{safe}.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")

    # 擷取 title 與 og:image，寫同名 meta 檔
    title = ""; og_image = ""
    try:
        r = requests.get(url, timeout=6, headers={"User-Agent":"Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else ""
        og = soup.find("meta", property="og:image")
        if og and og.get("content"): og_image = og.get("content").strip()
    except Exception:
        pass

    meta = {"url": url, "title": title}
    if og_image: meta["image"] = og_image
    (Path(KNOWLEDGE_DIR) / f"{safe}.txt.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


    _reload_rag_provider()
    return {"ok": True, "saved": dst.name, "chars": len(text), "chunks": len(getattr(rag_provider, "chunks", []))}



@app.get("/rag/list")
def rag_list():
    return {"ok": True, "files": _list_knowledge_files()}

@app.post("/rag/delete")
def rag_delete(name: str = Form(...)):
    """刪除 knowledge 內指定檔案，並重建索引。"""
    root = Path(KNOWLEDGE_DIR).resolve()
    target = (root / name).resolve()
    if not str(target).startswith(str(root)) or not target.exists() or not target.is_file():
        raise HTTPException(400, "file not found")
    target.unlink()
    _reload_rag_provider()
    return {"ok": True, "deleted": name, "chunks": len(getattr(rag_provider, "chunks", []))}

@app.post("/rag/reindex")
def rag_reindex():
    """手動重建索引。"""
    _reload_rag_provider()
    return {"ok": True, "chunks": len(getattr(rag_provider, "chunks", []))}


# RAG 從管理頁上傳後，再打包下載，把線上內容拉回本機備份 
@app.get("/rag/backup")
def rag_backup():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        root = Path(KNOWLEDGE_DIR)
        for p in root.glob("*"):
            if p.is_file():
                z.write(p, arcname=p.name)
    buf.seek(0)
    fname = f"knowledge_backup_{int(time.time())}.zip"
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})



# ====== ⬇ 新增：WAV TTS 串流端點（單聲道 22050Hz） ======
# 這裡示範：用你既有的 TTS 模組回傳「整段 wav bytes」後，再邊讀邊送。
# 若你使用 OpenAI python sdk v1：
#   audio_resp = client.audio.speech.with_streaming_response.create(
#       model="gpt-4o-mini-tts", voice="alloy",
#       input=text, format="wav", sample_rate=22050
#   )
#   再把 audio_resp.iter_bytes() 串出來即可。
#
# 下面給一個最簡單的「一次拿到 wav bytes」→ 包裝成 generator 的版本，
# 你可以直接把 get_wav_bytes(text) 換成你的實作。

def get_wav_bytes(text: str) -> bytes:
    """
    TODO: 用你的 TTS 取得 'wav (PCM16, mono, 22050Hz)' 原始位元組。
    這裡先提供一個占位示例：回傳一段很短的無聲 wav（44-byte header + 0 data），方便你先跑通流程。
    跑通後，請將此函式換成你實際的 TTS 產生 wav 的程式。
    """
    # 生成一個合法的 22050Hz mono 16-bit, 長度 0 的最小 WAV（僅示範）
    # 你實作時應該回傳：wav_header + PCM資料
    import struct
    sample_rate = 22050
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data = b""

    # RIFF/WAVE header
    riff = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    fmt  = b"fmt " + struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample)
    data_chunk = b"data" + struct.pack("<I", len(data)) + data
    return riff + fmt + data_chunk

@app.get("/tts_wav")
def tts_wav(text: str = Query(..., description="要合成的文字")):
    # 取得整段 wav bytes（先跑通，你再換成真 TTS）
    wav_bytes = get_wav_bytes(text)

    # 用 generator 邊送邊到 ESP；這裡先一次 yield 完整 wav_bytes
    # 若你有真正串流（iter_bytes），可以逐塊 yield，前 44 bytes 先送 header，之後送 data
    def gen():
        CHUNK = 4096
        mv = memoryview(wav_bytes)
        for i in range(0, len(wav_bytes), CHUNK):
            yield mv[i:i+CHUNK]

    headers = {
        "Content-Type": "audio/wav",
        # 非必需，但對部分客戶端有幫助：
        "Cache-Control": "no-store",
        # 若能知道長度最好也回 Content-Length；不知長度也沒關係（chunked）
        "Content-Length": str(len(wav_bytes)),
        "Connection": "close",
    }
    return StreamingResponse(gen(), headers=headers)


# ============= 開發啟動（直接 python main.py 時） =============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
