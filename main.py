# main.py —— 可切換 provider 的後端骨架 + /tts_wav 串流
# 路由：/stt（語音→文字）、/chat（文字→RAG→LLM）、/tts（WAV 一次性）、/tts_wav（WAV 串流）、/tts_mp3（MP3 一次性）、/health
# 用法：設定環境變數決定 STT / RAG / LLM / TTS 供應商，啟動：
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
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, HTMLResponse, JSONResponse
from pathlib import Path
from urllib.parse import urljoin

# 你的工具與 RAG/管理頁（保留）
from app.tools.router import run_tools
from app.rag.loaders import (read_text_file, extract_pdf_text, extract_docx_text, extract_url_text)
from app.ui.rag_admin import mount_rag_admin


# ============= 環境參數（可在 PowerShell/cmd 或 .env 設定） =============
# 供應商選項：
#   STT:   "local"（faster-whisper）| "openai"
#   RAG:   "local"（極簡檢索）| "openai"（OpenAI Embeddings）| "none"
#   LLM:   "openai"
#   TTS:   "openai" | "piper"
PROVIDER_STT  = os.getenv("PROVIDER_STT",  "local")     # local | openai
PROVIDER_RAG  = os.getenv("PROVIDER_RAG",  "none")      # local | openai | none
PROVIDER_LLM  = os.getenv("PROVIDER_LLM",  "openai")    # openai
PROVIDER_TTS  = os.getenv("PROVIDER_TTS",  "openai")    # openai | piper

# Whisper（本地 STT）設定
WHISPER_MODEL     = os.getenv("WHISPER_MODEL", "small") # tiny/base/small/medium/large-v3 ...
WHISPER_DEVICE    = os.getenv("WHISPER_DEVICE", "cuda") # cuda | cpu
WHISPER_COMPUTE   = os.getenv("WHISPER_COMPUTE", "float16")  # float16 | int8_float16 | float32

# OpenAI 設定
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL= os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_STT_MODEL  = os.getenv("OPENAI_STT_MODEL", "whisper-1")  # 或 gpt-4o-mini-transcribe
OPENAI_BASE_URL   = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# RAG 知識來源（會讀這個資料夾下的 .txt/.md/.mdx/.html 檔）
KNOWLEDGE_DIR     = os.getenv("KNOWLEDGE_DIR", "./knowledge")
RAG_TOP_K         = int(os.getenv("RAG_TOP_K", "4"))

# 語言偏好（繁體）
LANG_HINT         = os.getenv("LANG_HINT", "zh")
REPLY_LANG        = os.getenv("REPLY_LANG", "zh-TW")

# TTS 設定
OPENAI_TTS_MODEL  = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE  = os.getenv("OPENAI_TTS_VOICE", "alloy")
PIPER_MODEL_PATH  = os.getenv("PIPER_MODEL_PATH", "")  # 例：/models/zh_CN-piper-medium.onnx


# ============= FastAPI 初始化與 CORS =============
app = FastAPI(title="ESP32 Voice Backend (Pluggable)", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# mount RAG admin
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
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE
        )

    def transcribe(self, audio_bytes: bytes) -> str:
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
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        r = self.client.audio.transcriptions.create(
            model=OPENAI_STT_MODEL,
            file=open(tmp_path, "rb"),
            response_format="json"
        )
        return r.text.strip()


# ============= RAG Providers =============
def _load_documents_from_dir(path: str) -> List[str]:
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

def _images_from_tool_sources(sources, max_n=3):
    imgs = []
    for s in sources or []:
        url = (s.get("url") if isinstance(s, dict) else None) or ""
        if not url or not url.startswith("http"):
            continue
        try:
            r = requests.get(url, timeout=6, headers={"User-Agent":"Mozilla/5.0"})
            html = r.text
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

def _images_from_meta_by_query(query: str, max_n=3):
    def _tokens(s: str):
        s = (s or "").lower().strip()
        out, buf = [], []
        for ch in s:
            if "\u4e00" <= ch <= "\u9fff":
                if buf: out.append("".join(buf)); buf=[]
                out.append(ch)
            elif ch.isalnum():
                buf.append(ch)
            else:
                if buf: out.append("".join(buf)); buf=[]
        if buf: out.append("".join(buf))
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
        if any(tok in hay for tok in q_tokens):
            if meta.get("image"):
                out.append(meta["image"])
            if isinstance(meta.get("images"), list):
                out.extend([x for x in meta["images"] if isinstance(x, str)])
        if len(out) >= max_n:
            break
    seen, uniq = set(), []
    for u in out:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq[:max_n]


class LocalNaiveRAG(RAGProvider):
    def __init__(self, knowledge_dir: str):
        self.chunks = _load_documents_from_dir(knowledge_dir)
        self.N = len(self.chunks)
        self.df = {}
        for c in self.chunks:
            tokens = set(self._tokenize(c))
            for t in tokens:
                self.df[t] = self.df.get(t, 0) + 1

    def _tokenize(self, s: str) -> List[str]:
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
        def score(text: str) -> float:
            tokens = self._tokenize(text)
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q_tokens:
                idf = math.log((self.N + 1) / (1 + self.df.get(t, 0) )) + 1.0
                s += tf.get(t, 0) * idf
            overlap = len(set(query) & set(text)) / (len(set(query)) + 1e-6)
            return s + overlap
        ranked = sorted(self.chunks, key=score, reverse=True)
        return ranked[:k]

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
# A) OpenAI TTS（雲端）
class OpenAITTSProvider(TTSProvider):
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing for OpenAI TTS")
        import requests
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
        from piper import PiperVoice
        self.voice = PiperVoice.load(PIPER_MODEL_PATH)

    def synth(self, text: str, sr: int = 16000) -> bytes:
        import wave, struct
        import numpy as np
        pcm = self.voice.synthesize(text, length_scale=1.0)  # 可能回 float32
        if pcm.dtype != np.int16:
            x = np.clip(pcm, -1.0, 1.0)
            pcm16 = (x * 32767.0).astype(np.int16)
        else:
            pcm16 = pcm
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()


# ============= 小工具：產生 beep WAV =============
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


# =============== RAG 上傳與載入 ================
def _reload_rag_provider():
    global rag_provider
    if PROVIDER_RAG == "local":
        rag_provider = LocalNaiveRAG(KNOWLEDGE_DIR)
    elif PROVIDER_RAG == "openai":
        rag_provider = OpenAIRAG(KNOWLEDGE_DIR)
    elif PROVIDER_RAG == "none":
        rag_provider = RAGProvider()

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
        rag_provider = RAGProvider()
    else:
        raise RuntimeError(f"Unknown PROVIDER_RAG: {PROVIDER_RAG}")

    # LLM
    if PROVIDER_LLM == "openai":
        llm_provider = OpenAILLM()
    else:
        raise RuntimeError(f"Unknown PROVIDER_LLM: {PROVIDER_LLM}")

    # TTS
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


# --- 輕量預熱 ---
_warm_ready = False
def _do_warm():
    global _warm_ready
    try:
        try:
            _ = getattr(rag_provider, "retrieve", lambda q, k=1: [])("hi", 1)
        except Exception as e:
            print("[warm] rag:", e)
        try:
            if llm_provider:
                _ = llm_provider.chat("ping", [], lang="zh-TW")
        except Exception as e:
            print("[warm] llm:", e)
        _warm_ready = True
    except Exception as e:
        print("[warm] error:", e)

@app.get("/warm")
def warm(tasks: BackgroundTasks):
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
    device_context: Optional[dict] = None

@app.post("/chat")
def chat(body: ChatIn):
    if llm_provider is None or rag_provider is None:
        raise HTTPException(500, "Providers not loaded")
    try:
        tool_chunks, tool_sources = run_tools(body.text, body.device_context)
        rag_chunks = rag_provider.retrieve(body.text, k=RAG_TOP_K) if hasattr(rag_provider, "retrieve") else []
        chunks = tool_chunks + rag_chunks
        reply  = llm_provider.chat(body.text, chunks, lang=REPLY_LANG)
        images_a = _images_from_tool_sources(tool_sources, max_n=3)
        images_b = _images_from_meta_by_query(body.text, max_n=3)
        images = (images_a + images_b)[:3]
        return {"ok": True, "reply": reply, "ctx_used": len(chunks), "sources": tool_sources, "images": images}
    except Exception as e:
        raise HTTPException(500, f"Chat error: {e}")


# ====== 既有 TTS（一次性回傳整段 WAV） ======
class TTSIn(BaseModel):
    text: str
    sr: Optional[int] = 16000

@app.post("/tts")
def tts(body: TTSIn):
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    sr = int(body.sr or 16000)

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

@app.get("/tts")
def tts_get(text: Optional[str] = "", sr: Optional[int] = 16000):
    t = (text or "").strip() or "系統測試音"
    return tts(TTSIn(text=t, sr=sr))


# ====== 新增：WAV 串流端點（ESP32 直串 I2S 用） ======
# 說明：
# 1) 優先使用 OpenAI SDK 的 with_streaming_response，逐塊產生 WAV bytes → StreamingResponse。
# 2) 若發生錯誤（權限/模型/網路），回退到 tts_provider.synth() 一次性拿到整段 WAV，仍以 chunked 形式回傳。
@app.get("/tts_wav")
def tts_wav(text: str, sr: int = 22050):
    t = (text or "").strip()
    if not t:
        raise HTTPException(400, "empty text")

    # 嘗試：OpenAI streaming（僅當 PROVIDER_TTS=openai 且有 API key）
    if PROVIDER_TTS == "openai" and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI()
            # 有些版本對 sample_rate 嚴格；先嘗試帶 sr，失敗再不帶
            def _stream_req(with_sr=True):
                kwargs = dict(model=OPENAI_TTS_MODEL or "gpt-4o-mini-tts",
                              voice=OPENAI_TTS_VOICE or "alloy",
                              input=t,
                              format="wav")
                if with_sr:
                    kwargs["sample_rate"] = int(sr)
                return client.audio.speech.with_streaming_response.create(**kwargs)

            try:
                resp = _stream_req(with_sr=True)
            except Exception:
                resp = _stream_req(with_sr=False)

            def gen():
                with resp as r:
                    for chunk in r.iter_bytes():
                        yield chunk
            # 不提供 Content-Length（chunked）；ESP32 端 HTTPClient 可邊收邊播
            return StreamingResponse(gen(), media_type="audio/wav",
                                     headers={"Cache-Control": "no-store", "Connection": "close"})
        except Exception as e:
            # 失敗則落回一次性
            pass

    # 回退：一次性 WAV（仍用 StreamingResponse 包裝）
    try:
        if tts_provider is not None:
            data = tts_provider.synth(t, sr=sr)
        else:
            data = make_beep_wav(sr=sr, dur_s=0.6, freq=800.0)
        # 提供 Content-Length 對部分客戶端更穩
        return Response(
            content=data,
            media_type="audio/wav",
            headers={
                "Content-Length": str(len(data)),
                "Cache-Control": "no-store",
                "Connection": "close",
            },
        )
    except Exception as e:
        # 最後保底：回短 beep，避免前端卡死
        beep = make_beep_wav(sr=sr, dur_s=0.5, freq=600.0)
        return Response(
            content=beep,
            media_type="audio/wav",
            headers={"X-TTS-Error": str(e)[:200], "Connection": "close"},
        )


# ====== MP3（一次性）仍保留，給需要 MP3 播放器的客戶端 ======
@app.get("/tts_mp3")
def tts_mp3(text: str = "", voice: str = "alloy"):
    t = (text or "").strip()
    if not t:
        raise HTTPException(400, "empty text")
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY missing")

    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_TTS_MODEL or "gpt-4o-mini-tts",
        "voice": voice or (OPENAI_TTS_VOICE or "alloy"),
        "input": t,
        "format": "mp3",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise HTTPException(500, f"TTS failed: {r.status_code} {r.text[:200]}")
    b = r.content
    return Response(
        content=b,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(b)),
            "Cache-Control": "no-store",
            "Connection": "close",
        },
    )


# =============== RAG 檔案/網址 ========
@app.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    suf = Path(file.filename).suffix.lower()
    if suf not in [".txt", ".md", ".mdx", ".html", ".htm", ".pdf", ".docx"]:
        raise HTTPException(400, "Unsupported file type")
    dst = Path(KNOWLEDGE_DIR) / Path(file.filename).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    dst.write_bytes(data)
    _reload_rag_provider()
    return {"ok": True, "saved": str(dst.name), "chunks": len(getattr(rag_provider, "chunks", []))}

@app.post("/rag/add_url")
async def rag_add_url(url: str = Form(...)):
    text = extract_url_text(url)
    if not text or not text.strip():
        print(f"[rag_add_url] extract failed: {url}")
        raise HTTPException(400, "Failed to extract text from URL (no text)")
    safe = "".join(ch for ch in url if ch.isalnum() or ch in "-_").strip("-_")[:60] or "page"
    dst = Path(KNOWLEDGE_DIR) / f"{safe}.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")

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
    root = Path(KNOWLEDGE_DIR).resolve()
    target = (root / name).resolve()
    if not str(target).startswith(str(root)) or not target.exists() or not target.is_file():
        raise HTTPException(400, "file not found")
    target.unlink()
    _reload_rag_provider()
    return {"ok": True, "deleted": name, "chunks": len(getattr(rag_provider, "chunks", []))}

@app.post("/rag/reindex")
def rag_reindex():
    _reload_rag_provider()
    return {"ok": True, "chunks": len(getattr(rag_provider, "chunks", []))}

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


# ============= 開發啟動（直接 python main.py 時） =============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
