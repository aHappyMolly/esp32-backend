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
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# RAG 知識來源（會讀這個資料夾下的 .txt/.md/.mdx/.html 檔）
KNOWLEDGE_DIR     = os.getenv("KNOWLEDGE_DIR", "./knowledge")
RAG_TOP_K         = int(os.getenv("RAG_TOP_K", "4"))

# 語言偏好（繁體）
LANG_HINT         = os.getenv("LANG_HINT", "zh")     # STT 語言提示
REPLY_LANG        = os.getenv("REPLY_LANG", "zh-TW") # LLM 回覆語言偏好


# ============= FastAPI 初始化與 CORS =============
app = FastAPI(title="ESP32 Voice Backend (Pluggable)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)


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
    """讀取資料夾中的文字文件，做極簡清洗與切塊（按段或固定長度）。"""
    exts = ("*.txt", "*.md", "*.mdx", "*.html")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, ext)))
    chunks = []
    for fp in files:
        try:
            txt = open(fp, "r", encoding="utf-8", errors="ignore").read()
        except:
            continue
        # 粗略切塊：先按段落拆，再限制每塊長度
        for para in txt.splitlines():
            t = para.strip()
            if not t:
                continue
            # 最多 600-800 字一塊
            while len(t) > 800:
                chunks.append(t[:800])
                t = t[800:]
            if t:
                chunks.append(t)
    return chunks

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


# ============= Provider 載入器 =============
stt_provider: Optional[STTProvider] = None
rag_provider: Optional[RAGProvider] = None
llm_provider: Optional[LLMProvider] = None

@app.on_event("startup")
def load_providers():
    global stt_provider, rag_provider, llm_provider

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


# ============= 路由 =============
@app.get("/health")
def health():
    return {
        "ok": True,
        "stt": PROVIDER_STT,
        "rag": PROVIDER_RAG,
        "llm": PROVIDER_LLM,
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

@app.post("/chat")
def chat(body: ChatIn):
    if llm_provider is None or rag_provider is None:
        raise HTTPException(500, "Providers not loaded")
    try:
        chunks = rag_provider.retrieve(body.text, k=RAG_TOP_K) if hasattr(rag_provider, "retrieve") else []
        reply  = llm_provider.chat(body.text, chunks, lang=REPLY_LANG)
        return {"ok": True, "reply": reply, "ctx_used": len(chunks)}
    except Exception as e:
        raise HTTPException(500, f"Chat error: {e}")


# ============= 開發啟動（直接 python main.py 時） =============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
