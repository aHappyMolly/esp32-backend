# /app/ui/rag_admin.py

from fastapi.responses import HTMLResponse

# ---- 管理子頁（/rag） ----
_RAG_ADMIN_HTML = r"""<!doctype html>
<html lang="zh-TW"><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RAG 管理</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:24px;line-height:1.5}
  .card{max-width:920px;margin:auto;padding:16px;border:1px solid #ddd;border-radius:12px}
  h2{margin-top:0} input,button{font-size:15px;padding:8px} .mono{font-family:ui-monospace,Menlo,Consolas,monospace}
  table{width:100%;border-collapse:collapse;margin-top:12px} th,td{border-bottom:1px solid #eee;padding:6px 8px;text-align:left}
  .muted{opacity:.7;font-size:13px} .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .btn{padding:6px 10px;border:1px solid #ccc;border-radius:8px;background:#fafafa;cursor:pointer}
  .btn.danger{border-color:#f66;background:#fee}
  .chip{padding:2px 8px;border:1px solid #eee;border-radius:999px;font-size:12px}
</style>
<div class="card">
  <h2>RAG 管理</h2>
  <p><a href="http://esp32s3.local/">← 回首頁</a></p>
  <p class="muted">※ 上傳與貼網址只會寫入後端檔案系統，<b>記得完成後自行 push 到 GitHub / 重新部署 Render</b>。</p>

  <h3>上傳檔案（.txt .md .html .pdf .docx）</h3>
  <div class="row">
    <input id="file" type="file" />
    <button class="btn" onclick="upload()">上傳</button>
  </div>

  <h3>加入網址（會擷取正文並存為 .txt）</h3>
  <div class="row">
    <input id="url" type="url" placeholder="https://example.com/article" style="flex:1;min-width:280px"/>
    <button class="btn" onclick="addUrl()">加入</button>
  </div>

  <div class="row" style="margin-top:8px">
    <button class="btn" onclick="reindex()">重建索引</button>
    <button class="btn" onclick="backup()">下載備份</button>
    <span id="status" class="muted"></span>
  </div>

  <div style="border-top:1px solid ; margin-top:20px; padding-top:16px">
    <h3 style="margin-top:18px">目前檔案</h3>
    <div id="tableWrap"></div>
  </div>

</div>

<script>
const el = id => document.getElementById(id);
function fmtBytes(n){ if(n<1024) return n+' B'; if(n<1048576) return (n/1024).toFixed(1)+' KB'; if(n<1073741824) return (n/1048576).toFixed(1)+' MB'; return (n/1073741824).toFixed(1)+' GB'; }
function fmtTime(s){ const d=new Date(s*1000); return d.toLocaleString(); }

// 若沒設定，給預設值（改成你的後端 Base URL）
window.API_BASE = window.API_BASE || "https://esp32-backend-o4fc.onrender.com";

async function upload(){
  const f = el('file').files[0];
  if(!f){ alert('請先選擇檔案'); return; }
  const fd = new FormData(); fd.append('file', f, f.name);
  el('status').textContent = '上傳中…';
  const r = await fetch('/rag/upload', {method:'POST', body: fd});
  const j = await r.json();
  el('status').textContent = j.ok ? '✅ 上傳完成並已重建索引' : ('❌ '+(j.detail||JSON.stringify(j)));
  if(j.ok) load();
}
async function addUrl(){
  const u = (el('url').value||'').trim();
  if(!u){ alert('請先輸入網址'); return; }
  const fd = new FormData(); fd.append('url', u);
  el('status').textContent = '抓取中…';
  const r = await fetch('/rag/add_url', {method:'POST', body: fd});
  const j = await r.json();
  el('status').textContent = j.ok ? ('✅ 已儲存 '+j.saved+' 並重建索引') : ('❌ '+(j.detail||JSON.stringify(j)));
  if(j.ok) { el('url').value=''; load(); }
}
async function reindex(){
  el('status').textContent = '重建索引中…';
  const r = await fetch('/rag/reindex', {method:'POST'});
  const j = await r.json();
  el('status').textContent = j.ok ? ('✅ 重新索引完成，chunks='+j.chunks) : ('❌ '+(j.detail||JSON.stringify(j)));
}
async function backup(){
  window.open('/rag/backup', '_blank');  // 直接打開新分頁觸發下載
}
async function del(name){
  if(!confirm('確定刪除 '+name+' ?')) return;
  const fd = new FormData(); fd.append('name', name);
  const r = await fetch('/rag/delete', {method:'POST', body: fd});
  const j = await r.json();
  el('status').textContent = j.ok ? ('🗑️ 已刪除 '+name) : ('❌ '+(j.detail||JSON.stringify(j)));
  if(j.ok) load();
}
function renderTable(list){
  if(!list.length){ el('tableWrap').innerHTML = '<p class="muted">（目前沒有檔案）</p>'; return; }
  let html = '<table><thead><tr><th>檔名</th><th>大小</th><th>更新時間</th><th></th></tr></thead><tbody>';
  for(const it of list){
    html += `<tr>
      <td><span class="chip">${it.ext||''}</span> ${it.name}</td>
      <td>${fmtBytes(it.size||0)}</td>
      <td>${fmtTime(it.mtime||0)}</td>
      <td><button class="btn danger" onclick="del('${it.name}')">刪除</button></td>
    </tr>`;
  }
  html += '</tbody></table>';
  el('tableWrap').innerHTML = html;
}
async function load(){
  el('status').textContent = '載入中…';
  const r = await fetch('/rag/list'); const j = await r.json();
  if(j.ok){ renderTable(j.files||[]); el('status').textContent=''; }
  else { el('status').textContent='❌ 無法載入清單'; }
}
load();
</script>
</html>
"""

def mount_rag_admin(app):
    @app.get("/rag", response_class=HTMLResponse)
    def rag_admin_page():
        return HTMLResponse(content=_RAG_ADMIN_HTML, status_code=200)
