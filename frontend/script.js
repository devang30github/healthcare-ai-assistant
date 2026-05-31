const API = '';   // same origin — FastAPI serves this file
let sessionStats = { count: 0, intent: [], retrieval: [], generation: [] };

// ── Tab switching ──────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', (i === 0 && name === 'chat') || (i === 1 && name === 'admin'));
  });
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(name + '-panel').classList.add('active');
  if (name === 'admin') fetchHealth();
}

// ── Health check ───────────────────────────────────────
async function fetchHealth() {
  try {
    const r    = await fetch(`${API}/health`);
    const data = await r.json();

    const dot   = document.getElementById('dot');
    const label = document.getElementById('status-label');
    dot.className   = 'dot online';
    label.textContent = 'Online';

    document.getElementById('h-status').textContent = data.status === 'ok' ? 'OK' : 'Error';
    document.getElementById('h-status').className   = 'hi-value ' + (data.status === 'ok' ? 'ok' : 'error');
    document.getElementById('h-vs').textContent     = data.vector_store_loaded ? 'Loaded' : 'Not loaded';
    document.getElementById('h-vs').className       = 'hi-value ' + (data.vector_store_loaded ? 'ok' : 'error');
    document.getElementById('h-env').textContent    = data.environment;
    document.getElementById('h-time').textContent   = new Date().toLocaleTimeString();
  } catch {
    document.getElementById('dot').className       = 'dot offline';
    document.getElementById('status-label').textContent = 'Offline';
  }
}

// ── Ingest ─────────────────────────────────────────────
async function runIngest() {
  const btn = document.getElementById('ingest-btn');
  const res = document.getElementById('ingest-result');
  btn.disabled = true;
  btn.textContent = 'Ingesting...';
  res.style.display = 'none';

  try {
    const r    = await fetch(`${API}/ingest`, { method: 'POST' });
    const data = await r.json();
    res.style.display = 'block';
    if (r.ok) {
      res.className   = 'success';
      res.textContent = `Done — ${data.documents_processed} documents, ${data.chunks_created} chunks created.`;
      fetchHealth();
    } else {
      res.className   = 'error';
      res.textContent = data.detail || 'Ingestion failed.';
    }
  } catch (e) {
    res.style.display = 'block';
    res.className     = 'error';
    res.textContent   = 'Could not reach the API.';
  } finally {
    btn.disabled    = false;
    btn.innerHTML   = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg> Run Ingestion`;
  }
}

// ── Clear DB ───────────────────────────────────────────
async function clearDB() {
  if (!confirm('Clear the vector store? You will need to re-ingest.')) return;
  const btn = document.getElementById('clear-btn');
  const res = document.getElementById('clear-result');
  btn.disabled = true;

  try {
    const r    = await fetch(`${API}/clear-db`, { method: 'POST' });
    const data = await r.json();
    res.style.display = 'block';
    if (r.ok) {
      res.className   = 'success';
      res.textContent = data.message || 'Vector store cleared.';
      fetchHealth();
    } else {
      res.className   = 'error';
      res.textContent = data.detail || 'Clear failed.';
    }
  } catch {
    res.style.display = 'block';
    res.className     = 'error';
    res.textContent   = 'Could not reach the API.';
  } finally {
    btn.disabled = false;
  }
}

// ── Chat ───────────────────────────────────────────────
function fillQuestion(el) {
  const inp = document.getElementById('question-input');
  inp.value = el.textContent;
  autoResize(inp);
  inp.focus();
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

async function sendQuestion() {
const inp      = document.getElementById('question-input');
const question = inp.value.trim();
if (!question) return;

const provider = document.getElementById('provider-select').value;
const sendBtn  = document.getElementById('send-btn');

inp.value = '';
inp.style.height = 'auto';
sendBtn.disabled = true;

document.getElementById('empty-state')?.remove();
appendMessage('user', question);

const msgId    = 'msg-' + Date.now();
const bubbleId = 'bubble-' + Date.now();
const metaId   = 'meta-' + Date.now();
const srcId    = 'src-' + Date.now();
appendStreamingMessage(msgId, bubbleId, metaId, srcId);

let fullText        = '';
let pendingMetadata = null;   // store metadata, render only after streaming ends

try {
  const resp = await fetch(`${API}/ask/stream`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ question, provider }),
  });

  if (!resp.ok) {
    const err = await resp.json();
    document.getElementById(bubbleId).textContent = `Error: ${err.detail || 'Something went wrong.'}`;
    return;
  }

  const reader  = resp.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop();

    for (const part of parts) {
      if (!part.trim()) continue;

      const lines     = part.split('\n');
      const eventLine = lines.find(l => l.startsWith('event:'));
      const dataLine  = lines.find(l => l.startsWith('data:'));
      if (!eventLine || !dataLine) continue;

      const eventType = eventLine.replace('event:', '').trim();
      let data;
      try { data = JSON.parse(dataLine.replace('data:', '').trim()); }
      catch { continue; }

      if (eventType === 'metadata') {
        pendingMetadata = data;                          // store, don't render yet
      } else if (eventType === 'token') {
        fullText += data.text;
        document.getElementById(bubbleId).innerHTML = escHtml(fullText);
        document.getElementById('messages').scrollTop = 999999;
      } else if (eventType === 'done') {
        if (pendingMetadata) {
          renderStreamMetadata(metaId, srcId, pendingMetadata);
        }
        updateMetricsFromStream(data, metaId, pendingMetadata);  // pass before nulling
        pendingMetadata = null;
      } else if (eventType === 'error') {
        document.getElementById(bubbleId).textContent = `Error: ${data.message}`;
      }
    }  // ← for loop closes here
  }    // ← while loop closes here

} catch (e) {
  document.getElementById(bubbleId).textContent = 'Could not reach the API. Make sure the server is running.';
} finally {
  sendBtn.disabled = false;
  inp.focus();
}
}


function appendStreamingMessage(msgId, bubbleId, metaId, srcId) {
  const msgs = document.getElementById('messages');
  const div  = document.createElement('div');
  div.className = 'msg assistant';
  div.id = msgId;
  div.innerHTML = `
    <div class="msg-avatar">
      <svg viewBox="0 0 24 24" fill="none" stroke="#6b6b65" stroke-width="2" width="14" height="14">
        <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
      </svg>
    </div>
    <div class="msg-body">
      <div class="msg-bubble" id="${bubbleId}">
        <span class="typing-indicator" style="display:inline-flex">
          <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
        </span>
      </div>
      <div class="msg-meta" id="${metaId}"></div>
      <div class="sources-box" id="${srcId}"></div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function renderStreamMetadata(metaId, srcId, data) {
  const metaEl = document.getElementById(metaId);
  const confClass = `confidence-${data.confidence}`;
  const confLabel = data.confidence.charAt(0).toUpperCase() + data.confidence.slice(1);
  const toolLabel = data.tool_used === 'appointment_tool' ? 'Appointment' : 'RAG Search';

  metaEl.innerHTML = `
    <span class="badge ${confClass}">${confLabel} confidence</span>
    <span class="badge tool">${toolLabel}</span>
    <span class="latency" id="lat-${metaId}">...</span>`;

  // Render sources
  if (data.sources && data.sources.length > 0) {
    const srcEl  = document.getElementById(srcId);
    const btnId  = 'sbtn-' + metaId;
    const items  = data.sources.map(s => `
      <div class="source-item">
        <div class="source-doc">${escHtml(s.document)} <span style="color:var(--text-ter);font-weight:400">· chunk #${s.chunk_id}</span></div>
        <div class="source-text">${escHtml(s.chunk)}</div>
      </div>`).join('');

    metaEl.innerHTML += `<button class="sources-toggle" id="${btnId}" onclick="toggleSources('${srcId}')">
      ${data.sources.length} source${data.sources.length > 1 ? 's' : ''}</button>`;
    srcEl.innerHTML = items;
  }
}

function toggleSources(srcId) {
  const box = document.getElementById(srcId);
  if (box) box.classList.toggle('open');
}

function updateMetricsFromStream(doneData, metaId, meta) {
  // Update latency badge in the message
  const latEl = document.getElementById('lat-' + metaId);
  if (latEl) {
    const intentMs   = meta ? meta.intent_ms    : 0;
    const retrievalMs = meta ? meta.retrieval_ms : 0;
    latEl.title       = `intent: ${intentMs}ms | retrieval: ${retrievalMs}ms | generation: ${doneData.generation_ms}ms`;
    latEl.textContent = `total: ${doneData.total_ms}ms`;
  }

  // Update session stats — all three dimensions
  sessionStats.count++;
  if (meta) {
    sessionStats.intent.push(meta.intent_ms);
    sessionStats.retrieval.push(meta.retrieval_ms);
  }
  sessionStats.generation.push(doneData.generation_ms);

  const avg = arr => arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : 0;
  document.getElementById('m-count').textContent   = sessionStats.count;
  document.getElementById('m-avg-int').textContent = avg(sessionStats.intent);
  document.getElementById('m-avg-ret').textContent = avg(sessionStats.retrieval);
  document.getElementById('m-avg-gen').textContent = avg(sessionStats.generation);
}

function appendMessage(role, text) {
  const msgs = document.getElementById('messages');
  const div  = document.createElement('div');
  div.className = `msg ${role}`;

  const avatarSVG = role === 'user'
    ? `<svg viewBox="0 0 24 24" fill="white" width="14" height="14"><path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/></svg>`
    : `<svg viewBox="0 0 24 24" fill="none" stroke="#6b6b65" stroke-width="2" width="14" height="14"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>`;

  div.innerHTML = `
    <div class="msg-avatar">${avatarSVG}</div>
    <div class="msg-body">
      <div class="msg-bubble">${escHtml(text)}</div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function appendAssistantMessage(data) {
  const msgs = document.getElementById('messages');
  const div  = document.createElement('div');
  div.className = 'msg assistant';

  const confClass = `confidence-${data.confidence}`;
  const confLabel = data.confidence.charAt(0).toUpperCase() + data.confidence.slice(1);
  const toolLabel = data.tool_used === 'appointment_tool' ? 'Appointment' : 'RAG Search';

  let sourcesHtml = '';
  if (data.sources && data.sources.length > 0) {
    const id = 'src-' + Date.now();
    const items = data.sources.map(s => `
      <div class="source-item">
        <div class="source-doc">${escHtml(s.document)} <span style="color:var(--text-ter);font-weight:400">· chunk #${s.chunk_id}</span></div>
        <div class="source-text">${escHtml(s.chunk)}</div>
      </div>`).join('');
    sourcesHtml = `
      <button class="sources-toggle" onclick="toggleSources('${id}')">
        ${data.sources.length} source${data.sources.length > 1 ? 's' : ''}
      </button>
      <div class="sources-box" id="${id}">${items}</div>`;
  }

  div.innerHTML = `
    <div class="msg-avatar">
      <svg viewBox="0 0 24 24" fill="none" stroke="#6b6b65" stroke-width="2" width="14" height="14">
        <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
      </svg>
    </div>
    <div class="msg-body">
      <div class="msg-bubble">${escHtml(data.answer)}</div>
      <div class="msg-meta">
        <span class="badge ${confClass}">${confLabel} confidence</span>
        <span class="badge tool">${toolLabel}</span>
        <span class="latency" title="intent: ${data.intent_ms}ms | retrieval: ${data.retrieval_time_ms}ms | generation: ${data.generation_time_ms}ms">total: ${data.total_time_ms}ms</span>
      </div>
      ${sourcesHtml}
    </div>`;

  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

// toggleSources defined above — single unified version

// typing indicator
function appendTyping() {
  const msgs = document.getElementById('messages');
  const id   = 'typing-' + Date.now();
  const div  = document.createElement('div');
  div.className = 'msg assistant';
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">
      <svg viewBox="0 0 24 24" fill="none" stroke="#6b6b65" stroke-width="2" width="14" height="14">
        <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
      </svg>
    </div>
    <div class="msg-body">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return id;
}

function removeTyping(id) {
  document.getElementById(id)?.remove();
}

// ── Metrics ────────────────────────────────────────────
function updateMetrics(data) {
  sessionStats.count++;
  sessionStats.intent.push(data.intent_ms);
  sessionStats.retrieval.push(data.retrieval_time_ms);
  sessionStats.generation.push(data.generation_time_ms);

  const avg = arr => arr.length ? Math.round(arr.reduce((a,b)=>a+b,0)/arr.length) : 0;

  document.getElementById("m-count").textContent    = sessionStats.count;
  document.getElementById("m-avg-int").textContent  = avg(sessionStats.intent);
  document.getElementById('m-avg-ret').textContent = avg(sessionStats.retrieval);
  document.getElementById('m-avg-gen').textContent = avg(sessionStats.generation);
}

// ── Helpers ────────────────────────────────────────────
function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}


// ── File Upload ────────────────────────────────────────
let selectedFiles = [];

function handleFileSelect(files) {
  selectedFiles = Array.from(files);
  renderFileList();
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').style.borderColor = 'var(--border)';
  selectedFiles = Array.from(e.dataTransfer.files).filter(f =>
    ['.pdf','.txt','.docx'].some(ext => f.name.toLowerCase().endsWith(ext))
  );
  renderFileList();
}

function renderFileList() {
  const list = document.getElementById('file-list');
  const btn  = document.getElementById('upload-btn');
  list.innerHTML = selectedFiles.map(f => `
    <div style="display:flex;align-items:center;gap:8px;background:var(--gray-light);border-radius:var(--radius-xs);padding:7px 10px;font-size:12px;">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      <span style="flex:1;color:var(--text-primary)">${escHtml(f.name)}</span>
      <span style="color:var(--text-ter)">${(f.size/1024).toFixed(1)} KB</span>
    </div>`).join('');
  btn.disabled = selectedFiles.length === 0;
}

async function uploadFiles() {
  if (!selectedFiles.length) return;
  const btn = document.getElementById('upload-btn');
  const res = document.getElementById('upload-result');
  btn.disabled = true;
  btn.textContent = 'Uploading...';
  res.style.display = 'none';

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append('files', f));

  try {
    const r    = await fetch(`${API}/upload`, { method: 'POST', body: formData });
    const data = await r.json();
    res.style.display = 'block';
    if (r.ok) {
      res.className   = 'success';
      res.style.background = 'var(--green-light)';
      res.style.color = 'var(--green)';
      res.textContent = `Uploaded: ${data.saved.join(', ')}. ${data.rejected.length ? 'Rejected: ' + data.rejected.join(', ') + '.' : ''} Now run ingestion.`;
      selectedFiles = [];
      document.getElementById('file-list').innerHTML = '';
    } else {
      res.className   = 'error';
      res.style.background = 'var(--red-light)';
      res.style.color = 'var(--red)';
      res.textContent = data.detail || 'Upload failed.';
    }
  } catch {
    res.style.display = 'block';
    res.style.background = 'var(--red-light)';
    res.style.color = 'var(--red)';
    res.textContent = 'Could not reach the API.';
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" width="14" height="14"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg> Upload Files`;
  }
}

// ── Init ───────────────────────────────────────────────
fetchHealth();