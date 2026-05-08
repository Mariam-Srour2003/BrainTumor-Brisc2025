/* BRISC 2025 — Brain Tumor Analysis Dashboard */
'use strict';

// ─── UTILITIES ────────────────────────────────────────────────────────────────

function base() {
  return document.getElementById('apiBase').value.replace(/\/$/, '');
}

async function apiFetch(path, opts = {}) {
  const url = base() + path;
  try {
    const res = await fetch(url, opts);
    const text = await res.text();
    try { return { ok: res.ok, status: res.status, data: JSON.parse(text) }; }
    catch { return { ok: false, status: res.status, data: { error: text } }; }
  } catch (err) {
    return { ok: false, status: 0, data: { error: err.message } };
  }
}

function toast(msg, type = 'info') {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

function show(el) { if (el) el.style.display = ''; }
function hide(el) { if (el) el.style.display = 'none'; }
function addClass(el, cls) { el && el.classList.add(cls); }
function removeClass(el, cls) { el && el.classList.remove(cls); }

function setResult(elId, data, asJson = false) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.textContent = asJson ? JSON.stringify(data, null, 2) : String(data);
  el.classList.add('visible');
}

function spinner(id, on) {
  const el = document.getElementById(id);
  if (!el) return;
  if (on) el.classList.remove('hidden'); else el.classList.add('hidden');
}

function status(id, on) {
  const el = document.getElementById(id);
  if (!el) return;
  if (on) el.classList.add('visible'); else el.classList.remove('visible');
}

function pct(v) { return (v * 100).toFixed(1) + '%'; }
function fmt(v) { return (typeof v === 'number') ? v.toFixed(4) : (v ?? '—'); }

// ─── TAB SWITCHING ────────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ─── MODEL REGISTRY ───────────────────────────────────────────────────────────

let registeredModels = [];

function chipClass(name) {
  if (name.startsWith('classification.')) return 'classification';
  if (name.startsWith('segmentation.'))   return 'segmentation';
  if (name.startsWith('hybrid.'))         return 'hybrid';
  return '';
}

function populateSelects(models) {
  const selects = ['trainModelSelect', 'predictModelSelect', 'evalModelSelect', 'xaiModelSelect'];
  selects.forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const cur = sel.value;
    // keep the "auto" placeholder for predict/xai selects
    const keepFirst = (id === 'predictModelSelect' || id === 'xaiModelSelect');
    sel.innerHTML = keepFirst ? '<option value="">— Auto: latest checkpoint —</option>' : '<option value="">— Select model —</option>';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      sel.appendChild(opt);
    });
    if (cur) sel.value = cur;
  });

  ['btnTrain', 'btnTrainAll', 'btnTune', 'btnEvaluate'].forEach(id => {
    const b = document.getElementById(id);
    if (b) b.disabled = models.length === 0;
  });
}

async function loadModels() {
  const { ok, data } = await apiFetch('/models/list');
  if (!ok || !data.models) return;
  registeredModels = data.models;

  document.getElementById('statModels').textContent = registeredModels.length;

  const chips = document.getElementById('modelChips');
  chips.innerHTML = '';
  registeredModels.forEach(m => {
    const d = document.createElement('span');
    d.className = 'model-chip ' + chipClass(m);
    d.textContent = m;
    chips.appendChild(d);
  });

  populateSelects(registeredModels);
}

// ─── HEALTH / CONNECT ─────────────────────────────────────────────────────────

async function connect() {
  const dot = document.getElementById('connDot');
  const health = document.getElementById('statHealth');
  const sub = document.getElementById('statHealthSub');

  dot.className = 'conn-dot';
  const { ok, data } = await apiFetch('/health');
  if (ok && data.status === 'ok') {
    dot.className = 'conn-dot ok';
    dot.title = 'Connected';
    health.textContent = '✓ Online';
    health.style.color = 'var(--success)';
    sub.textContent = base();
    toast('Connected to ML service', 'success');
    await loadModels();
    ['btnPredict', 'btnExplain'].forEach(id => {
      const b = document.getElementById(id); if (b) b.disabled = false;
    });
  } else {
    dot.className = 'conn-dot err';
    dot.title = 'Connection failed';
    health.textContent = '✗ Offline';
    health.style.color = 'var(--error)';
    sub.textContent = data.error || 'Cannot reach API';
    toast('Cannot reach API — is the server running?', 'error');
  }
}

document.getElementById('btnConnect').addEventListener('click', connect);

// ─── DASHBOARD ACTIONS ────────────────────────────────────────────────────────

document.getElementById('dashBtnPrepare').addEventListener('click', async () => {
  setResult('dashOutput', 'Preparing dataset…');
  const { data } = await apiFetch('/train/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'prepare' }),
  });
  setResult('dashOutput', data, true);
  if (data.status === 'completed') toast(`Dataset ready: ${data.processed_count} processed`, 'success');
  else toast('Prepare failed: ' + (data.message || 'unknown error'), 'error');
});

document.getElementById('dashBtnRefresh').addEventListener('click', async () => {
  await loadModels();
  toast('Models refreshed', 'info');
});

// ─── TRAIN ────────────────────────────────────────────────────────────────────

document.getElementById('btnPrepare').addEventListener('click', async () => {
  spinner('prepareSpinner', true);
  setResult('prepareResult', 'Preparing dataset…');
  const { data } = await apiFetch('/train/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'prepare' }),
  });
  spinner('prepareSpinner', false);
  setResult('prepareResult', data, true);
  if (data.status === 'completed')
    toast(`Dataset prepared — ${data.processed_count} images processed, ${data.skipped_count} skipped`, 'success');
  else toast('Prepare failed: ' + (data.message || JSON.stringify(data)), 'error');
});

// ─── LOSS CHART ──────────────────────────────────────────────────────────────

const lossChart = (() => {
  let trainPts = [], valPts = [], sse = null;

  function draw() {
    const canvas = document.getElementById('lossChart');
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth, H = canvas.offsetHeight;
    if (!W || !H) return;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const pad = { top: 8, right: 12, bottom: 28, left: 44 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    const all = [...trainPts, ...valPts];
    if (all.length === 0) return;

    const maxX = Math.max(...all.map(p => p.x));
    const minY = Math.min(...all.map(p => p.y));
    const maxY = Math.max(...all.map(p => p.y));
    const rangeY = maxY - minY || 1;

    function sx(x) { return pad.left + (x / Math.max(maxX, 1)) * cw; }
    function sy(y) { return pad.top + ch - ((y - minY) / rangeY) * ch; }

    // Grid lines
    ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (ch / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
      const val = maxY - (rangeY / 4) * i;
      ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(3), pad.left - 4, y + 3);
    }

    // X axis labels
    ctx.textAlign = 'center'; ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui';
    const ticks = Math.min(maxX, 6);
    for (let i = 0; i <= ticks; i++) {
      const x = Math.round(i * maxX / ticks);
      ctx.fillText(x, sx(x), pad.top + ch + 16);
    }

    function drawLine(pts, color) {
      if (pts.length < 2) return;
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
      pts.forEach((p, i) => { if (i === 0) ctx.moveTo(sx(p.x), sy(p.y)); else ctx.lineTo(sx(p.x), sy(p.y)); });
      ctx.stroke();
      // Dot at last point
      const last = pts[pts.length - 1];
      ctx.beginPath(); ctx.fillStyle = color; ctx.arc(sx(last.x), sy(last.y), 3, 0, Math.PI * 2); ctx.fill();
    }

    drawLine(trainPts, '#388bfd');
    drawLine(valPts, '#3fb950');
  }

  function reset() {
    trainPts = []; valPts = [];
    if (sse) { sse.close(); sse = null; }
    hide(document.getElementById('lossChartWrap'));
    document.getElementById('lossChartEpoch').textContent = 'Epoch 0 / —';
  }

  function addEpoch(evt) {
    trainPts.push({ x: evt.epoch, y: evt.train_loss });
    valPts.push({ x: evt.epoch, y: evt.val_loss });
    document.getElementById('lossChartEpoch').textContent = `Epoch ${evt.epoch} / ${evt.total || '?'}`;
    show(document.getElementById('lossChartWrap'));
    draw();
  }

  function startSSE(modelName, apiBase, { onDone, onError } = {}) {
    reset();
    const url = apiBase.replace(/\/$/, '') + '/train/stream/' + encodeURIComponent(modelName);
    sse = new EventSource(url);
    sse.onmessage = ev => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === 'epoch') {
          addEpoch(data);
        } else if (data.type === 'done' || data.type === 'complete') {
          sse.close(); sse = null;
          if (onDone) onDone(data);
        } else if (data.type === 'error') {
          sse.close(); sse = null;
          if (onError) onError(data.message || 'Training error');
        }
      } catch (_) {}
    };
    sse.onerror = () => {
      if (sse) { sse.close(); sse = null; }
      if (onError) onError('SSE connection lost');
    };
  }

  window.addEventListener('resize', draw);
  return { startSSE, reset, addEpoch };
})();

document.getElementById('btnTrain').addEventListener('click', () => {
  const model = document.getElementById('trainModelSelect').value;
  if (!model) { toast('Select a model first', 'error'); return; }
  spinner('trainSpinner', true);
  status('trainStatus', true);
  document.getElementById('trainStatusText').textContent = `Training ${model}… (pre-loading images into RAM, then epochs will start)`;
  setResult('trainResult', `Starting ${model} — loading images into RAM (one-time, ~30s), then training begins…`);

  lossChart.startSSE(model, base(), {
    onDone: async (evt) => {
      spinner('trainSpinner', false);
      status('trainStatus', false);
      setResult('trainResult', evt, true);
      toast(`${model} trained successfully`, 'success');
      await loadModels();
    },
    onError: (msg) => {
      spinner('trainSpinner', false);
      status('trainStatus', false);
      setResult('trainResult', { error: msg });
      toast('Training failed: ' + msg, 'error');
    },
  });
  // The SSE endpoint runs training internally — no separate POST needed.
});

document.getElementById('btnTrainAll').addEventListener('click', async () => {
  spinner('trainSpinner', true);
  status('trainStatus', true);
  document.getElementById('trainStatusText').textContent = 'Training all models — this will take a while…';
  setResult('trainResult', 'Training all registered models…');

  const { data } = await apiFetch('/train/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'train_all' }),
  });
  spinner('trainSpinner', false);
  status('trainStatus', false);
  setResult('trainResult', data, true);
  toast('All model training complete', 'success');
  await loadModels();
});

document.getElementById('btnTune').addEventListener('click', async () => {
  const trials = parseInt(document.getElementById('tuneTrials').value) || 5;
  spinner('tuneSpinner', true);
  setResult('tuneResult', `Running ${trials} hyperparameter tuning trials…`);

  const { data } = await apiFetch('/train/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'tune', trials }),
  });
  spinner('tuneSpinner', false);
  setResult('tuneResult', data, true);
  if (data.status === 'completed')
    toast(`Tuning done — best ${data.metric_name}: ${fmt(data.best_metric)}`, 'success');
  else toast('Tuning failed: ' + (data.message || ''), 'error');
});

// ─── PREDICT — UPLOAD ZONE ────────────────────────────────────────────────────

function setupUploadZone(zoneId, inputId, previewId, previewImgId, onFile) {
  const zone  = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const wrap  = document.getElementById(previewId);
  const img   = document.getElementById(previewImgId);

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  input.addEventListener('change', () => { if (input.files[0]) handleFile(input.files[0]); });

  function handleFile(f) {
    const reader = new FileReader();
    reader.onload = ev => { img.src = ev.target.result; wrap.classList.add('visible'); };
    reader.readAsDataURL(f);
    if (onFile) onFile(f);
  }
}

setupUploadZone('predictUploadZone', 'predictFile', 'predictPreview', 'predictPreviewImg', () => {
  document.getElementById('btnPredict').disabled = false;
});
setupUploadZone('xaiUploadZone', 'xaiFile', 'xaiPreview', 'xaiPreviewImg', () => {
  document.getElementById('btnExplain').disabled = false;
});

document.getElementById('explainChk').addEventListener('change', function () {
  document.getElementById('predictXaiOpts').style.display = this.checked ? '' : 'none';
});

// ─── RENDER HELPERS ───────────────────────────────────────────────────────────

const CLASS_COLORS = {
  glioma:     { badge: 'cls-glioma',     bar: 'bar-glioma' },
  meningioma: { badge: 'cls-meningioma', bar: 'bar-meningioma' },
  no_tumor:   { badge: 'cls-no_tumor',   bar: 'bar-no_tumor' },
  pituitary:  { badge: 'cls-pituitary',  bar: 'bar-pituitary' },
};

function classColor(name) {
  return CLASS_COLORS[name] || { badge: '', bar: 'bar-default' };
}

function renderClassification(data, container) {
  const probs = data.probabilities || {};
  const confidence = typeof data.confidence === 'number' ? data.confidence : null;
  const name = data.class_name || data.class || data.predicted_class || 'Unknown';
  const col = classColor(name);

  let html = `
    <div class="cls-result">
      <span class="cls-badge ${col.badge}">${name.replace('_', ' ').toUpperCase()}</span>
      ${confidence !== null ? `<div class="cls-confidence">Confidence: <strong>${(confidence * 100).toFixed(2)}%</strong></div>` : ''}
  `;

  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  if (entries.length > 0) {
    html += '<div class="prob-bars">';
    entries.forEach(([cls, p]) => {
      const c = classColor(cls);
      html += `
        <div class="prob-row">
          <div class="prob-label">${cls.replace('_', ' ')}</div>
          <div class="prob-bar-wrap">
            <div class="prob-bar ${c.bar}" style="width:${(p * 100).toFixed(1)}%"></div>
          </div>
          <div class="prob-pct">${(p * 100).toFixed(1)}%</div>
        </div>
      `;
    });
    html += '</div>';
  }
  html += '</div>';
  container.innerHTML = html;
}

function imgCard(label, src) {
  return `
    <div class="seg-img-card">
      <div class="seg-img-label">${label}</div>
      <img src="${src}" alt="${label}" loading="lazy" />
    </div>
  `;
}

function renderSegmentation(data, originalSrc, container) {
  let html = '<div style="margin-bottom:10px"><span class="tag tag-seg">SEGMENTATION</span></div>';
  html += '<div class="seg-images">';
  if (originalSrc) html += imgCard('Original', originalSrc);
  if (data.segmentation_overlay_base64) html += imgCard('Tumor Overlay', data.segmentation_overlay_base64);
  else if (data.segmentation_overlay_path) html += imgCard('Tumor Overlay', data.segmentation_overlay_path);
  if (data.segmentation_mask_base64) html += imgCard('Binary Mask', data.segmentation_mask_base64);
  else if (data.segmentation_mask_path) html += imgCard('Binary Mask', data.segmentation_mask_path);
  html += '</div>';
  container.innerHTML = html;
}

function renderExplanation(explanation, container) {
  if (!explanation) return;

  // Missing optional dependency (shap / lime not installed)
  if (explanation.metadata && explanation.metadata.missing_dependency) {
    const div = document.createElement('div');
    div.style.cssText = 'color:var(--warn);padding:10px;font-size:13px';
    div.textContent = `⚠ ${explanation.metadata.error || 'Optional dependency not installed for ' + explanation.method}`;
    container.appendChild(div);
    return;
  }

  // Handle nested joint explanations (dict of classification + segmentation)
  if (explanation.task === 'joint' || (explanation.classification && explanation.segmentation)) {
    const clsPart = explanation.classification || explanation;
    const segPart = explanation.segmentation;
    renderExplanation(clsPart, container);
    if (segPart) {
      const div = document.createElement('div');
      div.style.marginTop = '16px';
      renderExplanation(segPart, div);
      container.appendChild(div);
    }
    return;
  }

  const method = (explanation.method || 'gradcam').toUpperCase();
  const task   = explanation.task || 'classification';

  let html = `<hr class="divider" /><div style="margin-bottom:10px">
    <strong>XAI — ${method}</strong>
    <span class="tag ${task === 'segmentation' ? 'tag-seg' : 'tag-cls'}" style="margin-left:8px">${task}</span>
  </div>
  <div class="xai-images">`;

  if (explanation.overlay_base64) {
    html += `<div class="xai-img-card">
      <div class="xai-img-label">Heatmap Overlay</div>
      <img src="data:image/png;base64,${explanation.overlay_base64}" alt="overlay" />
    </div>`;
  }
  if (explanation.heatmap_base64) {
    html += `<div class="xai-img-card">
      <div class="xai-img-label">Raw Heatmap</div>
      <img src="data:image/png;base64,${explanation.heatmap_base64}" alt="heatmap" />
    </div>`;
  }
  if (explanation.raw_base64 && explanation.method === 'lime') {
    html += `<div class="xai-img-card">
      <div class="xai-img-label">LIME Superpixels</div>
      <img src="data:image/png;base64,${explanation.raw_base64}" alt="lime" />
    </div>`;
  }

  html += '</div>';
  if (explanation.metadata) {
    const layer = explanation.metadata.target_layer;
    if (layer) html += `<div style="margin-top:8px;font-size:11px;color:var(--muted)">Target layer: <code>${layer}</code></div>`;
  }
  if (explanation.prediction) {
    html += `<div style="margin-top:6px;font-size:11px;color:var(--muted)">Score: ${fmt(explanation.prediction.score ?? explanation.prediction.mask_mean)}</div>`;
  }

  const wrapper = document.createElement('div');
  wrapper.innerHTML = html;
  container.appendChild(wrapper);
}

// ─── PREDICT ─────────────────────────────────────────────────────────────────

document.getElementById('btnPredict').addEventListener('click', async () => {
  const file = document.getElementById('predictFile').files[0];
  if (!file) { toast('Select an image first', 'error'); return; }

  const explain = document.getElementById('explainChk').checked;
  const model   = document.getElementById('predictModelSelect').value;
  const method  = document.getElementById('predictXaiMethod').value;

  status('predictStatus', true);
  document.getElementById('predictResults').innerHTML = '';

  const form = new FormData();
  form.append('file', file);
  form.append('explain', String(explain));
  form.append('method', method);
  if (model) form.append('model_name', model);

  const { ok, data } = await apiFetch('/predict/upload', { method: 'POST', body: form });
  status('predictStatus', false);

  const out = document.getElementById('predictResults');
  out.innerHTML = '';

  if (!ok) {
    out.innerHTML = `<div style="color:var(--error);padding:12px">${data.detail || data.error || 'Prediction failed'}</div>`;
    toast('Prediction failed', 'error');
    return;
  }

  const task = (data.task || 'classification').toLowerCase();
  const originalSrc = document.getElementById('predictPreviewImg').src;

  if (task === 'segmentation') {
    renderSegmentation(data, originalSrc, out);
  } else if (task === 'joint') {
    // Classification part
    renderClassification(data, out);
    // Segmentation part
    if (data.segmentation_overlay_base64 || data.segmentation_overlay_path) {
      const segDiv = document.createElement('div');
      segDiv.style.marginTop = '18px';
      renderSegmentation(data, originalSrc, segDiv);
      out.appendChild(segDiv);
    }
  } else {
    renderClassification(data, out);
  }

  // XAI explanation
  if (explain && data.explanation) {
    renderExplanation(data.explanation, out);
  }

  toast('Prediction complete', 'success');
});

// ─── EVALUATE ────────────────────────────────────────────────────────────────

document.getElementById('btnEvaluate').addEventListener('click', async () => {
  const model = document.getElementById('evalModelSelect').value;
  if (!model) { toast('Select a model first', 'error'); return; }
  const split = document.getElementById('evalSplit').value;

  spinner('evalSpinner', true);
  document.getElementById('evalResults').innerHTML = '<div style="color:var(--muted);padding:8px">Running evaluation…</div>';

  const { ok, data } = await apiFetch('/evaluate/checkpoint', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ checkpoint_name: model, split }),
  });
  spinner('evalSpinner', false);

  const out = document.getElementById('evalResults');
  if (!ok || data.status === 'error') {
    out.innerHTML = `<div style="color:var(--error);padding:12px">${data.message || data.detail || 'Evaluation failed'}</div>`;
    toast('Evaluation failed', 'error');
    return;
  }

  const metrics = data.metrics || {};
  const analysis = data.metadata?.analysis || {};
  const report = data.report || {};

  // ─ Metric cards
  const METRIC_LABELS = {
    accuracy: 'Accuracy',
    f1_score: 'F1 Score',
    auc: 'AUC',
    dice_score: 'Dice Score',
    iou: 'IoU',
    hausdorff_distance: 'Hausdorff Dist',
    classification_accuracy: 'Cls Accuracy',
    classification_f1_score: 'Cls F1',
    classification_auc: 'Cls AUC',
    segmentation_dice_score: 'Seg Dice',
    segmentation_iou: 'Seg IoU',
  };

  function metricColor(name, val) {
    if (name.includes('hausdorff')) return val < 10 ? 'good' : (val < 50 ? 'warn' : 'bad');
    return val >= 0.85 ? 'good' : (val >= 0.60 ? 'warn' : 'bad');
  }

  let html = `<div style="margin-bottom:16px">
    <span class="tag tag-${data.task === 'segmentation' ? 'seg' : data.task === 'joint' ? 'joint' : 'cls'}">${(data.task || 'classification').toUpperCase()}</span>
    <span style="font-size:12px;color:var(--muted);margin-left:10px">split: ${split} · model: ${data.model_name || model}</span>
  </div>`;

  const shownEntries = Object.entries(metrics).filter(([k]) => k in METRIC_LABELS || !k.startsWith('val_'));
  if (shownEntries.length > 0) {
    html += '<div class="metrics-grid">';
    shownEntries.forEach(([key, val]) => {
      const label = METRIC_LABELS[key] || key.replace(/_/g, ' ');
      const v = typeof val === 'number' ? val : null;
      const cls = v !== null ? metricColor(key, v) : '';
      const disp = v !== null ? (key.includes('hausdorff') ? v.toFixed(2) : (v * 100).toFixed(1) + '%') : '—';
      html += `<div class="metric-card"><div class="metric-name">${label}</div><div class="metric-val ${cls}">${disp}</div></div>`;
    });
    html += '</div>';
  }

  // Confusion matrix image
  if (report.confusion_matrix_path) {
    html += `<div class="section-title">Confusion Matrix</div>
      <div class="cm-wrap">
        <img src="${base() + '/outputs/' + report.confusion_matrix_path.replace(/^.*outputs[/\\\\]/, '')}"
             alt="Confusion Matrix"
             onerror="this.style.display='none'" />
      </div>`;
  }

  // Class-wise table
  const clsAnalysis = analysis.classification || analysis;
  if (clsAnalysis.classwise && clsAnalysis.classwise.length > 0) {
    html += `<div class="section-title" style="margin-top:20px">Class-wise Analysis</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead>
          <tr style="color:var(--muted);border-bottom:1px solid var(--border)">
            <th style="text-align:left;padding:6px 10px">Class</th>
            <th style="padding:6px 10px">Precision</th>
            <th style="padding:6px 10px">Recall</th>
            <th style="padding:6px 10px">F1</th>
            <th style="padding:6px 10px">Support</th>
          </tr>
        </thead>
        <tbody>`;
    clsAnalysis.classwise.forEach(row => {
      html += `<tr style="border-bottom:1px solid var(--border)">
        <td style="padding:6px 10px;font-family:monospace">${row.class}</td>
        <td style="padding:6px 10px;text-align:center">${typeof row.precision === 'number' ? (row.precision * 100).toFixed(1) + '%' : row.precision}</td>
        <td style="padding:6px 10px;text-align:center">${typeof row.recall === 'number' ? (row.recall * 100).toFixed(1) + '%' : row.recall}</td>
        <td style="padding:6px 10px;text-align:center">${typeof row.f1_score === 'number' ? (row.f1_score * 100).toFixed(1) + '%' : row.f1_score}</td>
        <td style="padding:6px 10px;text-align:center">${row.support ?? '—'}</td>
      </tr>`;
    });
    html += '</tbody></table>';
  }

  out.innerHTML = html;
  toast('Evaluation complete', 'success');
});

// ─── XAI ─────────────────────────────────────────────────────────────────────

document.getElementById('btnExplain').addEventListener('click', async () => {
  const file = document.getElementById('xaiFile').files[0];
  if (!file) { toast('Select an image first', 'error'); return; }

  const model  = document.getElementById('xaiModelSelect').value;
  const method = document.getElementById('xaiMethodSelect').value;
  const task   = document.getElementById('xaiTaskSelect').value;

  status('xaiStatus', true);
  document.getElementById('xaiResults').innerHTML = '';

  const form = new FormData();
  form.append('file', file);
  form.append('method', method);
  form.append('target_task', task);
  if (model) form.append('model_name', model);

  const { ok, data } = await apiFetch('/explain/upload', { method: 'POST', body: form });
  status('xaiStatus', false);

  const out = document.getElementById('xaiResults');
  out.innerHTML = '';

  if (!ok || data.status === 'error') {
    out.innerHTML = `<div style="color:var(--error);padding:12px">${data.message || data.detail || 'Explanation failed'}</div>`;
    toast('Explanation failed', 'error');
    return;
  }

  const explanation = data.explanation;
  if (!explanation) {
    out.innerHTML = '<div style="color:var(--warn);padding:12px">No explanation data returned.</div>';
    return;
  }

  // Show original image
  const originalSrc = document.getElementById('xaiPreviewImg').src;
  let html = '<div class="xai-images">';
  html += `<div class="xai-img-card"><div class="xai-img-label">Original</div><img src="${originalSrc}" alt="original" /></div>`;
  html += '</div>';

  const wrapper = document.createElement('div');
  wrapper.innerHTML = html;
  out.appendChild(wrapper);

  renderExplanation(explanation, out);
  toast('Explanation generated', 'success');
});

// ─── INITIAL LOAD ────────────────────────────────────────────────────────────

(async () => {
  await connect();
})();
