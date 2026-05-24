/* BRISC 2025 — Brain Tumor Analysis · app.js  v2 */
'use strict';

// ─── UTILITIES ────────────────────────────────────────────────────────────────

function getApiBase() {
  const el = document.getElementById('apiBase');
  return el ? el.value.replace(/\/$/, '') : 'http://localhost:8000';
}

async function apiFetch(path, opts) {
  const url = getApiBase() + path;
  try {
    const res  = await fetch(url, opts || {});
    const text = await res.text();
    try   { return { ok: res.ok, status: res.status, data: JSON.parse(text) }; }
    catch { return { ok: false,  status: res.status, data: { error: text   } }; }
  } catch (err) {
    return { ok: false, status: 0, data: { error: err.message } };
  }
}

function toast(msg, type) {
  type = type || 'info';
  const c = document.getElementById('toast-container');
  if (!c) return;
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(function() { t.remove(); }, 4500);
}

function setResult(elId, data, asJson) {
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

function statusBar(id, on, mode) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('visible', !!on);
  if (mode) {
    el.classList.remove('seg-mode', 'hybrid-mode', 'view-mode');
    if (mode === 'seg')    el.classList.add('seg-mode');
    if (mode === 'hybrid') el.classList.add('hybrid-mode');
    if (mode === 'view')   el.classList.add('view-mode');
  }
}

function fmt(v) { return (typeof v === 'number') ? v.toFixed(4) : (v != null ? v : '—'); }

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setHtml(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

// ─── MODEL DISPLAY NAMES ──────────────────────────────────────────────────────

var MODEL_DISPLAY_NAMES = {
  'classification.resnet.scratch':        'ResNet-50 (From Scratch)',
  'classification.resnet.finetune':       'ResNet-50 (Fine-tuned)',
  'classification.efficientnet.scratch':  'EfficientNet-B3 (From Scratch)',
  'classification.efficientnet.finetune': 'EfficientNet-B3 (Fine-tuned)',
  'classification.vit.finetune':          'Vision Transformer (Fine-tuned)',
  'classification.view_classifier':       'View Plane Classifier',
  'segmentation.unet.scratch':            'U-Net (From Scratch)',
  'segmentation.unet.finetune':           'U-Net (Fine-tuned)',
  'segmentation.resnet_hybrid.scratch':   'ResNet Hybrid U-Net (From Scratch)',
  'segmentation.resnet_hybrid.finetune':  'ResNet Hybrid U-Net (Fine-tuned)',
  'segmentation.swin_hafunet.finetune':   'Swin HA-FUNet (Fine-tuned)',
  'hybrid.multitask_joint':               'Multitask Joint Model',
  'hybrid.adpt_net':                      'Adaptive Dual Path Net',
  'hybrid.tgd_adpt_net':                  'TGD Adaptive Dual Path Net',
};

var VIEW_LABELS = { ax: 'Axial', co: 'Coronal', sa: 'Sagittal' };

function formatModelName(modelId) {
  if (!modelId) return '—';
  if (MODEL_DISPLAY_NAMES[modelId]) return MODEL_DISPLAY_NAMES[modelId];
  var m = modelId.match(/^(.+)\.(ax|co|sa)$/);
  if (m) {
    var baseName = MODEL_DISPLAY_NAMES[m[1]] || fallbackName(m[1]);
    return baseName + ' — ' + (VIEW_LABELS[m[2]] || m[2].toUpperCase());
  }
  return fallbackName(modelId);
}

function fallbackName(id) {
  return id.replace(/\./g, ' / ').replace(/_/g, ' ')
    .replace(/\b\w/g, function(c) { return c.toUpperCase(); });
}

// ─── MODEL FILTERING ──────────────────────────────────────────────────────────

function isViewSpecific(modelId) {
  return /\.(ax|co|sa)$/.test(modelId);
}

function isBaseModel(m) { return !isViewSpecific(m); }

function filterModels(models, mode) {
  if (mode === 'cls' || mode === 'view-cls') {
    return models.filter(function(m) {
      return m.startsWith('classification.') &&
             m !== 'classification.view_classifier' &&
             isBaseModel(m);
    });
  }
  if (mode === 'seg' || mode === 'view-seg') {
    return models.filter(function(m) { return m.startsWith('segmentation.') && isBaseModel(m); });
  }
  if (mode === 'hybrid' || mode === 'view-hybrid') {
    return models.filter(function(m) { return m.startsWith('hybrid.') && isBaseModel(m); });
  }
  if (mode === 'view') {
    return models.filter(function(m) { return m === 'classification.view_classifier'; });
  }
  return models.filter(isBaseModel);
}

// ─── PREDICT MODES ────────────────────────────────────────────────────────────

var currentPredictMode = 'cls';

var MODE_CONFIG = {
  'cls':        { icon: 'Classification', title: 'Predict Classification',
                  desc: 'Classify the brain tumor type (Glioma, Meningioma, No Tumor, Pituitary) using a model trained on the full dataset.',
                  modelLabel: 'Classification Model', statusMode: null },
  'seg':        { icon: 'Segmentation',   title: 'Predict Segmentation',
                  desc: 'Generate a pixel-level tumor segmentation mask using a model trained on the full dataset.',
                  modelLabel: 'Segmentation Model', statusMode: 'seg' },
  'hybrid':     { icon: 'Hybrid',         title: 'Predict Hybrid',
                  desc: 'Run simultaneous classification and segmentation with a joint multitask model trained on the full dataset.',
                  modelLabel: 'Hybrid Model', statusMode: 'hybrid' },
  'view':       { icon: 'View Model',     title: 'Predict View Model',
                  desc: 'Identify the MRI acquisition plane: Axial (AX), Coronal (CO), or Sagittal (SA) using the view classifier.',
                  modelLabel: 'View Classifier', statusMode: 'view' },
  'view-cls':   { icon: 'View → Classify',   title: 'View then Classify',
                  desc: 'Step 1: detect the MRI plane (Axial / Coronal / Sagittal). Step 2: classify the tumor using the matching view-specific model.',
                  modelLabel: 'Classification Model', statusMode: null },
  'view-seg':   { icon: 'View → Segment',    title: 'View then Segment',
                  desc: 'Step 1: detect the MRI plane (Axial / Coronal / Sagittal). Step 2: segment the tumor using the matching view-specific model.',
                  modelLabel: 'Segmentation Model', statusMode: 'seg' },
  'view-hybrid':{ icon: 'View -> Hybrid',    title: 'View then Hybrid',
                  desc: 'Step 1: detect the MRI plane (Axial / Coronal / Sagittal). Step 2: run the matching view-specific hybrid model.',
                  modelLabel: 'Hybrid Model', statusMode: 'hybrid' },
};

// ─── INIT (script is at end of <body> so DOM is already ready) ───────────────

function init() {
  // Main header tabs
  document.querySelectorAll('.tab-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
      document.querySelectorAll('.tab-content').forEach(function(s) { s.classList.remove('active'); });
      btn.classList.add('active');
      var section = document.getElementById('tab-' + btn.dataset.tab);
      if (section) section.classList.add('active');
      if (btn.dataset.tab === 'evaluate') loadSavedReports();
    });
  });

  // Predict mode buttons
  document.querySelectorAll('.mode-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      setMode(btn.dataset.mode || 'cls');
    });
  });

  // Explain checkbox
  var explainChk = document.getElementById('explainChk');
  if (explainChk) {
    explainChk.addEventListener('change', function() {
      var opts = document.getElementById('predictXaiOpts');
      if (opts) opts.style.display = this.checked ? '' : 'none';
    });
  }

  // Upload zones
  setupUploadZone('predictUploadZone', 'predictFile', 'predictPreview', 'predictPreviewImg', function() {
    var btn = document.getElementById('btnPredict');
    if (btn) btn.disabled = false;
  });
  setupUploadZone('xaiUploadZone', 'xaiFile', 'xaiPreview', 'xaiPreviewImg', function() {
    var btn = document.getElementById('btnExplain');
    if (btn) btn.disabled = false;
  });

  // Button listeners
  wireButtons();

  // Set initial predict mode and auto-connect
  setMode('cls');
  connect();
}

// ─── SET PREDICT MODE ─────────────────────────────────────────────────────────

function setMode(mode) {
  try {
    if (!MODE_CONFIG[mode]) {
      console.warn('Unknown mode:', mode);
      mode = 'cls';
    }
    currentPredictMode = mode;
    var cfg = MODE_CONFIG[mode];

    // Update active button
    document.querySelectorAll('.mode-btn').forEach(function(b) { b.classList.remove('active'); });
    var activeBtn = document.querySelector('.mode-btn[data-mode="' + mode + '"]');
    if (activeBtn) activeBtn.classList.add('active');

    // Update banner
    setText('modeBannerTitle', cfg.title);
    setText('modeBannerDesc',  cfg.desc);

    // Update label
    setText('predictModelLabel', cfg.modelLabel);

    // Show/hide model row
    var modelRow = document.getElementById('predictModelRow');
    if (modelRow) modelRow.style.display = (mode === 'view') ? 'none' : '';

    // Populate model select
    populateSelectsForMode(mode);

    // Reset results panel
    setHtml('predictResults',
      '<div class="empty-state">' +
      '<div class="empty-state-icon" style="font-size:36px;opacity:.4">&#128202;</div>' +
      '<div>Upload an image and click <strong>Run Prediction</strong></div>' +
      '</div>');

  } catch(e) {
    console.error('setMode error:', e);
  }
}

// ─── MODEL REGISTRY ───────────────────────────────────────────────────────────

var registeredModels = [];

function chipClass(name) {
  if (name === 'classification.view_classifier') return 'view';
  if (name.startsWith('classification.')) return 'classification';
  if (name.startsWith('segmentation.'))   return 'segmentation';
  if (name.startsWith('hybrid.'))         return 'hybrid';
  return '';
}

function renderModelRegistry(models) {
  var registry = document.getElementById('modelRegistry');
  if (!registry) return;

  if (!models || models.length === 0) {
    registry.innerHTML = '<div style="color:var(--muted);font-size:13px">No models registered. Run training first.</div>';
    return;
  }

  var groups = {
    classification: { label: 'Classification',        dot: '#4f8ef7', models: [] },
    segmentation:   { label: 'Segmentation',          dot: '#34d058', models: [] },
    hybrid:         { label: 'Hybrid / Joint',        dot: '#c084fc', models: [] },
    view:           { label: 'View Classifier',       dot: '#f0b429', models: [] },
    view_variants:  { label: 'View-Specific Variants',dot: '#7a8fbb', models: [] },
  };

  models.forEach(function(m) {
    if (isViewSpecific(m))                         { groups.view_variants.models.push(m);  return; }
    if (m === 'classification.view_classifier')    { groups.view.models.push(m);           return; }
    if (m.startsWith('classification.'))           { groups.classification.models.push(m); return; }
    if (m.startsWith('segmentation.'))             { groups.segmentation.models.push(m);   return; }
    if (m.startsWith('hybrid.'))                   { groups.hybrid.models.push(m);         return; }
  });

  registry.innerHTML = '';
  Object.values(groups).forEach(function(group) {
    if (group.models.length === 0) return;
    var div = document.createElement('div');
    div.className = 'model-group';
    var chipsHtml = group.models.map(function(m) {
      return '<span class="model-chip ' + chipClass(m) + '" title="' + m + '">' +
             formatModelName(m) + '</span>';
    }).join('');
    div.innerHTML =
      '<div class="model-group-title" style="color:' + group.dot + '">' +
        '<span class="dot" style="background:' + group.dot + '"></span>' +
        group.label +
        ' <span style="color:var(--muted2);font-weight:400">(' + group.models.length + ')</span>' +
      '</div>' +
      '<div class="model-chips">' + chipsHtml + '</div>';
    registry.appendChild(div);
  });
}

function populateSelectsForMode(mode) {
  var sel = document.getElementById('predictModelSelect');
  if (!sel) return;
  var filtered = filterModels(registeredModels, mode);
  var cur = sel.value;
  sel.innerHTML = '<option value="">— Select model —</option>';
  filtered.forEach(function(m) {
    var opt = document.createElement('option');
    opt.value = m;
    opt.textContent = formatModelName(m);
    sel.appendChild(opt);
  });
  if (cur && filtered.indexOf(cur) >= 0) sel.value = cur;
}

// ─── TRAIN MODEL LIST STATE ──────────────────────────────────────────────────

var _trainAllModels = [];
var _trainFilter = 'all';
var _trainSearch = '';

function _modelTaskCategory(m) {
  if (m === 'classification.view_classifier') return 'view';
  if (isViewSpecific(m)) return 'view';
  if (m.startsWith('classification.')) return 'classification';
  if (m.startsWith('segmentation.'))   return 'segmentation';
  if (m.startsWith('hybrid.'))         return 'hybrid';
  return 'other';
}

function _trainTagHtml(cat) {
  var map = {
    classification: ['CLS',  'cls'],
    segmentation:   ['SEG',  'seg'],
    hybrid:         ['JOINT','hyb'],
    view:           ['VIEW', 'view'],
  };
  var pair = map[cat] || ['?', 'cls'];
  return '<span class="tm-tag tm-tag-' + pair[1] + '">' + pair[0] + '</span>';
}

function renderTrainList() {
  var list = document.getElementById('trainModelList');
  if (!list) return;
  var search = _trainSearch.toLowerCase();
  var items = _trainAllModels.filter(function(m) {
    var matchFilter = _trainFilter === 'all' || _modelTaskCategory(m) === _trainFilter;
    var matchSearch = !search || m.toLowerCase().indexOf(search) >= 0
                              || formatModelName(m).toLowerCase().indexOf(search) >= 0;
    return matchFilter && matchSearch;
  });

  if (items.length === 0) {
    list.innerHTML = '<div style="color:var(--muted);padding:12px;font-size:13px">No models match filter.</div>';
    updateSelectedCount();
    return;
  }

  list.innerHTML = items.map(function(m) {
    var cat = _modelTaskCategory(m);
    return '<label class="tm-item">'
      + '<input type="checkbox" value="' + m + '" />'
      + '<span class="tm-label">' + formatModelName(m) + '</span>'
      + _trainTagHtml(cat)
      + '</label>';
  }).join('');
  updateSelectedCount();
}

function updateSelectedCount() {
  var checked = document.querySelectorAll('#trainModelList input[type=checkbox]:checked');
  var el = document.getElementById('trainSelectedCount');
  if (el) el.textContent = checked.length + ' model' + (checked.length !== 1 ? 's' : '') + ' selected';
  var btn = document.getElementById('btnTrain');
  if (btn) btn.textContent = checked.length > 1
    ? '▶ Train Selected (' + checked.length + ')'
    : '▶ Train Selected';
}

function getSelectedTrainModels() {
  var checked = document.querySelectorAll('#trainModelList input[type=checkbox]:checked');
  return Array.from(checked).map(function(cb) { return cb.value; });
}

function populateAllSelects(models) {
  _trainAllModels = models.slice();
  renderTrainList();

  // Attach live checkbox listeners after rendering
  var list = document.getElementById('trainModelList');
  if (list) list.addEventListener('change', updateSelectedCount);

  // Eval & XAI selects (all models)
  ['evalModelSelect', 'xaiModelSelect'].forEach(function(id) {
    var sel = document.getElementById(id);
    if (!sel) return;
    var cur = sel.value;
    var isXai = (id === 'xaiModelSelect');
    sel.innerHTML = isXai
      ? '<option value="">— Auto: latest checkpoint —</option>'
      : '<option value="">— Select model —</option>';
    models.forEach(function(m) {
      var opt = document.createElement('option');
      opt.value = m; opt.textContent = formatModelName(m);
      sel.appendChild(opt);
    });
    if (cur) sel.value = cur;
  });

  // Enable buttons
  var hasModels = models.length > 0;
  ['btnTrain','btnTrainAll','btnTune','btnEvaluate','btnReport'].forEach(function(id) {
    var b = document.getElementById(id);
    if (b) b.disabled = !hasModels;
  });
}

// ─── LOAD MODELS ─────────────────────────────────────────────────────────────

async function loadModels() {
  try {
    var result = await apiFetch('/models/list');
    if (!result.ok || !result.data || !Array.isArray(result.data.models)) {
      console.warn('models/list returned unexpected data:', result.data);
      toast('Could not load model list', 'error');
      return;
    }

    registeredModels = result.data.models;

    var isBase   = function(m) { return !isViewSpecific(m); };
    var clsCount = registeredModels.filter(function(m) {
      return m.startsWith('classification.') && m !== 'classification.view_classifier' && isBase(m);
    }).length;
    var segCount = registeredModels.filter(function(m) { return m.startsWith('segmentation.') && isBase(m); }).length;
    var hybCount = registeredModels.filter(function(m) { return m.startsWith('hybrid.') && isBase(m); }).length;
    var viewCount = registeredModels.filter(isViewSpecific).length;

    setText('statModels', registeredModels.length);
    setText('statCls',    clsCount);
    setText('statSeg',    segCount);
    setText('statHybrid', hybCount);
    setText('statView',   viewCount);

    renderModelRegistry(registeredModels);
    populateAllSelects(registeredModels);
    populateSelectsForMode(currentPredictMode);

  } catch(e) {
    console.error('loadModels error:', e);
    toast('Error loading models: ' + e.message, 'error');
  }
}

// ─── CONNECT ─────────────────────────────────────────────────────────────────

async function connect() {
  var dot    = document.getElementById('connDot');
  var health = document.getElementById('statHealth');
  var sub    = document.getElementById('statHealthSub');

  if (dot) dot.className = 'conn-dot';

  try {
    var result = await apiFetch('/health');
    if (result.ok && result.data && result.data.status === 'ok') {
      if (dot)    dot.className = 'conn-dot ok';
      if (health) { health.textContent = 'Online'; health.style.color = 'var(--success)'; }
      if (sub)    sub.textContent = getApiBase();
      toast('Connected to ML service', 'success');
      await loadModels();
      var btnXai = document.getElementById('btnExplain');
      if (btnXai) btnXai.disabled = false;
    } else {
      if (dot)    dot.className = 'conn-dot err';
      if (health) { health.textContent = 'Offline'; health.style.color = 'var(--error)'; }
      if (sub)    sub.textContent = (result.data && result.data.error) || 'Cannot reach API';
      toast('Cannot reach API — is the server running?', 'error');
    }
  } catch(e) {
    console.error('connect error:', e);
    if (dot)    dot.className = 'conn-dot err';
    if (health) { health.textContent = 'Error'; health.style.color = 'var(--error)'; }
    toast('Connection error: ' + e.message, 'error');
  }
}

// ─── UPLOAD ZONES ─────────────────────────────────────────────────────────────

function setupUploadZone(zoneId, inputId, previewId, previewImgId, onFile) {
  var zone  = document.getElementById(zoneId);
  var input = document.getElementById(inputId);
  var wrap  = document.getElementById(previewId);
  var img   = document.getElementById(previewImgId);
  if (!zone || !input) return;

  zone.addEventListener('click', function() { input.click(); });
  zone.addEventListener('dragover', function(e) { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', function() { zone.classList.remove('drag'); });
  zone.addEventListener('drop', function(e) {
    e.preventDefault(); zone.classList.remove('drag');
    var f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  input.addEventListener('change', function() { if (input.files[0]) handleFile(input.files[0]); });

  function handleFile(f) {
    var reader = new FileReader();
    reader.onload = function(ev) {
      if (img) img.src = ev.target.result;
      if (wrap) wrap.classList.add('visible');
    };
    reader.readAsDataURL(f);
    if (onFile) onFile(f);
  }
}

// ─── BUTTON WIRING ─────────────────────────────────────────────────────────────

function wireButtons() {

  // Connect button
  var btnConnect = document.getElementById('btnConnect');
  if (btnConnect) btnConnect.addEventListener('click', connect);

  // Dashboard buttons
  var dashPrepare = document.getElementById('dashBtnPrepare');
  if (dashPrepare) dashPrepare.addEventListener('click', async function() {
    spinner('dashSpinner', true);
    setResult('dashOutput', 'Preparing dataset...');
    var r = await apiFetch('/train/', { method:'POST', headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({ action:'prepare' }) });
    spinner('dashSpinner', false);
    setResult('dashOutput', r.data, true);
    if (r.data.status === 'completed') toast('Dataset ready: ' + r.data.processed_count + ' processed', 'success');
    else toast('Prepare failed: ' + (r.data.message || 'unknown'), 'error');
  });

  var dashDivided = document.getElementById('dashBtnPrepareDivided');
  if (dashDivided) dashDivided.addEventListener('click', async function() {
    spinner('dashSpinner', true);
    setResult('dashOutput', 'Preprocessing divided dataset...');
    var r = await apiFetch('/train/', { method:'POST', headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({ action:'prepare_divided' }) });
    spinner('dashSpinner', false);
    setResult('dashOutput', r.data, true);
    if (r.data.status === 'completed') toast('Divided dataset ready: ' + r.data.processed_count + ' processed', 'success');
    else toast('Preprocess failed: ' + (r.data.message || 'unknown'), 'error');
  });

  var dashRefresh = document.getElementById('dashBtnRefresh');
  if (dashRefresh) dashRefresh.addEventListener('click', async function() {
    await loadModels();
    toast('Models refreshed', 'info');
  });

  // Train tab
  var btnPrepare = document.getElementById('btnPrepare');
  if (btnPrepare) btnPrepare.addEventListener('click', async function() {
    spinner('prepareSpinner', true);
    setResult('prepareResult', 'Preparing dataset...');
    var r = await apiFetch('/train/', { method:'POST', headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({ action:'prepare' }) });
    spinner('prepareSpinner', false);
    setResult('prepareResult', r.data, true);
    if (r.data.status === 'completed')
      toast('Dataset prepared: ' + r.data.processed_count + ' processed, ' + r.data.skipped_count + ' skipped', 'success');
    else toast('Prepare failed: ' + (r.data.message || JSON.stringify(r.data)), 'error');
  });

  var btnPrepareDivided = document.getElementById('btnPrepareDivided');
  if (btnPrepareDivided) btnPrepareDivided.addEventListener('click', async function() {
    spinner('prepareSpinner', true);
    setResult('prepareResult', 'Preprocessing divided dataset...');
    var r = await apiFetch('/train/', { method:'POST', headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({ action:'prepare_divided' }) });
    spinner('prepareSpinner', false);
    setResult('prepareResult', r.data, true);
    if (r.data.status === 'completed')
      toast('Divided preprocessed: ' + r.data.processed_count + ' processed', 'success');
    else toast('Preprocess failed: ' + (r.data.message || JSON.stringify(r.data)), 'error');
  });

  // ── Filter chips ──
  var chips = document.querySelectorAll('.train-chips .chip');
  chips.forEach(function(chip) {
    chip.addEventListener('click', function() {
      chips.forEach(function(c) { c.classList.remove('chip-active'); });
      chip.classList.add('chip-active');
      _trainFilter = chip.getAttribute('data-filter') || 'all';
      renderTrainList();
    });
  });

  // ── Search input ──
  var trainSearch = document.getElementById('trainSearch');
  if (trainSearch) trainSearch.addEventListener('input', function() {
    _trainSearch = trainSearch.value;
    renderTrainList();
  });

  // ── Select All / Clear All ──
  var btnSelectAll = document.getElementById('btnSelectAll');
  if (btnSelectAll) btnSelectAll.addEventListener('click', function() {
    document.querySelectorAll('#trainModelList input[type=checkbox]').forEach(function(cb) { cb.checked = true; });
    updateSelectedCount();
  });
  var btnClearAll = document.getElementById('btnClearAll');
  if (btnClearAll) btnClearAll.addEventListener('click', function() {
    document.querySelectorAll('#trainModelList input[type=checkbox]').forEach(function(cb) { cb.checked = false; });
    updateSelectedCount();
  });

  // ── Batch Train ──
  var _batchQueue = [], _batchIndex = 0, _batchResults = [];

  function _batchNext() {
    if (_batchIndex >= _batchQueue.length) {
      spinner('trainSpinner', false);
      statusBar('trainStatus', false);
      var prog = document.getElementById('batchProgress');
      if (prog) prog.style.display = 'none';
      var failed = _batchResults.filter(function(r) { return r.status === 'error'; }).length;
      var ok = _batchResults.length - failed;
      toast('Training complete: ' + ok + ' succeeded, ' + failed + ' failed', failed ? 'warn' : 'success');
      var html = '<div style="margin-top:8px">';
      _batchResults.forEach(function(r) {
        var icon = r.status === 'error' ? '✗' : '✓';
        var col  = r.status === 'error' ? 'var(--error)' : 'var(--success)';
        html += '<div style="font-size:13px;padding:3px 0;color:' + col + '">' + icon + ' ' + formatModelName(r.model) + (r.msg ? ' — ' + r.msg : '') + '</div>';
      });
      html += '</div>';
      setHtml('trainResult', html);
      loadModels();
      return;
    }

    var model = _batchQueue[_batchIndex];
    var total = _batchQueue.length;
    var curr  = _batchIndex + 1;

    // Update progress bar
    var pct = Math.round((_batchIndex / total) * 100);
    var fill = document.getElementById('batchProgressFill');
    var ptext = document.getElementById('batchProgressText');
    if (fill) fill.style.width = pct + '%';
    if (ptext) ptext.textContent = curr + ' / ' + total + ': ' + formatModelName(model);

    setText('trainStatusText', '(' + curr + '/' + total + ') Training ' + formatModelName(model) + '…');
    setHtml('trainResult', '<div style="color:var(--muted);font-size:13px">Training ' + curr + '/' + total + ': <strong>' + formatModelName(model) + '</strong>…</div>');

    lossChart.startSSE(model, getApiBase(), {
      onDone: function(evt) {
        _batchResults.push({ model: model, status: 'ok' });
        _batchIndex++;
        _batchNext();
      },
      onError: function(msg) {
        _batchResults.push({ model: model, status: 'error', msg: msg });
        _batchIndex++;
        toast(formatModelName(model) + ' failed — continuing…', 'error');
        _batchNext();
      },
    });
  }

  var btnTrain = document.getElementById('btnTrain');
  if (btnTrain) btnTrain.addEventListener('click', function() {
    var selected = getSelectedTrainModels();
    if (selected.length === 0) { toast('Select at least one model', 'error'); return; }
    _batchQueue = selected;
    _batchIndex = 0;
    _batchResults = [];
    spinner('trainSpinner', true);
    statusBar('trainStatus', true);
    var prog = document.getElementById('batchProgress');
    if (prog) prog.style.display = 'flex';
    _batchNext();
  });

  var btnTrainAll = document.getElementById('btnTrainAll');
  if (btnTrainAll) btnTrainAll.addEventListener('click', function() {
    // Check all visible, then trigger train
    document.querySelectorAll('#trainModelList input[type=checkbox]').forEach(function(cb) { cb.checked = true; });
    updateSelectedCount();
    var selected = getSelectedTrainModels();
    if (selected.length === 0) { toast('No models loaded', 'error'); return; }
    _batchQueue = selected;
    _batchIndex = 0;
    _batchResults = [];
    spinner('trainSpinner', true);
    statusBar('trainStatus', true);
    var prog = document.getElementById('batchProgress');
    if (prog) prog.style.display = 'flex';
    _batchNext();
  });

  var btnTune = document.getElementById('btnTune');
  if (btnTune) btnTune.addEventListener('click', async function() {
    var trialsEl = document.getElementById('tuneTrials');
    var trials = trialsEl ? (parseInt(trialsEl.value) || 5) : 5;
    spinner('tuneSpinner', true);
    setResult('tuneResult', 'Running ' + trials + ' tuning trials...');
    var r = await apiFetch('/train/', { method:'POST', headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({ action:'tune', trials: trials }) });
    spinner('tuneSpinner', false);
    setResult('tuneResult', r.data, true);
    if (r.data.status === 'completed')
      toast('Tuning done — best ' + r.data.metric_name + ': ' + fmt(r.data.best_metric), 'success');
    else toast('Tuning failed: ' + (r.data.message || ''), 'error');
  });

  // Predict button
  var btnPredict = document.getElementById('btnPredict');
  if (btnPredict) btnPredict.addEventListener('click', runPredict);

  // Evaluate button
  var btnEvaluate = document.getElementById('btnEvaluate');
  if (btnEvaluate) btnEvaluate.addEventListener('click', runEvaluate);

  // Report button
  var btnReport = document.getElementById('btnReport');
  if (btnReport) btnReport.addEventListener('click', runReport);

  // Refresh saved reports button
  var btnRefreshReports = document.getElementById('btnRefreshReports');
  if (btnRefreshReports) btnRefreshReports.addEventListener('click', loadSavedReports);

  // XAI button
  var btnExplain = document.getElementById('btnExplain');
  if (btnExplain) btnExplain.addEventListener('click', runExplain);
}

// ─── LOSS CHART ───────────────────────────────────────────────────────────────

var lossChart = (function() {
  var trainPts = [], valPts = [], sse = null;

  function draw() {
    var canvas = document.getElementById('lossChart');
    if (!canvas) return;
    var dpr = window.devicePixelRatio || 1;
    var W = canvas.offsetWidth, H = canvas.offsetHeight;
    if (!W || !H) return;
    canvas.width = W * dpr; canvas.height = H * dpr;
    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    var pad = { top:8, right:12, bottom:28, left:44 };
    var cw  = W - pad.left - pad.right;
    var ch  = H - pad.top  - pad.bottom;
    ctx.clearRect(0, 0, W, H);
    var all = trainPts.concat(valPts);
    if (all.length === 0) return;
    var maxX  = Math.max.apply(null, all.map(function(p) { return p.x; }));
    var minY  = Math.min.apply(null, all.map(function(p) { return p.y; }));
    var maxY  = Math.max.apply(null, all.map(function(p) { return p.y; }));
    var rng   = maxY - minY || 1;
    function sx(x) { return pad.left + (x / Math.max(maxX, 1)) * cw; }
    function sy(y) { return pad.top + ch - ((y - minY) / rng) * ch; }
    ctx.strokeStyle = '#2a3550'; ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
      var gy = pad.top + (ch / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(pad.left + cw, gy); ctx.stroke();
      ctx.fillStyle = '#7a8fbb'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
      ctx.fillText((maxY - (rng / 4) * i).toFixed(3), pad.left - 4, gy + 3);
    }
    ctx.textAlign = 'center'; ctx.fillStyle = '#7a8fbb'; ctx.font = '10px system-ui';
    var ticks = Math.min(maxX, 6);
    for (var j = 0; j <= ticks; j++) {
      var ex = Math.round(j * maxX / ticks);
      ctx.fillText(ex, sx(ex), pad.top + ch + 16);
    }
    function drawLine(pts, color) {
      if (pts.length < 2) return;
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
      pts.forEach(function(p, i) { if (i===0) ctx.moveTo(sx(p.x),sy(p.y)); else ctx.lineTo(sx(p.x),sy(p.y)); });
      ctx.stroke();
      var last = pts[pts.length-1];
      ctx.beginPath(); ctx.fillStyle = color; ctx.arc(sx(last.x), sy(last.y), 4, 0, Math.PI*2); ctx.fill();
    }
    drawLine(trainPts, '#4f8ef7');
    drawLine(valPts,   '#34d058');
  }

  function reset() {
    trainPts = []; valPts = [];
    if (sse) { sse.close(); sse = null; }
    var wrap = document.getElementById('lossChartWrap');
    if (wrap) wrap.classList.remove('visible');
    setText('lossChartEpoch', 'Epoch 0 / —');
  }

  function startSSE(modelName, apiBase, callbacks) {
    reset();
    var url = apiBase.replace(/\/$/, '') + '/train/stream/' + encodeURIComponent(modelName);
    sse = new EventSource(url);
    sse.onmessage = function(ev) {
      try {
        var d = JSON.parse(ev.data);
        if (d.type === 'epoch') {
          trainPts.push({ x: d.epoch, y: d.train_loss });
          valPts.push({   x: d.epoch, y: d.val_loss   });
          setText('lossChartEpoch', 'Epoch ' + d.epoch + ' / ' + (d.total || '?'));
          var wrap = document.getElementById('lossChartWrap');
          if (wrap) wrap.classList.add('visible');
          draw();
        } else if (d.type === 'done' || d.type === 'complete') {
          sse.close(); sse = null;
          if (callbacks && callbacks.onDone) callbacks.onDone(d);
        } else if (d.type === 'error') {
          sse.close(); sse = null;
          if (callbacks && callbacks.onError) callbacks.onError(d.message || 'Training error');
        }
      } catch(_) {}
    };
    sse.onerror = function() {
      if (sse) { sse.close(); sse = null; }
      if (callbacks && callbacks.onError) callbacks.onError('SSE connection lost');
    };
  }

  window.addEventListener('resize', draw);
  return { startSSE: startSSE, reset: reset };
})();

// ─── RENDER HELPERS ───────────────────────────────────────────────────────────

var CLASS_COLORS = {
  glioma:    { badge:'cls-glioma',    bar:'bar-glioma'    },
  meningioma:{ badge:'cls-meningioma',bar:'bar-meningioma'},
  no_tumor:  { badge:'cls-no_tumor',  bar:'bar-no_tumor'  },
  pituitary: { badge:'cls-pituitary', bar:'bar-pituitary' },
  ax:        { badge:'cls-ax',        bar:'bar-ax'        },
  co:        { badge:'cls-co',        bar:'bar-co'        },
  sa:        { badge:'cls-sa',        bar:'bar-sa'        },
};

function classColor(name) {
  return CLASS_COLORS[String(name == null ? '' : name).toLowerCase()] || { badge:'', bar:'bar-default' };
}

var CLASS_LABELS = {
  glioma:'Glioma', meningioma:'Meningioma', no_tumor:'No Tumor', pituitary:'Pituitary',
  ax:'Axial', co:'Coronal', sa:'Sagittal',
};

function formatClassName(name) {
  var key = String(name == null ? '' : name).toLowerCase();
  return CLASS_LABELS[key] || String(name == null ? 'Unknown' : name).replace(/_/g, ' ');
}

function renderClassification(data, container, sectionTitle) {
  var probs      = data.probabilities || {};
  var confidence = typeof data.confidence === 'number' ? data.confidence : null;
  var rawName    = data.class_name || data.class || data.predicted_class || 'Unknown';
  var label      = formatClassName(rawName);
  var col        = classColor(rawName);

  var html = '';
  if (sectionTitle) {
    html += '<div class="result-section-header"><span class="tag tag-cls">CLASSIFICATION</span>' +
            '<span class="result-section-title">' + sectionTitle + '</span></div>';
  }
  html += '<div class="cls-result">' +
          '<span class="cls-badge ' + col.badge + '">' + label + '</span>';
  if (confidence !== null) {
    html += '<div class="cls-confidence">Confidence: <strong>' + (confidence*100).toFixed(2) + '%</strong></div>';
  }
  var entries = Object.entries(probs).sort(function(a,b){ return b[1]-a[1]; });
  if (entries.length > 0) {
    html += '<div class="prob-bars">';
    entries.forEach(function(pair) {
      var cls = pair[0], p = pair[1];
      var c = classColor(cls);
      html += '<div class="prob-row">' +
              '<div class="prob-label">' + formatClassName(cls) + '</div>' +
              '<div class="prob-bar-wrap"><div class="prob-bar ' + c.bar + '" style="width:' + (p*100).toFixed(1) + '%"></div></div>' +
              '<div class="prob-pct">' + (p*100).toFixed(1) + '%</div>' +
              '</div>';
    });
    html += '</div>';
  }
  html += '</div>';
  container.insertAdjacentHTML('beforeend', html);
}

function imgCard(label, src) {
  return '<div class="seg-img-card"><div class="seg-img-label">' + label + '</div>' +
         '<img src="' + src + '" alt="' + label + '" loading="lazy" /></div>';
}

function renderSegmentation(data, originalSrc, container, sectionTitle) {
  var html = '';
  if (sectionTitle) {
    html += '<div class="result-section-header" style="margin-top:18px">' +
            '<span class="tag tag-seg">SEGMENTATION</span>' +
            '<span class="result-section-title">' + sectionTitle + '</span></div>';
  } else {
    html += '<div style="margin-bottom:10px"><span class="tag tag-seg">SEGMENTATION</span></div>';
  }
  html += '<div class="seg-images">';
  if (originalSrc) html += imgCard('Original', originalSrc);
  if (data.segmentation_overlay_base64)  html += imgCard('Tumor Overlay', data.segmentation_overlay_base64);
  else if (data.segmentation_overlay_path) html += imgCard('Tumor Overlay', data.segmentation_overlay_path);
  if (data.segmentation_mask_base64)     html += imgCard('Binary Mask', data.segmentation_mask_base64);
  else if (data.segmentation_mask_path)  html += imgCard('Binary Mask', data.segmentation_mask_path);
  html += '</div>';
  container.insertAdjacentHTML('beforeend', html);
}

function renderExplanation(explanation, container) {
  if (!explanation) return;
  if (explanation.metadata && explanation.metadata.error) {
    var isWarn = explanation.metadata.missing_dependency;
    container.insertAdjacentHTML('beforeend',
      '<div style="color:' + (isWarn ? 'var(--warn)' : 'var(--error)') + ';padding:10px;font-size:13px">' +
      (isWarn ? 'Warning: ' : 'Error: ') + explanation.metadata.error + '</div>');
    return;
  }
  if (explanation.task === 'joint' || (explanation.classification && explanation.segmentation)) {
    renderExplanation(explanation.classification || explanation, container);
    if (explanation.segmentation) {
      var div = document.createElement('div');
      div.style.marginTop = '16px';
      renderExplanation(explanation.segmentation, div);
      container.appendChild(div);
    }
    return;
  }
  var method = ((explanation.method || 'gradcam')).toUpperCase();
  var task   = explanation.task || 'classification';
  var html   = '<div style="margin-top:20px"><div class="result-section-header">' +
               '<span class="tag ' + (task==='segmentation'?'tag-seg':'tag-cls') + '">XAI &middot; ' + method + '</span>' +
               '<span class="result-section-title">' + task.charAt(0).toUpperCase() + task.slice(1) + '</span></div>' +
               '<div class="xai-images">';
  if (explanation.overlay_base64) {
    html += '<div class="xai-img-card"><div class="xai-img-label">Heatmap Overlay</div>' +
            '<img src="data:image/png;base64,' + explanation.overlay_base64 + '" alt="overlay" /></div>';
  }
  if (explanation.heatmap_base64) {
    html += '<div class="xai-img-card"><div class="xai-img-label">Raw Heatmap</div>' +
            '<img src="data:image/png;base64,' + explanation.heatmap_base64 + '" alt="heatmap" /></div>';
  }
  if (explanation.raw_base64 && explanation.method === 'lime') {
    html += '<div class="xai-img-card"><div class="xai-img-label">LIME Superpixels</div>' +
            '<img src="data:image/png;base64,' + explanation.raw_base64 + '" alt="lime" /></div>';
  }
  html += '</div>';
  if (explanation.metadata && explanation.metadata.target_layer) {
    html += '<div style="margin-top:8px;font-size:11px;color:var(--muted)">Target layer: <code>' +
            explanation.metadata.target_layer + '</code></div>';
  }
  html += '</div>';
  container.insertAdjacentHTML('beforeend', html);
}

// ─── RUN PREDICT ─────────────────────────────────────────────────────────────

async function runPredict() {
  var fileInput = document.getElementById('predictFile');
  var file = fileInput && fileInput.files[0];
  if (!file) { toast('Select an image first', 'error'); return; }

  var mode    = currentPredictMode;
  var cfg     = MODE_CONFIG[mode];
  var selEl   = document.getElementById('predictModelSelect');
  var modelId = selEl ? selEl.value : '';
  var explain = document.getElementById('explainChk') && document.getElementById('explainChk').checked;
  var methodEl= document.getElementById('predictXaiMethod');
  var method  = methodEl ? methodEl.value : 'gradcam';

  if (!modelId && mode !== 'view') { toast('Select a model first', 'error'); return; }

  var btn = document.getElementById('btnPredict');
  if (btn) btn.disabled = true;
  statusBar('predictStatus', true, cfg ? cfg.statusMode : null);
  var statusText = document.getElementById('predictStatusText');

  var out = document.getElementById('predictResults');
  if (out) out.innerHTML = '';

  var originalSrc = '';
  var prevImg = document.getElementById('predictPreviewImg');
  if (prevImg) originalSrc = prevImg.src;

  var isViewPipeline = (mode.indexOf('view-') === 0);

  try {
    if (isViewPipeline) {

      // ── Step 1: view classifier ────────────────────
      if (statusText) statusText.textContent = 'Step 1: Detecting MRI plane...';

      var vForm = new FormData();
      vForm.append('file', file);
      vForm.append('model_name', 'classification.view_classifier');
      var vRes = await apiFetch('/predict/upload', { method:'POST', body: vForm });
      if (!vRes.ok) throw new Error(vRes.data.detail || vRes.data.error || 'View classification failed');

      var vData     = vRes.data;
      var viewClass = (vData.class_name || vData.class || vData.predicted_class || '').toLowerCase();
      if (viewClass !== 'ax' && viewClass !== 'co' && viewClass !== 'sa') {
        throw new Error('Unexpected view class: "' + viewClass + '"');
      }

      // Render Step 1
      var step1 = document.createElement('div');
      step1.className = 'pipeline-step';
      step1.innerHTML =
        '<div class="pipeline-step-header">' +
          '<div class="pipeline-step-num">1</div>' +
          '<div class="pipeline-step-label">View Detection</div>' +
          '<div class="pipeline-step-sub">View Plane Classifier</div>' +
        '</div>' +
        '<div class="pipeline-step-body" id="pipeStep1"></div>';
      if (out) out.appendChild(step1);
      var body1 = document.getElementById('pipeStep1');
      if (body1) renderClassification(vData, body1);

      // ── Step 2: model on view-specific variant ─────
      var viewLabel  = VIEW_LABELS[viewClass] || viewClass.toUpperCase();
      var fullModel  = modelId + '.' + viewClass;
      if (statusText) statusText.textContent = 'Step 2: Running ' + formatModelName(modelId) + ' (' + viewLabel + ')...';

      var mForm = new FormData();
      mForm.append('file', file);
      mForm.append('model_name', fullModel);
      if (explain) { mForm.append('explain', 'true'); mForm.append('method', method); }
      var mRes = await apiFetch('/predict/upload', { method:'POST', body: mForm });
      if (!mRes.ok) throw new Error(mRes.data.detail || mRes.data.error || 'Model prediction failed');

      var mData = mRes.data;
      var taskType = (mData.task || '').toLowerCase();

      var step2TypeClass = (mode === 'view-seg') ? 'seg' : (mode === 'view-hybrid') ? 'hybrid' : '';
      var step2 = document.createElement('div');
      step2.className = 'pipeline-step';
      step2.innerHTML =
        '<div class="pipeline-step-header">' +
          '<div class="pipeline-step-num step2 ' + step2TypeClass + '">2</div>' +
          '<div class="pipeline-step-label">' + (cfg ? cfg.title.replace('View then ', '') : '') + '</div>' +
          '<div class="pipeline-step-sub">' + fullModel + '</div>' +
        '</div>' +
        '<div class="pipeline-step-body" id="pipeStep2"></div>';
      if (out) out.appendChild(step2);
      var body2 = document.getElementById('pipeStep2');
      if (body2) {
        if (taskType === 'segmentation') {
          renderSegmentation(mData, originalSrc, body2);
        } else if (taskType === 'joint') {
          renderClassification(mData, body2, 'Classification');
          if (mData.segmentation_overlay_base64 || mData.segmentation_overlay_path) {
            renderSegmentation(mData, originalSrc, body2, 'Segmentation');
          }
        } else {
          renderClassification(mData, body2);
        }
        if (explain && mData.explanation) renderExplanation(mData.explanation, body2);
      }

      toast('Pipeline complete — View: ' + viewLabel, 'success');

    } else if (mode === 'view') {

      // ── Single: view classifier ────────────────────
      if (statusText) statusText.textContent = 'Classifying MRI plane...';
      var svForm = new FormData();
      svForm.append('file', file);
      svForm.append('model_name', 'classification.view_classifier');
      var svRes = await apiFetch('/predict/upload', { method:'POST', body: svForm });
      if (!svRes.ok) throw new Error(svRes.data.detail || svRes.data.error || 'Classification failed');
      if (out) renderClassification(svRes.data, out);
      toast('View classified', 'success');

    } else {

      // ── Single: cls / seg / hybrid ─────────────────
      if (statusText) statusText.textContent = 'Running ' + formatModelName(modelId) + '...';
      var sForm = new FormData();
      sForm.append('file', file);
      sForm.append('model_name', modelId);
      if (explain) { sForm.append('explain', 'true'); sForm.append('method', method); }
      var sRes = await apiFetch('/predict/upload', { method:'POST', body: sForm });
      if (!sRes.ok) throw new Error(sRes.data.detail || sRes.data.error || 'Prediction failed');

      var sData = sRes.data;
      var sTask = (sData.task || 'classification').toLowerCase();
      if (out) {
        if (sTask === 'segmentation') {
          renderSegmentation(sData, originalSrc, out);
        } else if (sTask === 'joint') {
          renderClassification(sData, out, 'Classification');
          if (sData.segmentation_overlay_base64 || sData.segmentation_overlay_path) {
            renderSegmentation(sData, originalSrc, out, 'Segmentation');
          }
        } else {
          renderClassification(sData, out);
        }
        if (explain && sData.explanation) renderExplanation(sData.explanation, out);
      }
      toast('Prediction complete', 'success');
    }

  } catch(e) {
    console.error('runPredict error:', e);
    if (out) out.insertAdjacentHTML('beforeend',
      '<div style="color:var(--error);padding:14px;font-size:13px">Error: ' + e.message + '</div>');
    toast(e.message, 'error');
  } finally {
    if (btn) btn.disabled = false;
    statusBar('predictStatus', false);
  }
}

// ─── RUN EVALUATE ─────────────────────────────────────────────────────────────

async function runEvaluate() {
  var selEl = document.getElementById('evalModelSelect');
  var model = selEl ? selEl.value : '';
  if (!model) { toast('Select a model first', 'error'); return; }
  var splitEl = document.getElementById('evalSplit');
  var split   = splitEl ? splitEl.value : 'test';

  spinner('evalSpinner', true);
  setHtml('evalResults', '<div style="color:var(--muted);padding:8px">Running evaluation...</div>');

  try {
    var r = await apiFetch('/evaluate/checkpoint', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ checkpoint_name: model, split: split }),
    });
    spinner('evalSpinner', false);

    var out = document.getElementById('evalResults');
    if (!r.ok || r.data.status === 'error' || r.data.status === 'failed') {
      var errMsg = r.data.message || r.data.detail || r.data.error || 'Evaluation failed';
      if (out) out.innerHTML = '<div style="color:var(--error);padding:12px">' + errMsg + '</div>';
      toast('Evaluation failed: ' + errMsg, 'error');
      return;
    }

  var metrics  = r.data.metrics || {};
  var analysis = (r.data.metadata && r.data.metadata.analysis) || {};
  var report   = r.data.report || {};

  var METRIC_LABELS = {
    accuracy:'Accuracy', f1_score:'F1 Score', auc:'AUC',
    dice_score:'Dice Score', iou:'IoU', hausdorff_distance:'Hausdorff',
    classification_accuracy:'Cls Accuracy', classification_f1_score:'Cls F1', classification_auc:'Cls AUC',
    segmentation_dice_score:'Seg Dice', segmentation_iou:'Seg IoU',
  };

  function mColor(name, val) {
    if (name.indexOf('hausdorff') >= 0) return val < 10 ? 'good' : (val < 50 ? 'warn' : 'bad');
    return val >= 0.85 ? 'good' : (val >= 0.60 ? 'warn' : 'bad');
  }

  var taskTag = r.data.task === 'segmentation' ? 'tag-seg' : r.data.task === 'joint' ? 'tag-joint' : 'tag-cls';
  var html = '<div style="margin-bottom:18px;display:flex;align-items:center;gap:10px">' +
             '<span class="tag ' + taskTag + '">' + ((r.data.task || 'classification').toUpperCase()) + '</span>' +
             '<span style="font-size:12px;color:var(--muted)">split: ' + split + ' &middot; ' + formatModelName(r.data.model_name || model) + '</span></div>';

  var entries = Object.entries(metrics).filter(function(e) { return e[0] in METRIC_LABELS || !e[0].startsWith('val_'); });
  if (entries.length > 0) {
    html += '<div class="metrics-grid">';
    entries.forEach(function(e) {
      var key = e[0], val = e[1];
      var lbl  = METRIC_LABELS[key] || key.replace(/_/g, ' ');
      var v    = typeof val === 'number' ? val : null;
      var cls  = v !== null ? mColor(key, v) : '';
      var disp = v !== null ? (key.indexOf('hausdorff') >= 0 ? v.toFixed(2) : (v*100).toFixed(1)+'%') : '—';
      html += '<div class="metric-card"><div class="metric-name">' + lbl + '</div><div class="metric-val ' + cls + '">' + disp + '</div></div>';
    });
    html += '</div>';
  }

  if (report.confusion_matrix_path) {
    html += '<div class="section-title">Confusion Matrix</div><div class="cm-wrap">' +
            '<img src="' + getApiBase() + '/outputs/' + report.confusion_matrix_path.replace(/^.*outputs[/\\\\]/, '') +
            '" alt="Confusion Matrix" onerror="this.style.display=\'none\'" /></div>';
  }

  var clsAnalysis = analysis.classification || analysis;
  if (clsAnalysis.classwise && clsAnalysis.classwise.length > 0) {
    html += '<div class="section-title" style="margin-top:24px">Class-wise Analysis</div>' +
            '<table class="data-table"><thead><tr>' +
            '<th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>' +
            '</tr></thead><tbody>';
    clsAnalysis.classwise.forEach(function(row) {
      html += '<tr><td>' + formatClassName(row.class) + '</td>' +
              '<td style="text-align:center">' + (typeof row.precision==='number' ? (row.precision*100).toFixed(1)+'%' : row.precision) + '</td>' +
              '<td style="text-align:center">' + (typeof row.recall   ==='number' ? (row.recall   *100).toFixed(1)+'%' : row.recall   ) + '</td>' +
              '<td style="text-align:center">' + (typeof row.f1_score ==='number' ? (row.f1_score *100).toFixed(1)+'%' : row.f1_score ) + '</td>' +
              '<td style="text-align:center">' + (row.support != null ? row.support : '—') + '</td></tr>';
    });
    html += '</tbody></table>';
  }

  if (out) out.innerHTML = html;
  toast('Evaluation complete', 'success');
  } catch(e) {
    spinner('evalSpinner', false);
    var out2 = document.getElementById('evalResults');
    if (out2) out2.innerHTML = '<div style="color:var(--error);padding:12px">Error: ' + (e.message || e) + '</div>';
    toast('Evaluation error: ' + (e.message || e), 'error');
  }
}

// ─── RUN REPORT ───────────────────────────────────────────────────────────────

async function runReport() {
  var splitEl = document.getElementById('evalSplit');
  var split   = splitEl ? splitEl.value : 'test';

  spinner('reportSpinner', true);
  setHtml('reportResult', '<div style="color:var(--muted);font-size:13px;padding:6px 0">Generating report — evaluating all checkpoints, please wait…</div>');

  try {
    var r = await apiFetch('/evaluate/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ split: split, run_missing: true }),
    });
    spinner('reportSpinner', false);

    if (!r.ok || r.data.status !== 'completed') {
      var errMsg = r.data.message || r.data.error || 'Report generation failed';
      setHtml('reportResult', '<div style="color:var(--error);font-size:13px">' + errMsg + '</div>');
      toast('Report failed: ' + errMsg, 'error');
      return;
    }

    var reportUrl = getApiBase() + '/outputs/reports/' + r.data.filename;
    setHtml('reportResult',
      '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:4px">' +
      '<span style="color:var(--success);font-size:13px">Report ready — ' + r.data.evaluated_count + ' models evaluated.</span>' +
      '<a href="' + reportUrl + '" target="_blank" ' +
         'style="background:var(--accent);color:#fff;padding:6px 14px;border-radius:6px;font-size:13px;font-weight:600;text-decoration:none">Open Report</a>' +
      '</div>'
    );
    toast('Evaluation report ready!', 'success');
    loadSavedReports();
  } catch(e) {
    spinner('reportSpinner', false);
    setHtml('reportResult', '<div style="color:var(--error);font-size:13px">Error: ' + (e.message || e) + '</div>');
    toast('Report error: ' + (e.message || e), 'error');
  }
}

// ─── SAVED REPORTS ────────────────────────────────────────────────────────────

async function loadSavedReports() {
  var el = document.getElementById('savedReportsList');
  if (!el) return;
  el.innerHTML = '<div style="color:var(--muted);font-size:13px">Loading…</div>';

  var r = await apiFetch('/evaluate/reports');
  if (!r.ok || !r.data.reports) {
    el.innerHTML = '<div style="color:var(--muted);font-size:13px">Could not load reports.</div>';
    return;
  }

  var reports = r.data.reports;
  if (!reports.length) {
    el.innerHTML = '<div style="color:var(--muted);font-size:13px">No saved reports yet. Generate one above.</div>';
    return;
  }

  var rows = reports.map(function(rep) {
    var date = _reportDateLabel(rep.filename);
    var url  = getApiBase() + rep.url;
    return (
      '<div style="display:flex;align-items:center;justify-content:space-between;' +
           'padding:9px 0;border-bottom:1px solid var(--border)">' +
        '<div>' +
          '<div style="font-size:13px;font-weight:600;color:var(--text)">' + date + '</div>' +
          '<div style="font-size:11px;color:var(--muted);margin-top:2px">' + rep.filename + ' &middot; ' + rep.size_kb + ' KB</div>' +
        '</div>' +
        '<a href="' + url + '" target="_blank" ' +
           'style="background:var(--accent);color:#fff;padding:5px 13px;border-radius:6px;' +
                  'font-size:12px;font-weight:600;text-decoration:none;white-space:nowrap">Open</a>' +
      '</div>'
    );
  }).join('');

  el.innerHTML = rows;
}

function _reportDateLabel(filename) {
  // evaluation_report_test_20260519_143022.html → "Test · 2026-05-19 14:30"
  var m = filename.match(/evaluation_report_(\w+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
  if (!m) return filename;
  return m[1].charAt(0).toUpperCase() + m[1].slice(1) +
         ' split &middot; ' + m[2] + '-' + m[3] + '-' + m[4] +
         ' ' + m[5] + ':' + m[6];
}

// ─── RUN EXPLAIN ──────────────────────────────────────────────────────────────

async function runExplain() {
  var fileInput = document.getElementById('xaiFile');
  var file = fileInput && fileInput.files[0];
  if (!file) { toast('Select an image first', 'error'); return; }

  var selEl    = document.getElementById('xaiModelSelect');
  var taskEl   = document.getElementById('xaiTaskSelect');
  var methodEl = document.getElementById('xaiMethodSelect');
  var model  = selEl    ? selEl.value    : '';
  var task   = taskEl   ? taskEl.value   : 'auto';
  var method = methodEl ? methodEl.value : 'gradcam';

  statusBar('xaiStatus', true);
  var out = document.getElementById('xaiResults');
  if (out) out.innerHTML = '';

  var form = new FormData();
  form.append('file', file);
  form.append('method', method);
  form.append('target_task', task);
  if (model) form.append('model_name', model);

  var r = await apiFetch('/explain/upload', { method:'POST', body: form });
  statusBar('xaiStatus', false);

  if (!out) return;
  if (!r.ok || r.data.status === 'error') {
    out.innerHTML = '<div style="color:var(--error);padding:12px">' + (r.data.message || r.data.detail || 'Explanation failed') + '</div>';
    toast('Explanation failed', 'error');
    return;
  }

  var explanation = r.data.explanation;
  if (!explanation) {
    out.innerHTML = '<div style="color:var(--warn);padding:12px">No explanation data returned.</div>';
    return;
  }

  var prevImg = document.getElementById('xaiPreviewImg');
  var origSrc = prevImg ? prevImg.src : '';
  if (origSrc) {
    out.insertAdjacentHTML('beforeend',
      '<div class="xai-images"><div class="xai-img-card">' +
      '<div class="xai-img-label">Original</div>' +
      '<img src="' + origSrc + '" alt="original" /></div></div>');
  }

  renderExplanation(explanation, out);
  toast('Explanation generated', 'success');
}

// ─── START ───────────────────────────────────────────────────────────────────
init();
