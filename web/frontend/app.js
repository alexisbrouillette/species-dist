const API = window.location.port === '5500' ? 'http://localhost:8090' : window.location.origin;
const FULLMAP_GRID_N = 100;   // must match main.py FULLMAP_GRID_N

// ── State ─────────────────────────────────────────────────────────────────────
let sdmLayer       = null;
let currentSpecies = null;
let pollInterval   = null;
let currentVmin    = 0.0;
let currentVmax    = 1.0;
let tileCacheBuster = Date.now();
let currentPercentile = 'otsu';   // 'otsu' = default threshold, or null/'adaptive'/'mean'
let currentPercentiles = {};    // { adaptive, otsu, mean } in raw suitability space

// All species data — kept in memory so search is instant
let _allCached = [];   // [{name, slug, tile_count}]
let _allKnown  = [];   // [{name, slug, cached}]
let currentTab = 'cached';


// ── Log ───────────────────────────────────────────────────────────────────────
function log(msg, color = '#3d8b5e') {
  const el = document.getElementById('log-entries');
  if (el) {
    el.innerHTML += `<span style="color:${color}">[${new Date().toLocaleTimeString()}] ${msg}</span><br>`;
    el.scrollTop = el.scrollHeight;
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let _toastTimer;
function toast(msg, isError = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'show' + (isError ? ' error' : '');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.className = ''; }, 3000);
}

// ── Map setup ─────────────────────────────────────────────────────────────────
const map = L.map('map', { center: [45.4, -71.9], zoom: 7, zoomAnimation: true });

map.createPane('basePane');
map.createPane('hillshadePane');
map.createPane('sdmPane');
map.createPane('linesPane');
map.createPane('labelsPane');
map.getPane('basePane').style.zIndex      = 200;
map.getPane('hillshadePane').style.zIndex = 250;
map.getPane('sdmPane').style.zIndex       = 300;
map.getPane('linesPane').style.zIndex     = 350;  // water + roads above SDM
map.getPane('labelsPane').style.zIndex    = 400;  // all labels on top

// Hillshade uses multiply blend — darkens valleys/slopes on the base map
// without touching the SDM colour layer above it.
map.getPane('hillshadePane').style.mixBlendMode = 'multiply';

const baseLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', {
  pane: 'basePane', opacity: 1.0,
  attribution: '© OpenStreetMap contributors © CARTO',
}).addTo(map);

// ESRI World Hillshade — free, no API key, pure B&W shaded relief.
const hillshadeLayer = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}',
  { pane: 'hillshadePane', opacity: 0.45, attribution: 'Hillshade © Esri', maxZoom: 16 }
);

// CartoDB Dark Matter labels — city names, boundaries, on top of overlay
const terrainLabelsLayer = L.tileLayer(
  'https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png',
  { pane: 'labelsPane', opacity: 0.9, attribution: '© OpenStreetMap contributors © CARTO', maxZoom: 20 }
).addTo(map);

// CartoDB Dark Matter does not have separate lines, so we omit them (opacity 0)
// but keep the layer declared.
const linesLayer = L.tileLayer(
  'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png',
  { pane: 'linesPane', opacity: 0.0, attribution: '', maxZoom: 20 }
).addTo(map);

// ── Layer state ───────────────────────────────────────────────────────────────
let hillshadeOn = false;

function setHillshade(on) {
  hillshadeOn = on;
  if (on) hillshadeLayer.addTo(map);
  else     map.removeLayer(hillshadeLayer);
  document.getElementById('toggle-hillshade').classList.toggle('active', on);
}

// ── Theme Manager ─────────────────────────────────────────────────────────────
let currentTheme = localStorage.getItem('sdm-theme') || 'forest';

function applyTheme(theme) {
  currentTheme = theme;
  localStorage.setItem('sdm-theme', theme);
  document.documentElement.setAttribute('data-theme', theme);
  
  const select = document.getElementById('theme-select');
  if (select) select.value = theme;
  
  if (baseLayer) {
    const isLight = theme === 'light';
    baseLayer.setUrl(isLight
      ? 'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png'
      : 'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png'
    );
  }
  if (terrainLabelsLayer) {
    const isLight = theme === 'light';
    terrainLabelsLayer.setUrl(isLight
      ? 'https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png'
      : 'https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png'
    );
  }
}

// ── Legend ────────────────────────────────────────────────────────────────────
function updateLegend(vmin, vmax) {
  const lo = Math.round(vmin * 100), hi = Math.round(vmax * 100);
  document.getElementById('leg-min').textContent = `${lo}%`;
  document.getElementById('leg-mid').textContent = `${Math.round((lo+hi)/2)}%`;
  document.getElementById('leg-max').textContent = `${hi}%`;
}

// ── Progress (bottom bar) ─────────────────────────────────────────────────────
function showProgress(pct, text) {
  const track = document.getElementById('progress-track');
  const fill  = document.getElementById('progress-fill');
  const txt   = document.getElementById('progress-text');
  track.hidden = false;
  fill.style.width      = `${pct}%`;
  fill.style.background = pct < 40 ? '#e74c3c' : pct < 80 ? '#f39c12' : '#2ecc71';
  fill.style.boxShadow  = `0 0 6px ${pct < 40 ? '#e74c3c' : pct < 80 ? '#f39c12' : '#2ecc71'}`;
  txt.textContent = text;
}

function hideProgress() {
  document.getElementById('progress-track').hidden = true;
  document.getElementById('progress-fill').style.width = '0%';
}

// ── Active bar (bottom-center pill) ───────────────────────────────────────────
function setActiveBar(name) {
  const bar    = document.getElementById('active-bar');
  const dot    = document.getElementById('active-dot');
  const nm     = document.getElementById('active-name');
  const btn    = document.getElementById('generate-btn');
  const fmBtn  = document.getElementById('fullmap-btn');
  if (name) {
    bar.hidden      = false;
    nm.textContent  = name;
    btn.disabled    = false;
    fmBtn.disabled  = false;
    dot.className   = '';
    dot.style.cssText = 'width:6px;height:6px;border-radius:50%;background:#2ecc71;box-shadow:0 0 6px #2ecc71;flex-shrink:0';
    
    // Toggle button visibility: show viewport refinement if cached, else show generate map
    if (isSpeciesCached(name)) {
      btn.hidden = false;
      fmBtn.hidden = true;
    } else {
      btn.hidden = true;
      fmBtn.hidden = false;
    }
  } else {
    bar.hidden      = true;
    btn.disabled    = true;
    fmBtn.disabled  = true;
  }
}

function setActiveDotRunning(running) {
  const dot = document.getElementById('active-dot');
  if (running) {
    dot.style.animation = 'pulse 1s ease-in-out infinite';
  } else {
    dot.style.animation = '';
    dot.style.boxShadow = '0 0 6px #2ecc71';
  }
}

// ── Map lock ──────────────────────────────────────────────────────────────────
function setMapLocked(locked, message = '') {
  let overlay = document.getElementById('map-lock-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'map-lock-overlay';
    document.getElementById('map').appendChild(overlay);
  }
  if (locked) {
    overlay.innerHTML = `<div class="map-lock-inner">
      <div class="map-lock-spinner"></div>
      <div class="map-lock-msg">${message}</div>
    </div>`;
    overlay.style.display = 'flex';
  } else {
    overlay.style.display = 'none';
  }
}

let currentRenderMode = 'smooth'; // 'smooth' or 'raw'

function buildTileUrl(species) {
  let url = `${API}/tile/${encodeURIComponent(species)}/{z}/{x}/{y}.png?vmin=${currentVmin}&vmax=${currentVmax}&_cb=${tileCacheBuster}&render_mode=${currentRenderMode}`;
  if (currentPercentile !== null) url += `&percentile=${currentPercentile}`;
  return url;
}

function createSDMLayer(species) {
  return L.tileLayer(buildTileUrl(species), {
    opacity: 0.78, tileSize: 256, pane: 'sdmPane', keepBuffer: 2,
  });
}

// ── Polling ───────────────────────────────────────────────────────────────────
function startPolling(species) {
  stopPolling();
  setActiveDotRunning(true);
  let lastRedraw = 0;

  pollInterval = setInterval(() => {
    fetch(`${API}/progress/${encodeURIComponent(species)}`)
      .then(r => r.json())
      .then(p => {
        showProgress(p.pct ?? 0, `${p.done}/${p.total} patches · ${p.pct}%`);

        const now = Date.now();
        if (now - lastRedraw > 1000) {
          lastRedraw = now;
          if (sdmLayer) sdmLayer.redraw();
        }

        if (p.status === 'done' || (p.total > 0 && p.done >= p.total)) {
          showProgress(100, `✓ ${p.total} patches`);
          stopPolling();
          setActiveDotRunning(false);
          setMapLocked(false);
          tileCacheBuster = Date.now(); // Bust cache when new viewport tiles finish generating
          if (sdmLayer) sdmLayer.setUrl(buildTileUrl(currentSpecies));
          log(`✓ ${p.total} patches · vmin=${currentVmin.toFixed(2)} vmax=${currentVmax.toFixed(2)}`);
          setTimeout(hideProgress, 2000);
          refreshLists();
          // Refresh percentile panel — new tiles may shift the distribution
          initThresholdPanel(currentSpecies);
        }
      })
      .catch(() => {});
  }, 1000);
}

function stopPolling() {
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

// ── Viewport request ──────────────────────────────────────────────────────────
function requestViewport(species) {
  const b = map.getBounds();
  log(`Queuing tiles for ${species}…`, '#f39c12');
  fetch(`${API}/generate_viewport`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      species,
      minlon: b.getWest(), minlat: b.getSouth(),
      maxlon: b.getEast(), maxlat: b.getNorth(),
    }),
  })
  .then(r => r.json())
  .then(d => {
    if (d.queued > 0) {
      log(`${d.queued} sectors queued.`, '#2ecc71');
      startPolling(species);
    } else {
      log('Viewport already cached.', '#3498db');
      showProgress(100, '✓ Cached');
      setTimeout(hideProgress, 1500);
    }
  })
  .catch(e => log(`Error: ${e}`, '#e74c3c'));
}

// ── Threshold panel ───────────────────────────────────────────────────────────

function setThreshold(percentile) {
  if (percentile === null || percentile === 'null') {
    currentPercentile = null;
  } else {
    currentPercentile = percentile;
  }

  // Update active button state
  document.querySelectorAll('.threshold-step').forEach(btn => {
      const val = btn.dataset.percentile;
      const isActive = (val === 'null' && currentPercentile === null) ||
                       (val !== 'null' && val === currentPercentile);
    btn.classList.toggle('active', isActive);
  });

  // Update the inline display showing the raw cutoff value
  const disp = document.getElementById('threshold-value-display');
  if (currentPercentile === null) {
    disp.textContent = '';
  } else {
    const rawVal = currentPercentiles[currentPercentile];
    disp.textContent = rawVal != null ? `≥ ${rawVal.toFixed(4)}` : '';
  }

  // Bust tile cache and redraw
  if (sdmLayer && currentSpecies) {
    tileCacheBuster = Date.now();
    sdmLayer.setUrl(buildTileUrl(currentSpecies));
    sdmLayer.redraw();
  }
}

async function initThresholdPanel(species) {
  const panel = document.getElementById('threshold-panel');
  if (!species) { panel.hidden = true; return; }

  try {
    const data = await fetch(`${API}/species/percentiles/${encodeURIComponent(species)}`).then(r => r.json());
    if (!data.tile_count) { panel.hidden = true; return; }

    currentPercentiles = {
      adaptive: data.adaptive_raw,
      otsu: data.otsu_raw,
      mean: data.mean_raw
    };

    // Annotate buttons with the raw values as tooltips
    document.querySelectorAll('.threshold-step[data-percentile]').forEach(btn => {
      const pct = btn.dataset.percentile;
      if (pct !== 'null') {
        const val = currentPercentiles[pct];
        if (val !== undefined) btn.title = `Raw cutoff: ${val.toFixed(4)}`;
      }
    });

    // Re-apply the display for the currently selected threshold
    const disp = document.getElementById('threshold-value-display');
    if (currentPercentile !== null) {
      const rawVal = currentPercentiles[currentPercentile];
      disp.textContent = rawVal != null ? `≥ ${rawVal.toFixed(4)}` : '';
    }

    panel.hidden = false;
  } catch (e) {
    panel.hidden = true;
  }
}

// Wire up step buttons (once — event delegation on parent)
document.getElementById('threshold-steps').addEventListener('click', e => {
  const btn = e.target.closest('.threshold-step');
  if (!btn) return;
  setThreshold(btn.dataset.percentile);
});

// ── Load species ──────────────────────────────────────────────────────────────
async function loadSpecies(name) {
  if (sdmLayer) { map.removeLayer(sdmLayer); sdmLayer = null; }
  stopPolling();
  setMapLocked(false);

  currentSpecies    = name;
  currentVmin       = 0.0; currentVmax = 1.0;
  currentPercentile = 'otsu';   // Reset threshold to 'otsu' when switching species
  currentPercentiles = {};
  tileCacheBuster   = Date.now(); // Bust cache when changing species
  updateLegend(0, 1);
  setActiveBar(name);
  updateSpeciesCard(name);
  closeResults();

  // Reset threshold button state to "Otsu"
  setThreshold('otsu');
  // Hide panel until percentiles are loaded
  document.getElementById('threshold-panel').hidden = true;

  // Fetch true min/max from disk-backed scale endpoint to log true scale metadata
  try {
    const scale = await fetch(`${API}/species/scale/${encodeURIComponent(name)}`).then(r => r.json());
    if (scale.tile_count > 0 && scale.max_prob > scale.min_prob) {
      log(`Species true scale on disk: min=${scale.min_prob.toFixed(3)} max=${scale.max_prob.toFixed(3)}`, '#3498db');
    }
  } catch(e) {}

  sdmLayer = createSDMLayer(name);
  sdmLayer.addTo(map);
  log(`Selected: ${name}`, '#2ecc71');
  hideProgress();

  // Load percentile info (async, non-blocking — panel appears once data arrives)
  initThresholdPanel(name);

  // Update active state in search results if list is open
  renderResults(document.getElementById('cmd-input').value);
}

function generateViewport() {
  if (!currentSpecies) return;
  setMapLocked(true, 'Computing tiles…');
  requestViewport(currentSpecies);
}

// ── Command palette — search & render ─────────────────────────────────────────
function scoreItem(name, query) {
  if (!query) return 1;
  const n = name.toLowerCase(), q = query.toLowerCase();
  if (n === q)           return 4;
  if (n.startsWith(q))   return 3;
  if (n.includes(' ' + q)) return 2;
  if (n.includes(q))     return 1;
  return 0;
}

function updateTabCounts() {
  const cachedCount = _allCached.length;
  const allCount    = _allKnown.length;
  const plantsCount = _allKnown.filter(s => s.category === 'Trees & Plants').length;
  const birdsCount  = _allKnown.filter(s => s.category === 'Birds').length;
  const insectsCount= _allKnown.filter(s => s.category === 'Insects').length;

  const elAll = document.getElementById('count-all');
  if (elAll) elAll.textContent = allCount;
  const elPlants = document.getElementById('count-plants');
  if (elPlants) elPlants.textContent = plantsCount;
  const elBirds = document.getElementById('count-birds');
  if (elBirds) elBirds.textContent = birdsCount;
  const elInsects = document.getElementById('count-insects');
  if (elInsects) elInsects.textContent = insectsCount;
  const elCached = document.getElementById('count-cached');
  if (elCached) elCached.textContent = cachedCount;
}

function renderResults(query) {
  const list = document.getElementById('cmd-list');
  list.innerHTML = '';

  const q = (query || '').trim();
  updateTabCounts();

  // Unified pool of species
  const cachedMap = new Map(_allCached.map(s => [s.name, s]));
  
  let pool = _allKnown.map(sp => {
    const isCached = cachedMap.has(sp.name);
    const cachedData = cachedMap.get(sp.name);
    return {
      ...sp,
      type: isCached ? 'cached' : 'known',
      tile_count: isCached ? (cachedData.tile_count || 0) : 0,
      score: scoreItem(sp.name, q)
    };
  });

  // Filter by active tab
  if (currentTab === 'cached') {
    pool = pool.filter(sp => sp.type === 'cached');
  } else if (currentTab && currentTab !== 'all') {
    pool = pool.filter(sp => sp.category === currentTab);
  }

  // Filter by search query if active
  if (q) {
    pool = pool.filter(sp => sp.score > 0).sort((a, b) => b.score - a.score);
  }

  const totalMatches = pool.length;
  const capped = pool.slice(0, 100);

  if (totalMatches === 0) {
    list.innerHTML = `<div class="empty-row" style="padding:16px; text-align:center; color:var(--muted); font-size:12px;">No species found in ${currentTab} category.</div>`;
    return;
  }

  capped.forEach(sp => list.appendChild(makeCmdItem(sp)));

  if (totalMatches > 100) {
    const info = document.createElement('div');
    info.className = 'empty-row';
    info.style.cssText = 'padding:12px; text-align:center; color:var(--muted); font-size:11.5px; border-top:1px solid var(--border);';
    info.textContent = `showing top 100 of ${totalMatches} species · type to refine search`;
    list.appendChild(info);
  }

  // "Generate new" row if query doesn't match known species
  const allNames = _allKnown.map(s => s.name.toLowerCase());
  if (q.length > 1 && !allNames.includes(q.toLowerCase())) {
    const div = document.createElement('div');
    div.className = 'cmd-divider';
    list.appendChild(div);
    list.appendChild(makeGenerateRow(q));
  }
}

function makeCmdItem(sp) {
  const isRunning = pollInterval && currentSpecies === sp.name;
  const dotClass  = isRunning ? 'running' : sp.type === 'cached' ? 'cached' : 'known';
  
  const catIcon = sp.category === 'Insects' ? '🪲' :
                  sp.category === 'Birds' ? '🐦' :
                  sp.category === 'Trees & Plants' ? '🌲' : '🌿';

  const subText   = sp.type === 'cached'
    ? `${sp.tile_count} tile${sp.tile_count !== 1 ? 's' : ''} cached`
    : `${sp.category || 'Plant'} · ready to render`;

  const div = document.createElement('div');
  div.className = 'cmd-item' + (sp.name === currentSpecies ? ' active' : '');
  
  const badgeText = sp.type === 'cached' ? 'Cached' : (sp.category || 'Species');
  const badgeClass = sp.type === 'cached' ? 'cached' : 'known';
  
  div.innerHTML = `
    <span class="cmd-dot ${dotClass}"></span>
    <div class="cmd-item-info">
      <div class="cmd-item-name">${sp.name} <span class="cmd-cat-icon">${catIcon}</span></div>
      <div class="cmd-item-sub">${subText}</div>
    </div>
    <span class="cmd-badge ${badgeClass}">${badgeText}</span>`;

  div.addEventListener('click', () => {
    closeResults();
    loadSpecies(sp.name);
  });
  return div;
}

function makeGenerateRow(query) {
  const displayName = query.charAt(0).toUpperCase() + query.slice(1);
  const div = document.createElement('div');
  div.className = 'cmd-item generate-row';
  div.innerHTML = `
    <span class="cmd-dot new"></span>
    <div class="cmd-item-info">
      <div class="cmd-item-name">Generate "${displayName}"</div>
      <div class="cmd-item-sub">new species · register &amp; compute</div>
    </div>`;
  div.addEventListener('click', () => showDrawer(displayName));
  return div;
}

// ── Modal (new species form) ──────────────────────────────────────────────────
function showDrawer(name) {
  closeResults();
  
  const overlay = document.getElementById('add-species-overlay');
  const nameInput = document.getElementById('new-name');
  const subtitle  = document.getElementById('modal-species-name');
  const desc      = document.getElementById('new-desc');
  const fb        = document.getElementById('add-feedback');
  
  overlay.dataset.name = name;
  nameInput.value = name;
  
  if (name) {
    subtitle.textContent = 'Registering new species from search';
    nameInput.readOnly = true;
    nameInput.style.opacity = '0.75';
  } else {
    subtitle.textContent = 'Enter species name and description';
    nameInput.readOnly = false;
    nameInput.style.opacity = '1';
  }
  
  desc.value  = '';
  fb.textContent = '';
  
  overlay.hidden = false;
  setTimeout(() => {
    overlay.classList.add('show');
    if (name) {
      desc.focus();
    } else {
      nameInput.focus();
    }
  }, 20);
}

function hideDrawer() {
  const overlay = document.getElementById('add-species-overlay');
  if (overlay) {
    overlay.classList.remove('show');
    overlay.dataset.name = '';
    
    setTimeout(() => {
      overlay.hidden = true;
    }, 300);
  }
}

// ── Results open/close ────────────────────────────────────────────────────────
function openResults() {
  document.getElementById('cmd-results').hidden = false;
}

function closeResults() {
  document.getElementById('cmd-results').hidden = true;
}

// ── Refresh data lists ────────────────────────────────────────────────────────
async function refreshLists() {
  try {
    const [c, k] = await Promise.all([
      fetch(`${API}/species/cached`).then(r => r.json()),
      fetch(`${API}/species/list`).then(r => r.json()),
    ]);
    _allCached = c.species || [];
    _allKnown  = k.species || [];
    // Re-render if results are open
    if (!document.getElementById('cmd-results').hidden) {
      renderResults(document.getElementById('cmd-input').value);
    }
  } catch(e) {}
}

// ── Input wiring ──────────────────────────────────────────────────────────────
const cmdInput = document.getElementById('cmd-input');
const cmdClear = document.getElementById('cmd-clear');

cmdInput.addEventListener('focus', () => {
  openResults();
  renderResults(cmdInput.value);
});

cmdInput.addEventListener('input', () => {
  const q = cmdInput.value;
  cmdClear.classList.toggle('visible', q.length > 0);
  renderResults(q);
});

// Close on Escape
cmdInput.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeResults(); cmdInput.blur(); }
});

cmdClear.addEventListener('click', () => {
  cmdInput.value = '';
  cmdClear.classList.remove('visible');
  renderResults('');
  cmdInput.focus();
});

// Click outside to close
document.addEventListener('mousedown', e => {
  if (!document.getElementById('cmd').contains(e.target)) {
    closeResults();
  }
});

// Dropdown tabs wiring
document.querySelectorAll('.cmd-tab').forEach(tab => {
  tab.addEventListener('click', e => {
    document.querySelectorAll('.cmd-tab').forEach(t => t.classList.remove('active'));
    e.target.classList.add('active');
    currentTab = e.target.dataset.tab;
    renderResults(cmdInput.value);
  });
});

// Theme selector wiring
document.getElementById('theme-select').addEventListener('change', e => {
  applyTheme(e.target.value);
});

// Trigger add species modal
document.getElementById('btn-trigger-add').addEventListener('click', () => {
  showDrawer('');
});

// Close modal on Escape key
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    hideDrawer();
  }
});

// ── Layer toggles ─────────────────────────────────────────────────────────────
document.getElementById('toggle-hillshade').addEventListener('click', () => setHillshade(!hillshadeOn));

let borderLayer = null;
let borderOn    = true;

fetch(`${API}/geo/quebec_border.geojson?v=${Date.now()}`)
  .then(r => r.json())
  .then(geojsonData => {
    borderLayer = L.geoJSON(geojsonData, {
      style: {
        color: '#ffffff',
        weight: 3.5,
        opacity: 0.9,
        fill: false,
        dashArray: '7, 7',
        interactive: false,
      },
      pane: 'linesPane',
    }).addTo(map);
  })
  .catch(err => console.warn('Quebec border overlay unavailable:', err));

document.getElementById('toggle-border').addEventListener('click', () => {
  borderOn = !borderOn;
  const btn = document.getElementById('toggle-border');
  btn.classList.toggle('active', borderOn);
  if (borderLayer) {
    if (borderOn) map.addLayer(borderLayer);
    else map.removeLayer(borderLayer);
  }
});

document.getElementById('toggle-render-mode').addEventListener('click', () => {
  currentRenderMode = (currentRenderMode === 'smooth') ? 'raw' : 'smooth';
  const btn = document.getElementById('toggle-render-mode');
  btn.classList.toggle('active', currentRenderMode === 'raw');
  document.body.classList.toggle('raw-pixel-mode', currentRenderMode === 'raw');
  btn.textContent = (currentRenderMode === 'raw') ? '👾 1km Pixels' : '🌊 Isolines';
  log(`Render Mode: ${currentRenderMode === 'raw' ? 'Raw 1km Pixels' : 'Smoothed Isolines'}`, '#3498db');
  
  tileCacheBuster = Date.now();
  if (currentSpecies) {
    if (sdmLayer) map.removeLayer(sdmLayer);
    sdmLayer = createSDMLayer(currentSpecies).addTo(map);
  }
});

// ── Generate button ───────────────────────────────────────────────────────────
document.getElementById('generate-btn').addEventListener('click', generateViewport);

// ── Full map — 100×100 grid over full zarr extent ─────────────────────────────
async function generateFullmap() {
  if (!currentSpecies) return;

  const btn   = document.getElementById('fullmap-btn');
  const icon  = document.getElementById('fullmap-icon');
  const label = document.getElementById('fullmap-label');

  if (btn.classList.contains('running')) return;

  btn.classList.add('running');
  btn.disabled = true;
  icon.style.animation = 'spin 1s linear infinite';
  label.textContent    = 'Starting…';
  
  setMapLocked(true, 'Generating full map…');

  try {
    const r = await fetch(`${API}/generate_fullmap`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ name: currentSpecies }),
    });
    const d = await r.json();
    if (!r.ok) {
      toast(d.detail || 'Failed', true);
      setMapLocked(false);
      btn.classList.remove('running');
      btn.disabled = false;
      icon.style.animation = '';
      label.textContent = 'Generate Map';
      return;
    }

    if (d.status === 'cached') {
      log(`Full map already cached — ${currentSpecies}`, '#3498db');
      tileCacheBuster = Date.now();
      if (sdmLayer) {
        sdmLayer.setUrl(buildTileUrl(currentSpecies));
        sdmLayer.redraw();
      }
      btn.classList.remove('running');
      btn.disabled = false;
      icon.style.animation = '';
      label.textContent = 'Generate Map';
      setMapLocked(false);
      
      await refreshLists();
      setActiveBar(currentSpecies);
      return;
    }

    log(`Full map started — ${FULLMAP_GRID_N}×${FULLMAP_GRID_N} grid`, '#2ecc71');

    // Make sure tile layer is on map
    if (!sdmLayer) {
      sdmLayer = createSDMLayer(currentSpecies);
      sdmLayer.addTo(map);
    }

    // Poll /fullmap_progress — updates progress bar and redraws tiles as rows arrive
    showProgress(0, 'Computing full map…');
    setActiveDotRunning(true);

    const pollId = setInterval(async () => {
      try {
        const p = await fetch(
          `${API}/fullmap_progress/${encodeURIComponent(currentSpecies)}`
        ).then(r => r.json());

        const pct = p.pct ?? 0;
        showProgress(pct, `Full map: ${p.done ?? 0}/${p.total ?? FULLMAP_GRID_N} rows · ${pct}%`);
        label.textContent = `${Math.round(pct)}%`;

        // Redraw tiles so new rows appear immediately
        if (sdmLayer) sdmLayer.redraw();

        if (p.status === 'done') {
          clearInterval(pollId);
          setActiveDotRunning(false);
          showProgress(100, '✓ Full map ready');
          setTimeout(hideProgress, 2000);
          
          tileCacheBuster = Date.now(); // Bust cache to force reload
          if (sdmLayer) sdmLayer.setUrl(buildTileUrl(currentSpecies));
          setMapLocked(false); // Unlock the map!
          
          log(`✓ Full map complete — ${currentSpecies}`, '#2ecc71');
          await refreshLists();
          setActiveBar(currentSpecies);
          initThresholdPanel(currentSpecies);

          btn.classList.remove('running');
          btn.disabled = false;
          icon.style.animation = '';
          label.textContent = 'Generate Map';
        } else if (p.status === 'error') {
          clearInterval(pollId);
          setActiveDotRunning(false);
          showProgress(0, '❌ Full map failed');
          setTimeout(hideProgress, 2000);
          setMapLocked(false); // Unlock the map!
          log(`❌ Full map failed — ${currentSpecies}`, '#e74c3c');
          
          btn.classList.remove('running');
          btn.disabled = false;
          icon.style.animation = '';
          label.textContent = 'Generate Map';
        }
      } catch(e) {}
    }, 1200);

  } catch(e) {
    toast('Network error', true);
    log(`Full map error: ${e}`, '#e74c3c');
    btn.classList.remove('running');
    btn.disabled = false;
    icon.style.animation = '';
    label.textContent = 'Generate Map';
    setMapLocked(false);
  }
}

document.getElementById('fullmap-btn').addEventListener('click', generateFullmap);


// ── Description auto-generate ────────────────────────────────────────────────
document.getElementById('btn-gen-desc').addEventListener('click', async () => {
  const name = document.getElementById('new-name').value.trim();
  if (!name) {
    const feedback = document.getElementById('add-feedback');
    feedback.textContent = 'Please enter a species name first.';
    feedback.style.color = '#e74c3c';
    return;
  }

  const btn = document.getElementById('btn-gen-desc');
  btn.textContent = '…'; btn.disabled = true;

  try {
    const r = await fetch(`${API}/species/generate_description`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    if (r.ok) {
      document.getElementById('new-desc').value = (await r.json()).description;
    } else {
      toast('Could not generate description', true);
    }
  } catch(e) { toast('Server error', true); }
  finally { btn.textContent = '↻ auto-generate'; btn.disabled = false; }
});

// ── Register new species ──────────────────────────────────────────────────────
document.getElementById('btn-add').addEventListener('click', async () => {
  const name = document.getElementById('new-name').value.trim();
  const desc = document.getElementById('new-desc').value.trim() || undefined;
  if (!name) {
    const feedback = document.getElementById('add-feedback');
    feedback.textContent = 'Species name is required.';
    feedback.style.color = '#e74c3c';
    return;
  }

  const btn      = document.getElementById('btn-add');
  const feedback = document.getElementById('add-feedback');
  btn.disabled = true; btn.textContent = '⟳ Registering…';
  feedback.textContent = '';

  try {
    const r = await fetch(`${API}/species/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description: desc }),
    });
    const data = await r.json();

    if (!r.ok) {
      toast(data.detail || 'Registration failed', true);
      feedback.textContent = data.detail || 'Error';
      feedback.style.color = '#e74c3c';
      return;
    }

    toast(data.already_existed ? `${name} already registered` : `✓ ${name} registered`);
    log(`New species: ${name}`, '#2ecc71');

    await refreshLists();
    await loadSpecies(data.name);
    hideDrawer();
    generateFullmap();

  } catch(e) {
    toast('Network error', true);
    log(`Error: ${e}`, '#e74c3c');
  } finally {
    btn.disabled = false; btn.textContent = 'Register & generate map';
  }
});

// ── Log Console Toggle ────────────────────────────────────────────────────────
document.getElementById('log-header').addEventListener('click', () => {
  const logEl = document.getElementById('log');
  const toggleBtn = document.getElementById('log-toggle');
  const isCollapsed = logEl.classList.toggle('collapsed');
  toggleBtn.textContent = isCollapsed ? '▲' : '▼';
});

// ── Species Details Card ──────────────────────────────────────────────────────
let currentCardSpecies = null;

function isSpeciesCached(name) {
  if (!name) return false;
  return _allCached.some(sp => sp.name.toLowerCase() === name.toLowerCase() && sp.tile_count > 0);
}

async function fetchBackendDescription(speciesName) {
  const r = await fetch(`${API}/species/generate_description`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: speciesName }),
  });
  if (!r.ok) {
    throw new Error('Backend failed to generate description');
  }
  const data = await r.json();
  return data.description;
}

async function fetchSpeciesDetails(speciesName) {
  const formattedName = speciesName.replace(/ /g, '_');
  const wikiApiUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(formattedName)}`;

  try {
    const response = await fetch(wikiApiUrl);
    if (!response.ok) {
      throw new Error(`Wikipedia returned status ${response.status}`);
    }
    const data = await response.json();
    return {
      title: speciesName,
      commonName: data.title !== speciesName ? data.title : (data.description || ''),
      description: data.extract || '',
      imageUrl: data.thumbnail ? data.thumbnail.source : null,
      wikiUrl: data.content_urls ? data.content_urls.desktop.page : null
    };
  } catch (e) {
    console.warn(`Wikipedia API failed for ${speciesName}, falling back to backend:`, e);
    try {
      const fallbackDesc = await fetchBackendDescription(speciesName);
      return {
        title: speciesName,
        commonName: '',
        description: fallbackDesc,
        imageUrl: null,
        wikiUrl: null
      };
    } catch (err) {
      return {
        title: speciesName,
        commonName: '',
        description: `${speciesName} is a species with specific habitat preferences. Detailed ecological data is unavailable.`,
        imageUrl: null,
        wikiUrl: null
      };
    }
  }
}

function renderCardSkeleton() {
  const titleEl = document.getElementById('species-card-title');
  const commonEl = document.getElementById('species-card-common');
  const descEl = document.getElementById('species-card-desc');
  const imgEl = document.getElementById('species-img');
  const imgPlaceholder = document.getElementById('species-img-placeholder');
  const metaEl = document.getElementById('species-card-meta');

  imgEl.hidden = true;
  imgPlaceholder.style.display = 'flex';
  
  titleEl.innerHTML = '<div class="skeleton" style="width: 75%; height: 20px; margin-bottom: 8px;"></div>';
  commonEl.innerHTML = '<div class="skeleton" style="width: 45%; height: 14px; margin-bottom: 12px;"></div>';
  commonEl.style.display = 'block';
  
  descEl.innerHTML = `
    <div class="skeleton" style="width: 100%; height: 13px; margin-bottom: 6px;"></div>
    <div class="skeleton" style="width: 95%; height: 13px; margin-bottom: 6px;"></div>
    <div class="skeleton" style="width: 90%; height: 13px; margin-bottom: 6px;"></div>
    <div class="skeleton" style="width: 65%; height: 13px;"></div>
  `;
  metaEl.innerHTML = '';

  const cohortEl = document.getElementById('species-card-cohort');
  const cohortListEl = document.getElementById('species-card-cohort-list');
  if (cohortEl && cohortListEl) {
    cohortEl.style.display = 'none';
    cohortListEl.innerHTML = '';
  }
}

async function updateSpeciesCard(speciesName) {
  const card = document.getElementById('species-card');
  if (!card) return;

  currentCardSpecies = speciesName;
  card.hidden = false;
  card.classList.remove('species-card-collapsed');
  
  const toggleBtn = document.getElementById('species-card-toggle');
  if (toggleBtn) toggleBtn.textContent = '◀';

  renderCardSkeleton();

  const details = await fetchSpeciesDetails(speciesName);
  
  if (currentCardSpecies !== speciesName) return;

  const titleEl = document.getElementById('species-card-title');
  const commonEl = document.getElementById('species-card-common');
  const descEl = document.getElementById('species-card-desc');
  const imgEl = document.getElementById('species-img');
  const imgPlaceholder = document.getElementById('species-img-placeholder');
  const metaEl = document.getElementById('species-card-meta');

  titleEl.innerHTML = `<i>${details.title}</i>`;
  
  if (details.commonName) {
    commonEl.textContent = details.commonName;
    commonEl.style.display = 'block';
  } else {
    commonEl.style.display = 'none';
  }

  descEl.textContent = details.description;
  
  if (details.imageUrl) {
    imgEl.src = details.imageUrl;
    imgEl.hidden = false;
    imgPlaceholder.style.display = 'none';
  } else {
    imgEl.hidden = true;
    imgPlaceholder.style.display = 'flex';
  }

  let metaHtml = '';
  if (details.wikiUrl) {
    metaHtml += `<a href="${details.wikiUrl}" target="_blank" class="card-link">Learn more on Wikipedia ↗</a>`;
  }
  metaEl.innerHTML = metaHtml;

  // Fetch and display JSDM cohort associates
  try {
    const cohortData = await fetch(`${API}/species/cohort/${encodeURIComponent(speciesName)}`).then(r => r.json());
    if (currentCardSpecies === speciesName && cohortData.associates && cohortData.associates.length > 0) {
      const cohortEl = document.getElementById('species-card-cohort');
      const cohortListEl = document.getElementById('species-card-cohort-list');
      if (cohortEl && cohortListEl) {
        cohortListEl.innerHTML = cohortData.associates
          .map(name => `<button class="cohort-tag" data-species="${name}" title="Click to view ${name}">${name}</button>`)
          .join('');
        cohortEl.style.display = 'flex';
        
        // Add click listener for navigation
        cohortListEl.querySelectorAll('.cohort-tag').forEach(btn => {
          btn.addEventListener('click', () => {
            const sp = btn.getAttribute('data-species');
            if (sp) loadSpecies(sp);
          });
        });
      }
    }
  } catch (e) {
    console.warn(`Failed to fetch cohort associates for ${speciesName}:`, e);
  }
}

// Wire up card toggle button
const cardToggle = document.getElementById('species-card-toggle');
if (cardToggle) {
  cardToggle.addEventListener('click', () => {
    const card = document.getElementById('species-card');
    const isCollapsed = card.classList.toggle('species-card-collapsed');
    cardToggle.textContent = isCollapsed ? '▶' : '◀';
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async () => {
  applyTheme(currentTheme);
  try {
    const d = await fetch(`${API}/health`).then(r => r.json());
    log(`Server ready · ${d.species_count} species`);
  } catch(e) {
    log(`Cannot reach server at ${API}`, '#e74c3c');
  }
  await refreshLists();
})();