// Page wiring: DOM, canvas drawing, playback and calls to the server.
// The pure logic (presets, CSV, validation, colours, request bodies) lives in data.js as `MG`.

(function () {
  'use strict';

  const $ = function (id) { return document.getElementById(id); };
  const PLAYBACK_MS = 5000;   // whole animation, whatever the number of frames
  const UNDO_LIMIT = 50;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  let limits = null;        // from GET /api/limits
  let points = [];          // [[x, y, label], ...]
  let undoStack = [];       // earlier versions of `points`
  let dataVersion = 0;      // bumped on every data change
  let result = null;        // last /api/train response
  let trainedVersion = -1;  // dataVersion that `result` was trained on
  let frameIdx = 0;
  let playing = false;
  let playStart = 0;
  let busy = false;
  let pointerDown = null;   // where a click on the plot started

  // ---------------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------------

  function setStatus(message, isError) {
    const el = $('status');
    // Messages often end with a server error plus '.', so "running?." -> "running?".
    el.textContent = message.replace(/([.!?])\.$/, '$1');
    el.classList.toggle('error', !!isError);
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // Size a canvas's backing store for the screen's pixel density; drawing uses CSS pixels.
  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: rect.width, h: rect.height };
  }

  function formatLoss(v) {
    if (!Number.isFinite(v)) return String(v);
    if (v !== 0 && (Math.abs(v) < 0.01 || Math.abs(v) >= 1e5)) return v.toExponential(2);
    return v.toPrecision(4);
  }

  function pct(v) { return Math.round(v * 100) + '%'; }

  // POST (or GET without a body). Never throws: returns {ok, data} or {ok: false, status, error}.
  async function api(path, body) {
    const init = body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    };
    let res;
    try {
      res = await fetch(path, init);
    } catch (e) {
      return { ok: false, status: 0, error: 'Could not reach the server. Is it still running?' };
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* not JSON */ }
    if (res.ok && data && data.ok !== false) return { ok: true, data: data };
    const message = data && data.error ? data.error : 'server error (HTTP ' + res.status + ')';
    return { ok: false, status: res.status, error: message, data: data };
  }

  // ---------------------------------------------------------------------------
  // Data
  // ---------------------------------------------------------------------------

  function setPoints(next) {
    undoStack.push(points);
    if (undoStack.length > UNDO_LIMIT) undoStack.shift();
    points = next;
    dataChanged();
  }

  function dataChanged() {
    dataVersion++;
    $('undo').disabled = undoStack.length === 0;
    const pos = points.filter(function (p) { return p[2] === 1; }).length;
    $('count').textContent = points.length + ' points (' + pos + ' blue, ' + (points.length - pos) +
      ' orange) · max ' + limits.max_points;
    updateEstimate();
    render();
  }

  function generate() {
    const n = Number($('npoints').value);
    if (!Number.isInteger(n) || n < 2 || n > limits.max_points) {
      setStatus('Points must be a whole number from 2 to ' + limits.max_points + '.', true);
      return;
    }
    const seed = Number($('dseed').value);
    if (!Number.isInteger(seed) || seed < 0) {
      setStatus('Data seed must be a whole number, 0 or more.', true);
      return;
    }
    setPoints(MG.makePreset($('preset').value, n, Number($('noise').value), seed));
    setStatus('Generated ' + n + ' ' + $('preset').selectedOptions[0].text.toLowerCase() + ' points.');
  }

  function loadCSV(text, source) {
    const r = MG.parseCSV(text, limits.max_points);
    if (!r.ok) { setStatus('CSV ' + source + ': ' + r.error, true); return; }
    setPoints(r.points);
    setStatus('Loaded ' + r.points.length + ' points from ' + source + '.');
  }

  function exportCSV() {
    const blob = new Blob([MG.toCSV(points)], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'micrograd_data.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
    setStatus('Exported ' + points.length + ' points as micrograd_data.csv.');
  }

  // A click (or tap) on the plot adds a point. A drag or a scroll gesture does not.
  function onPointerDown(e) {
    pointerDown = { id: e.pointerId, x: e.clientX, y: e.clientY };
  }

  function onPointerUp(e) {
    const start = pointerDown;
    pointerDown = null;
    if (!start || start.id !== e.pointerId || !limits) return;
    if (Math.hypot(e.clientX - start.x, e.clientY - start.y) > 8) return;
    if (points.length >= limits.max_points) {
      setStatus('Point limit reached (' + limits.max_points + ').', true);
      return;
    }
    const rect = $('plot').getBoundingClientRect();
    const xy = MG.toData(e.clientX - rect.left, e.clientY - rect.top, rect.width);
    let label = Number(document.querySelector('input[name="cls"]:checked').value);
    if (e.shiftKey) label = -label;
    setPoints(points.concat([[xy[0], xy[1], label]]));
    setStatus('Added a class ' + (label > 0 ? '+1 (blue)' : '−1 (orange)') + ' point at (' + xy[0] + ', ' + xy[1] + ').');
  }

  // ---------------------------------------------------------------------------
  // Settings and the Python-time estimate
  // ---------------------------------------------------------------------------

  function readSettings() {
    return MG.readSettings({
      hidden: $('hidden').value, activation: $('activation').value,
      lr: $('lr').value, epochs: $('epochs').value, seed: $('seed').value,
    }, limits);
  }

  function updateEstimate() {
    const el = $('estimate');
    const s = readSettings();
    let canRun = false;
    if (!s.ok) {
      el.textContent = 'Fix the settings first: ' + s.error + '.';
    } else if (MG.checkPoints(points, limits)) {
      el.textContent = MG.checkPoints(points, limits) + '.';
    } else {
      const est = MG.estPythonSeconds(points.length, s.settings.epochs, s.settings.hidden, limits);
      if (est > limits.py_budget_s) {
        el.textContent = 'Too slow for Python: about ' + MG.formatSeconds(est) + ' per run (limit ' +
          limits.py_budget_s + ' s). Use fewer points, epochs or neurons.';
      } else {
        const runs = est > 5 ? 1 : limits.bench_runs;   // the server does not repeat slow Python runs
        el.textContent = 'Python: about ' + MG.formatSeconds(est) + ' per run, ' + runs + ' run' +
          (runs > 1 ? 's' : '') + '.';
        canRun = true;
      }
    }
    el.classList.toggle('warn', !canRun);
    $('bench').disabled = busy || !canRun;
  }

  function setBusy(on) {
    busy = on;
    ['train', 'reinit'].forEach(function (id) { $(id).disabled = on; });
    updateEstimate();   // also enables/disables the Benchmark button
  }

  // ---------------------------------------------------------------------------
  // Drawing
  // ---------------------------------------------------------------------------

  const offscreen = document.createElement('canvas');

  function drawPlot() {
    const c = setupCanvas($('plot'));
    const ctx = c.ctx, size = c.w;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, size, c.h);

    if (result && result.frames.length) {
      // One pixel per grid cell, then scale up with smoothing for a soft boundary.
      const frame = result.frames[frameIdx];
      const nx = result.grid_nx, ny = result.grid_ny;
      offscreen.width = nx;
      offscreen.height = ny;
      const octx = offscreen.getContext('2d');
      const img = octx.createImageData(nx, ny);
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const rgb = MG.predColor(frame.grid[j * nx + i]);
          const row = ny - 1 - j;   // j = 0 is ymin, which is the bottom of the canvas
          const k = (row * nx + i) * 4;
          img.data[k] = rgb[0]; img.data[k + 1] = rgb[1]; img.data[k + 2] = rgb[2]; img.data[k + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(offscreen, 0, 0, size, size);
    }

    // Faint axes through (0, 0).
    const origin = MG.toPixel(0, 0, size);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(origin[0], 0); ctx.lineTo(origin[0], size);
    ctx.moveTo(0, origin[1]); ctx.lineTo(size, origin[1]);
    ctx.stroke();

    const r = size < 360 ? 4 : 5;
    const edge = r + 2;   // keep points on the border fully visible instead of half cut off
    for (const p of points) {
      const xy = MG.toPixel(p[0], p[1], size).map(function (v) { return MG.clamp(v, edge, size - edge); });
      ctx.beginPath();
      ctx.arc(xy[0], xy[1], r, 0, 2 * Math.PI);
      ctx.fillStyle = p[2] === 1 ? cssVar('--pos') : cssVar('--neg');
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(xy[0], xy[1], r + 1.2, 0, 2 * Math.PI);
      ctx.lineWidth = 0.8;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.45)';
      ctx.stroke();
    }

    if (points.length === 0) {
      ctx.fillStyle = '#5d5d6b';
      ctx.font = '14px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No points: pick a preset or click to add some', size / 2, size / 2);
    }
  }

  // "Nice" step for about `count` ticks over `range`: 1, 2 or 5 times a power of ten.
  function niceStep(range, count) {
    const raw = range / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / mag;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag;
  }

  function drawLoss() {
    const c = setupCanvas($('loss'));
    const ctx = c.ctx, W = c.w, H = c.h;
    ctx.clearRect(0, 0, W, H);
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillStyle = cssVar('--muted');
    if (!result) {
      ctx.textAlign = 'center';
      ctx.fillText('Press Train to see the loss go down', W / 2, H / 2);
      return;
    }

    const hist = result.loss_history;
    const E = hist.length;
    const log = $('logscale').checked;
    const f = log ? function (v) { return Math.log10(Math.max(v, 1e-12)); } : function (v) { return v; };
    let lo = Infinity, hi = -Infinity;
    for (const v of hist) { lo = Math.min(lo, f(v)); hi = Math.max(hi, f(v)); }
    if (log) { lo = Math.floor(lo); hi = Math.ceil(hi); if (hi === lo) hi = lo + 1; }
    else { lo = 0; if (!(hi > 0)) hi = 1; }

    const m = { l: 56, r: 12, t: 10, b: 40 };   // bottom: tick labels, then the axis title
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const X = function (epoch) { return m.l + (E > 1 ? (epoch - 1) / (E - 1) : 0.5) * pw; };
    const Y = function (v) { return m.t + (1 - (f(v) - lo) / (hi - lo)) * ph; };

    // Horizontal grid lines and y labels.
    ctx.strokeStyle = cssVar('--grid');
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yStep = log ? Math.max(1, Math.ceil((hi - lo) / 6)) : niceStep(hi - lo, 5);
    for (let t = lo; t <= hi + 1e-9; t += yStep) {
      const y = m.t + (1 - (t - lo) / (hi - lo)) * ph;
      ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(W - m.r, y); ctx.stroke();
      // String(+x.toPrecision(6)) drops float noise like 0.30000000000000004 and trailing zeros.
      ctx.fillText(log ? (t === 0 ? '1' : '1e' + t) : String(+t.toPrecision(6)), m.l - 6, y);
    }

    // x labels (epochs).
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xStep = Math.max(1, niceStep(Math.max(E - 1, 1), 5));
    for (let e = 0; e <= E; e += xStep) {
      const ep = Math.max(1, e);
      ctx.fillText(String(ep), X(ep), H - m.b + 6);
    }
    ctx.fillText('epoch', m.l + pw / 2, H - 16);

    // The curve up to the epoch of the current frame.
    const upto = Math.min(E, result.frames.length ? result.frames[frameIdx].epoch : E);
    ctx.strokeStyle = cssVar('--curve');
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let e = 1; e <= upto; e++) {
      const x = X(e), y = Y(hist[e - 1]);
      if (e === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = cssVar('--curve');
    ctx.beginPath();
    ctx.arc(X(upto), Y(hist[upto - 1]), 3.5, 0, 2 * Math.PI);
    ctx.fill();
  }

  function render() {
    if (!limits) return;
    drawPlot();
    drawLoss();
    const readout = $('readout');
    if (!result) {
      readout.textContent = 'Not trained yet: press Train and the background will show what the network predicts.';
      return;
    }
    const n = result.frames.length;
    const frame = result.frames[frameIdx];
    $('scrub').value = String(frameIdx);
    $('scrub').setAttribute('aria-valuetext', 'epoch ' + frame.epoch);
    let text = 'Epoch ' + frame.epoch + ' of ' + result.loss_history.length +
      (frameIdx === n - 1 ? ' (final weights)' : '') + ' · loss ' + formatLoss(frame.loss);
    if (trainedVersion !== dataVersion) text += ' · data changed since training: train again';
    readout.textContent = text;
  }

  // ---------------------------------------------------------------------------
  // Playback
  // ---------------------------------------------------------------------------

  function setPlaying(on) {
    playing = on;
    $('play').textContent = on ? 'Pause' : 'Play';
    $('play').setAttribute('aria-label', on ? 'Pause' : 'Play');
  }

  function play() {
    if (!result) return;
    const n = result.frames.length;
    if (frameIdx >= n - 1) frameIdx = 0;
    // Start the clock so that playback continues from the current frame.
    playStart = performance.now() - (n > 1 ? (frameIdx / (n - 1)) * PLAYBACK_MS : 0);
    setPlaying(true);
    requestAnimationFrame(tick);
  }

  function tick(now) {
    if (!playing || !result) return;
    const n = result.frames.length;
    const idx = MG.frameAt(now - playStart, n, PLAYBACK_MS);
    if (idx !== frameIdx) { frameIdx = idx; render(); }
    if (idx >= n - 1) { setPlaying(false); render(); return; }
    requestAnimationFrame(tick);
  }

  // ---------------------------------------------------------------------------
  // Train and benchmark
  // ---------------------------------------------------------------------------

  async function train() {
    const s = readSettings();
    if (!s.ok) { setStatus('Settings: ' + s.error + '.', true); return; }
    const pointsError = MG.checkPoints(points, limits);
    if (pointsError) { setStatus(pointsError + '.', true); return; }

    setPlaying(false);
    setBusy(true);
    setStatus('Training in C++…');
    const version = dataVersion;
    const grid = limits.defaults.grid;
    const r = await api('/api/train', MG.buildTrainBody(points, s.settings, grid));
    setBusy(false);
    if (!r.ok) {
      setStatus(r.status === 409 ? 'Busy: ' + r.error + '.' : 'Training failed: ' + r.error + '.', true);
      return;
    }

    result = r.data;
    result.grid_nx = grid;
    result.grid_ny = grid;
    trainedVersion = version;
    const n = result.frames.length;
    $('scrub').max = String(n - 1);
    ['play', 'restart', 'scrub'].forEach(function (id) { $(id).disabled = false; });
    $('final').textContent = 'Final loss ' + formatLoss(result.final_loss) + ' · accuracy ' +
      pct(result.accuracy) + ' · ' + result.param_count + ' parameters · trained in ' +
      MG.formatSeconds(result.train_seconds) + ' (C++)';
    setStatus('Trained ' + result.loss_history.length + ' epochs in ' + MG.formatSeconds(result.train_seconds) +
      ' (C++). Accuracy ' + pct(result.accuracy) + '.');

    if (reducedMotion.matches) {   // no autoplay: show where training ended
      frameIdx = n - 1;
      render();
    } else {
      frameIdx = 0;
      render();
      play();
    }
  }

  function reinitialize() {
    $('seed').value = String(Math.floor(Math.random() * 2147483647));
    updateEstimate();
    train();
  }

  async function benchmark() {
    const s = readSettings();
    if (!s.ok) { setStatus('Settings: ' + s.error + '.', true); return; }
    const pointsError = MG.checkPoints(points, limits);
    if (pointsError) { setStatus(pointsError + '.', true); return; }

    setBusy(true);
    $('spinner').hidden = false;
    setStatus('Benchmarking: C++ first, then Python…');
    const r = await api('/api/benchmark', MG.buildBenchmarkBody(points, s.settings));
    $('spinner').hidden = true;
    setBusy(false);

    if (!r.ok) {
      if (r.status === 409) setStatus('Another benchmark or training run is in progress: ' + r.error + '.', true);
      else setStatus('Benchmark failed: ' + r.error + '.', true);
      return;
    }
    showBenchmark(r.data);
  }

  function showBenchmark(b) {
    $('bench-empty').hidden = true;
    $('bench-result').hidden = false;
    const cpp = b.cpp.train_seconds_median;
    const barCpp = $('bar-cpp'), barPy = $('bar-py');
    const m = b.machine;
    $('machine').textContent = 'Measured on: ' + m.cpu + ' (' + m.cores + ' threads) · Python ' + m.python +
      ' · ' + m.compiler + ' ' + m.flags + (m.wsl ? ' · running under WSL2' : '');
    $('time-cpp').textContent = MG.formatSeconds(cpp);

    if (b.python.status === 'timeout') {
      const limit = b.python.timeout_s;
      barCpp.style.width = Math.max(0.5, (cpp / limit) * 100) + '%';
      barPy.style.width = '100%';
      barPy.classList.add('timeout');
      $('time-py').textContent = '> ' + limit + ' s';
      $('speedup').textContent = 'Python did not finish within ' + limit + ' s, so there is no speedup.';
      $('match').hidden = true;
      $('bench-acc').textContent = 'C++ accuracy ' + pct(b.cpp.accuracy) + '.';
      $('runs').textContent = 'C++ runs: ' + b.cpp.runs.map(MG.formatSeconds).join(', ');
      setStatus('Benchmark done: Python timed out after ' + limit + ' s.', true);
      return;
    }

    const py = b.python.train_seconds_median;
    const slower = Math.max(cpp, py);
    barPy.classList.remove('timeout');
    barCpp.style.width = Math.max(0.5, (cpp / slower) * 100) + '%';
    barPy.style.width = Math.max(0.5, (py / slower) * 100) + '%';
    $('time-py').textContent = MG.formatSeconds(py);
    $('speedup').textContent = b.speedup >= 1
      ? 'C++ was ' + b.speedup.toFixed(1) + '× faster.'
      : 'Python was ' + (1 / b.speedup).toFixed(1) + '× faster.';
    const match = $('match');
    match.hidden = false;
    match.textContent = b.losses_match ? '✓ losses match' : '✗ losses differ';
    match.className = 'badge ' + (b.losses_match ? 'ok' : 'bad');
    $('bench-acc').textContent = 'Accuracy: C++ ' + pct(b.cpp.accuracy) + ', Python ' + pct(b.python.accuracy) +
      ' · final loss: C++ ' + formatLoss(b.cpp.final_loss) + ', Python ' + formatLoss(b.python.final_loss);
    $('runs').textContent = 'Median of runs. C++: ' + b.cpp.runs.map(MG.formatSeconds).join(', ') +
      ' · Python: ' + b.python.runs.map(MG.formatSeconds).join(', ') +
      ' · ' + b.settings.points + ' points, layers ' + b.settings.layers.join('-') + ', ' +
      b.settings.activation + ', ' + b.settings.epochs + ' epochs' + (b.python.note ? ' · ' + b.python.note : '');
    setStatus('Benchmark done: ' + $('speedup').textContent);
  }

  // ---------------------------------------------------------------------------
  // Start-up
  // ---------------------------------------------------------------------------

  function applyLimits() {
    const d = limits.defaults;
    $('npoints').max = String(limits.max_points);
    $('npoints').value = String(d.points);
    $('hidden').value = d.hidden.join(',');
    $('activation').value = d.activation;
    $('lr').min = '0.01';
    $('lr').max = String(limits.lr_max);
    $('lr').value = String(d.lr);
    $('lr-help').textContent = 'Per data point, up to ' + limits.lr_max + ' (default ' + d.lr + '). ' +
      $('lr-help').textContent.replace(/^Per data point\.\s*/, '');
    $('epochs').max = String(limits.max_epochs);
    $('epochs').value = String(d.epochs);
    $('hidden-help').textContent += ' At most ' + (limits.max_layers - 1) + ' hidden layers of 1–' +
      limits.max_neurons + ' neurons.';
    $('noise-out').textContent = $('noise').value;
  }

  function wire() {
    $('train').addEventListener('click', train);
    $('reinit').addEventListener('click', reinitialize);
    $('bench').addEventListener('click', benchmark);
    $('play').addEventListener('click', function () { if (playing) setPlaying(false); else play(); });
    $('restart').addEventListener('click', function () { frameIdx = 0; render(); play(); });
    $('scrub').addEventListener('input', function () {
      setPlaying(false);
      frameIdx = Number($('scrub').value);
      render();
    });
    $('logscale').addEventListener('change', drawLoss);

    $('preset').addEventListener('change', generate);
    $('npoints').addEventListener('change', generate);
    $('dseed').addEventListener('change', generate);
    $('noise').addEventListener('input', function () { $('noise-out').textContent = $('noise').value; });
    $('noise').addEventListener('change', generate);
    $('generate').addEventListener('click', generate);
    $('undo').addEventListener('click', function () {
      if (undoStack.length === 0) return;
      points = undoStack.pop();
      dataChanged();
      setStatus('Undone. ' + points.length + ' points.');
    });
    $('clear').addEventListener('click', function () { setPoints([]); setStatus('Cleared all points.'); });

    $('csv-load').addEventListener('click', function () { loadCSV($('csv-text').value, 'pasted text'); });
    $('csv-file').addEventListener('change', function () {
      const file = $('csv-file').files[0];
      if (!file) return;
      file.text().then(function (text) { loadCSV(text, file.name); },
                       function () { setStatus('Could not read ' + file.name + '.', true); });
      $('csv-file').value = '';   // so choosing the same file again fires `change`
    });
    $('csv-export').addEventListener('click', exportCSV);

    ['hidden', 'activation', 'lr', 'epochs', 'seed'].forEach(function (id) {
      $(id).addEventListener('input', updateEstimate);
    });

    const plot = $('plot');
    plot.addEventListener('pointerdown', onPointerDown);
    plot.addEventListener('pointerup', onPointerUp);
    plot.addEventListener('pointercancel', function () { pointerDown = null; });   // e.g. a scroll took over

    new ResizeObserver(render).observe(plot);
    new ResizeObserver(drawLoss).observe($('loss'));
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
  }

  async function init() {
    const r = await api('/api/limits');
    if (!r.ok) {
      setStatus('Could not load limits from the server (' + r.error + '). Start it with `make run`, then reload.', true);
      return;
    }
    limits = r.data;
    applyLimits();
    wire();
    $('train').disabled = false;
    $('reinit').disabled = false;
    generate();
    undoStack = [];          // the first preset is not something to undo
    $('undo').disabled = true;
    setStatus('Ready. Press Train to watch the network learn.');
  }

  init();
})();
