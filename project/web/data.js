// Pure logic for the page: presets, CSV, validation, colours, request building.
// No DOM access here, so it runs in the browser (as window.MG) and in node (`node --test`).
// Every bound comes from the server's /api/limits object (`limits`), never hard-coded.

(function (root) {
  'use strict';

  // The data space. Must match the grid the page asks the server for.
  const DATA_MIN = -2;
  const DATA_MAX = 2;

  // Okabe-Ito blue / orange: a colour-blind-safe pair. Class +1 is blue, class -1 is orange.
  const BLUE = [0, 114, 178];
  const ORANGE = [230, 159, 0];

  function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
  }

  // Small seeded PRNG (mulberry32), so a preset + seed always gives the same points.
  function makeRng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Standard normal sample (Box-Muller).
  function gauss(rng) {
    const u = 1 - rng();   // in (0, 1], so log(u) is finite
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function round3(v) {
    return Math.round(v * 1000) / 1000;
  }

  // Shapes are built in a unit-sized space, then scaled by 1.3 into [-2, 2]^2.
  // `noise` is the standard deviation in that unit space (same as python/bench.py's moons).
  function makePreset(name, n, noise, seed) {
    const rng = makeRng(seed);
    const pts = [];
    for (let k = 0; k < n; k++) {
      let x, y, label;
      if (name === 'moons') {
        const t = Math.PI * rng();
        if (k % 2 === 0) { x = Math.cos(t) - 0.5; y = Math.sin(t) - 0.25; label = 1; }
        else { x = 0.5 - Math.cos(t); y = 0.25 - Math.sin(t); label = -1; }
      } else if (name === 'circles') {
        const t = 2 * Math.PI * rng();
        const r = (k % 2 === 0) ? 0.5 : 1.15;
        x = r * Math.cos(t); y = r * Math.sin(t);
        label = (k % 2 === 0) ? 1 : -1;
      } else if (name === 'xor') {
        x = (rng() * 2 - 1) * 1.4; y = (rng() * 2 - 1) * 1.4;
        label = (x * y >= 0) ? 1 : -1;
      } else {
        throw new Error('unknown preset ' + name);
      }
      x = 1.3 * (x + noise * gauss(rng));
      y = 1.3 * (y + noise * gauss(rng));
      pts.push([round3(clamp(x, DATA_MIN, DATA_MAX)), round3(clamp(y, DATA_MIN, DATA_MAX)), label]);
    }
    return pts;
  }

  // CSV lines "x,y,label". Label 0/1 or -1/1, mapped to -1/+1. Blank lines and lines starting
  // with '#' are skipped; a first line made only of words (e.g. "x,y,label") is a header.
  // Returns {ok: true, points} or {ok: false, error} where the error names the line number.
  function parseCSV(text, maxPoints) {
    const lines = String(text).split(/\r?\n/);
    const points = [];
    let sawData = false;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      const where = 'line ' + (i + 1);
      if (line === '' || line.startsWith('#')) continue;
      const parts = line.split(',').map(function (s) { return s.trim(); });
      if (!sawData && parts.every(function (s) { return s !== '' && isNaN(Number(s)); })) {
        sawData = true;   // header row, e.g. "x,y,label"
        continue;
      }
      sawData = true;
      if (parts.length !== 3) {
        return { ok: false, error: where + ': expected 3 values (x,y,label), got ' + parts.length };
      }
      const nums = parts.map(function (s) { return s === '' ? NaN : Number(s); });
      if (!nums.every(Number.isFinite)) {
        return { ok: false, error: where + ': "' + line + '" is not three numbers' };
      }
      const x = nums[0], y = nums[1], label = nums[2];
      if (x < DATA_MIN || x > DATA_MAX || y < DATA_MIN || y > DATA_MAX) {
        return { ok: false, error: where + ': point (' + x + ', ' + y + ') is outside [' +
                 DATA_MIN + ', ' + DATA_MAX + ']' };
      }
      if (label !== 0 && label !== 1 && label !== -1) {
        return { ok: false, error: where + ': label must be 0, 1 or -1, got ' + parts[2] };
      }
      points.push([x, y, label === 1 ? 1 : -1]);
      if (points.length > maxPoints) {
        return { ok: false, error: 'too many points: more than ' + maxPoints + ' (the limit)' };
      }
    }
    if (points.length === 0) return { ok: false, error: 'no data rows found' };
    return { ok: true, points: points };
  }

  function toCSV(points) {
    return 'x,y,label\n' + points.map(function (p) { return p.join(','); }).join('\n') + '\n';
  }

  // "8, 8" -> [8, 8]. Each size 1..max_neurons, at most max_layers - 1 hidden layers.
  function parseHidden(text, limits) {
    const parts = String(text).split(',').map(function (s) { return s.trim(); });
    const maxHidden = limits.max_layers - 1;
    if (parts.length === 1 && parts[0] === '') return { ok: false, error: 'hidden layers: enter at least one size, e.g. 8' };
    if (parts.length > maxHidden) return { ok: false, error: 'hidden layers: at most ' + maxHidden + ' layers' };
    const sizes = [];
    for (const s of parts) {
      if (!/^\d+$/.test(s)) return { ok: false, error: 'hidden layers: "' + s + '" is not a whole number' };
      const n = Number(s);
      if (n < 1 || n > limits.max_neurons) {
        return { ok: false, error: 'hidden layers: sizes must be 1..' + limits.max_neurons };
      }
      sizes.push(n);
    }
    return { ok: true, hidden: sizes };
  }

  function paramCount(hidden) {
    const layers = [2].concat(hidden, [1]);
    let n = 0;
    for (let i = 0; i + 1 < layers.length; i++) n += (layers[i] + 1) * layers[i + 1];
    return n;
  }

  // Same formula as the server's guard (server.py, est_python_seconds).
  function estPythonSeconds(points, epochs, hidden, limits) {
    return points * epochs * paramCount(hidden) * limits.py_seconds_per_unit;
  }

  // Raw form values (strings) -> validated settings, or the first error.
  function readSettings(form, limits) {
    const h = parseHidden(form.hidden, limits);
    if (!h.ok) return h;
    if (['tanh', 'relu', 'sigmoid'].indexOf(form.activation) < 0) {
      return { ok: false, error: 'activation must be tanh, relu or sigmoid' };
    }
    const lr = Number(form.lr);
    if (!(lr > 0 && lr <= limits.lr_max)) {
      return { ok: false, error: 'learning rate must be above 0 and at most ' + limits.lr_max };
    }
    const epochs = Number(form.epochs);
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > limits.max_epochs) {
      return { ok: false, error: 'epochs must be a whole number from 1 to ' + limits.max_epochs };
    }
    const seed = Number(form.seed);
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) {
      return { ok: false, error: 'seed must be a whole number from 0 to 2147483647' };
    }
    return { ok: true, settings: { hidden: h.hidden, activation: form.activation, lr: lr, epochs: epochs, seed: seed } };
  }

  function checkPoints(points, limits) {
    if (points.length === 0) return 'add some points first (pick a preset or click the plot)';
    if (points.length > limits.max_points) return 'too many points: ' + points.length + ' (max ' + limits.max_points + ')';
    return null;
  }

  function buildTrainBody(points, settings, gridSize) {
    return {
      data: points, hidden: settings.hidden, activation: settings.activation,
      lr: settings.lr, epochs: settings.epochs, seed: settings.seed,
      grid: { nx: gridSize, ny: gridSize, xmin: DATA_MIN, xmax: DATA_MAX, ymin: DATA_MIN, ymax: DATA_MAX },
    };
  }

  function buildBenchmarkBody(points, settings) {
    return {
      data: points, hidden: settings.hidden, activation: settings.activation,
      lr: settings.lr, epochs: settings.epochs, seed: settings.seed,
    };
  }

  // Background colour for a prediction: white at 0, towards blue for +, orange for -.
  // Saturates at |v| = 1 and stays light (55% tint) so the points on top stand out.
  function predColor(v) {
    const t = Number.isFinite(v) ? clamp(v, -1, 1) : 0;
    const c = t >= 0 ? BLUE : ORANGE;
    const a = Math.abs(t) * 0.55;
    return [0, 1, 2].map(function (i) { return Math.round(255 + (c[i] - 255) * a); });
  }

  // Playback: which frame to show `elapsedMs` into an animation of `durationMs`.
  function frameAt(elapsedMs, nFrames, durationMs) {
    if (nFrames <= 1) return 0;
    return clamp(Math.floor((elapsedMs / durationMs) * (nFrames - 1)), 0, nFrames - 1);
  }

  // Data (x, y) <-> canvas pixel (px, py) for a square plot of `size` CSS pixels. y grows upward.
  function toPixel(x, y, size) {
    return [(x - DATA_MIN) / (DATA_MAX - DATA_MIN) * size, (DATA_MAX - y) / (DATA_MAX - DATA_MIN) * size];
  }

  function toData(px, py, size) {
    const x = DATA_MIN + (px / size) * (DATA_MAX - DATA_MIN);
    const y = DATA_MAX - (py / size) * (DATA_MAX - DATA_MIN);
    return [round3(clamp(x, DATA_MIN, DATA_MAX)), round3(clamp(y, DATA_MIN, DATA_MAX))];
  }

  // Short human-readable duration.
  function formatSeconds(s) {
    if (s < 0.001) return (s * 1e6).toFixed(0) + ' µs';
    if (s < 1) return (s * 1000).toFixed(s < 0.01 ? 2 : 1) + ' ms';
    return s.toFixed(s < 10 ? 2 : 1) + ' s';
  }

  const api = {
    DATA_MIN, DATA_MAX, BLUE, ORANGE,
    clamp, makeRng, makePreset, parseCSV, toCSV, parseHidden, paramCount, estPythonSeconds,
    readSettings, checkPoints, buildTrainBody, buildBenchmarkBody, predColor, frameAt,
    toPixel, toData, formatSeconds,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;   // node
  else root.MG = api;                                                          // browser
})(this);
