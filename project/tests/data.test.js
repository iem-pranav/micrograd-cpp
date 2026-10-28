// Tests for web/data.js. Run from project/:  node --test tests/
// (`make test` runs them when node is installed and skips them otherwise.)

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const MG = require('../web/data.js');

// Shaped like GET /api/limits.
const limits = {
  max_points: 400, max_epochs: 2000, max_layers: 5, max_neurons: 32,
  lr_max: 0.3, py_seconds_per_unit: 1e-5, py_budget_s: 15,
};

test('presets: right count, labels, inside [-2, 2], deterministic per seed', () => {
  for (const name of ['moons', 'circles', 'xor']) {
    const pts = MG.makePreset(name, 60, 0.1, 7);
    assert.equal(pts.length, 60);
    for (const [x, y, label] of pts) {
      assert.ok(x >= -2 && x <= 2 && y >= -2 && y <= 2, `${name}: (${x}, ${y}) outside`);
      assert.ok(label === 1 || label === -1);
    }
    assert.ok(pts.some((p) => p[2] === 1) && pts.some((p) => p[2] === -1), `${name}: both classes`);
    assert.deepEqual(MG.makePreset(name, 60, 0.1, 7), pts);
    assert.notDeepEqual(MG.makePreset(name, 60, 0.1, 8), pts);
  }
  // even huge noise stays inside the data space
  for (const [x, y] of MG.makePreset('moons', 200, 5, 1)) assert.ok(Math.abs(x) <= 2 && Math.abs(y) <= 2);
  assert.throws(() => MG.makePreset('spirals', 10, 0, 1));
});

test('CSV: header, comments, blank lines, 0/1 and -1/1 labels', () => {
  const r = MG.parseCSV('x,y,label\n# comment\n\n0.5, 1.2, 1\n-0.8,-0.3,0\r\n1,1,-1\n', 400);
  assert.equal(r.ok, true);
  assert.deepEqual(r.points, [[0.5, 1.2, 1], [-0.8, -0.3, -1], [1, 1, -1]]);
});

test('CSV: errors name the line number', () => {
  const cases = [
    ['0,0,1\n1,2\n', /^line 2: expected 3 values/],
    ['0,0,1\n0,0,1\nfoo,0,1\n', /^line 3: .* not three numbers/],
    ['1,abc,1\n', /^line 1: /],                 // a bad first row is not mistaken for a header
    ['0,0,2\n', /^line 1: label must be 0, 1 or -1/],
    ['\n\n3,0,1\n', /^line 3: point \(3, 0\) is outside/],
    ['0,0,1,5\n', /^line 1: expected 3 values/],
    ['x,y,label\n', /no data rows/],
  ];
  for (const [text, re] of cases) {
    const r = MG.parseCSV(text, 400);
    assert.equal(r.ok, false, text);
    assert.match(r.error, re, text);
  }
});

test('CSV: points limit is enforced', () => {
  const rows = Array.from({ length: 5 }, () => '0,0,1').join('\n');
  assert.equal(MG.parseCSV(rows, 5).ok, true);
  const r = MG.parseCSV(rows + '\n0,0,0', 5);
  assert.equal(r.ok, false);
  assert.match(r.error, /too many points/);
});

test('CSV: export round-trips', () => {
  const pts = MG.makePreset('xor', 30, 0.2, 3);
  const r = MG.parseCSV(MG.toCSV(pts), 400);
  assert.equal(r.ok, true);
  assert.deepEqual(r.points, pts);
});

test('sample data file parses: 60 points, both classes', () => {
  const text = fs.readFileSync(path.join(__dirname, '..', 'data', 'moons_sample.csv'), 'utf8');
  const r = MG.parseCSV(text, limits.max_points);
  assert.equal(r.ok, true, r.error);
  assert.equal(r.points.length, 60);
  assert.equal(r.points.filter((p) => p[2] === 1).length, 30);
});

test('hidden layers parsing uses the limits', () => {
  assert.deepEqual(MG.parseHidden('8', limits), { ok: true, hidden: [8] });
  assert.deepEqual(MG.parseHidden(' 8, 4 ', limits), { ok: true, hidden: [8, 4] });
  for (const bad of ['', '8,', '0', '33', '8,8,8,8,8', '2.5', 'abc', '-1']) {
    assert.equal(MG.parseHidden(bad, limits).ok, false, bad);
  }
  assert.equal(MG.parseHidden('32,32,32,32', limits).ok, true);
});

test('settings validation', () => {
  const good = { hidden: '8', activation: 'tanh', lr: '0.1', epochs: '120', seed: '1' };
  const r = MG.readSettings(good, limits);
  assert.equal(r.ok, true);
  assert.deepEqual(r.settings, { hidden: [8], activation: 'tanh', lr: 0.1, epochs: 120, seed: 1 });
  const bad = [
    { lr: '0' }, { lr: '0.31' }, { lr: 'x' }, { epochs: '0' }, { epochs: '2001' }, { epochs: '1.5' },
    { seed: '-1' }, { activation: 'softmax' }, { hidden: '40' },
  ];
  for (const change of bad) {
    assert.equal(MG.readSettings({ ...good, ...change }, limits).ok, false, JSON.stringify(change));
  }
  assert.equal(MG.checkPoints([], limits) !== null, true);
  assert.equal(MG.checkPoints([[0, 0, 1]], limits), null);
});

test('param count and Python estimate match the server formula', () => {
  assert.equal(MG.paramCount([8]), 33);
  assert.equal(MG.paramCount([8, 8]), 105);
  // the Phase 2 measurement point: 100 points, 8,8, 200 epochs
  assert.ok(Math.abs(MG.estPythonSeconds(100, 200, [8, 8], limits) - 21) < 1e-9);
  // the UI defaults: about 2.4 s
  assert.ok(Math.abs(MG.estPythonSeconds(60, 120, [8], limits) - 2.376) < 1e-9);
});

test('request bodies', () => {
  const s = { hidden: [8], activation: 'relu', lr: 0.1, epochs: 50, seed: 3 };
  const pts = [[0, 0, 1]];
  const t = MG.buildTrainBody(pts, s, 30);
  assert.deepEqual(t.grid, { nx: 30, ny: 30, xmin: -2, xmax: 2, ymin: -2, ymax: 2 });
  assert.equal(t.lr, 0.1);
  const b = MG.buildBenchmarkBody(pts, s);
  assert.equal(b.grid, undefined);
  assert.deepEqual(Object.keys(b).sort(), ['activation', 'data', 'epochs', 'hidden', 'lr', 'seed']);
});

test('colour scale: white at 0, blue for +, orange for -, saturates, handles NaN', () => {
  assert.deepEqual(MG.predColor(0), [255, 255, 255]);
  const pos = MG.predColor(1), neg = MG.predColor(-1);
  assert.ok(pos[2] > pos[0], 'positive is bluish');
  assert.ok(neg[0] > neg[2], 'negative is orangeish');
  assert.deepEqual(MG.predColor(5), pos);
  assert.deepEqual(MG.predColor(-5), neg);
  assert.deepEqual(MG.predColor(NaN), [255, 255, 255]);
  // matches the legend gradient in style.css
  assert.deepEqual(pos, [115, 177, 213]);
  assert.deepEqual(neg, [241, 202, 115]);
});

test('playback frame index', () => {
  assert.equal(MG.frameAt(0, 61, 5000), 0);
  assert.equal(MG.frameAt(2500, 61, 5000), 30);
  assert.equal(MG.frameAt(5000, 61, 5000), 60);
  assert.equal(MG.frameAt(9999, 61, 5000), 60);
  assert.equal(MG.frameAt(-5, 61, 5000), 0);
  assert.equal(MG.frameAt(100, 1, 5000), 0);
});

test('pixel <-> data mapping: y grows upward, corners, clamping', () => {
  assert.deepEqual(MG.toPixel(-2, 2, 400), [0, 0]);
  assert.deepEqual(MG.toPixel(2, -2, 400), [400, 400]);
  assert.deepEqual(MG.toPixel(0, 0, 400), [200, 200]);
  assert.deepEqual(MG.toData(0, 0, 400), [-2, 2]);
  assert.deepEqual(MG.toData(300, 100, 400), [1, 1]);
  assert.deepEqual(MG.toData(-10, 500, 400), [-2, -2]);
});

test('formatSeconds', () => {
  assert.equal(MG.formatSeconds(1.53), '1.53 s');
  assert.equal(MG.formatSeconds(0.126), '126.0 ms');
  assert.equal(MG.formatSeconds(0.0042), '4.20 ms');
  assert.equal(MG.formatSeconds(12.34), '12.3 s');
});
