"""Server tests with FastAPI's TestClient (no real HTTP, but the real CLIs run).

Run from project/:  make test
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, 'python'))
import server  # noqa: E402
from bench import make_moons  # noqa: E402

client = TestClient(server.app)


def small(**kw):
    body = {'data': make_moons(40), 'hidden': [4], 'activation': 'tanh', 'epochs': 30, 'lr': 0.1}
    body.update(kw)
    return body


def test_health_and_index():
    r = client.get('/api/health')
    assert r.status_code == 200 and r.json()['ok'] is True
    assert r.json()['cpp_cli_built'] is True
    r = client.get('/')
    assert r.status_code == 200 and 'micrograd' in r.text


def test_limits():
    lim = client.get('/api/limits').json()
    assert lim['max_points'] == 400 and lim['max_epochs'] == 2000
    assert lim['max_layers'] == 5 and lim['max_neurons'] == 32 and lim['max_frames'] == 100
    assert lim['py_timeout_s'] == 20 and lim['cpp_timeout_s'] == 10
    assert lim['py_seconds_per_unit'] == server.PY_SECONDS_PER_UNIT
    assert 0 < lim['lr_default'] <= lim['lr_max']
    d = lim['defaults']
    # the suggested UI defaults must be cheap for Python (about 3 s per run)
    params = server.param_count([2] + d['hidden'] + [1])
    assert server.est_python_seconds(d['points'], d['epochs'], params) <= 3.0


@pytest.mark.parametrize('field, value, expect', [
    ('data', make_moons(401), 'data'),
    ('data', [], 'data'),
    ('data', [[0.0, 0.0, 0]], 'labels must be -1 or 1'),
    ('data', [[0.0, 0.0]], 'must be [x, y, label]'),
    ('epochs', 2001, 'epochs'),
    ('epochs', 0, 'epochs'),
    ('hidden', [8, 8, 8, 8, 8], 'hidden'),
    ('hidden', [33], '1..32 neurons'),
    ('hidden', [], 'hidden'),
    ('lr', 0.5, 'lr'),
    ('lr', 0, 'lr'),
    ('activation', 'softmax', 'activation'),
    ('grid', {'nx': 61, 'ny': 10}, 'grid.nx'),
    ('grid', {'xmin': 1, 'xmax': 1}, 'xmin < xmax'),
    ('surprise', 1, 'surprise'),   # unknown fields are rejected, so typos don't pass silently
])
def test_train_validation(field, value, expect):
    r = client.post('/api/train', json=small(**{field: value}))
    assert r.status_code == 422
    body = r.json()
    assert body['ok'] is False and expect in body['error']


def frame_epochs(epochs, every):
    """Same rule as the CLIs: epoch 1, every `every` epochs, and the last epoch."""
    return [e for e in range(1, epochs + 1) if e == 1 or e % every == 0 or e == epochs]


def test_snapshot_every_keeps_frames_under_limit():
    for epochs in range(1, server.MAX_EPOCHS + 1):
        n = len(frame_epochs(epochs, server.snapshot_every_for(epochs)))
        assert n <= server.MAX_FRAMES, epochs


@pytest.mark.parametrize('epochs', [1, 7, 150, 2000])
def test_train_frames(epochs):
    r = client.post('/api/train', json=small(epochs=epochs, data=make_moons(20),
                                             grid={'nx': 12, 'ny': 10}))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['ok'] and out['impl'] == 'cpp'
    frames = out['frames']
    assert 1 <= len(frames) <= server.MAX_FRAMES
    assert [f['epoch'] for f in frames] == frame_epochs(epochs, out['snapshot_every'])
    assert frames[-1]['epoch'] == epochs
    assert all(len(f['grid']) == 12 * 10 for f in frames)
    assert len(out['loss_history']) == epochs
    assert out['lr_effective'] == pytest.approx(0.1 / 20)


def test_train_is_deterministic_for_a_seed():
    a = client.post('/api/train', json=small(seed=5)).json()
    b = client.post('/api/train', json=small(seed=5)).json()
    c = client.post('/api/train', json=small(seed=6)).json()
    assert a['loss_history'] == b['loss_history']
    assert a['loss_history'] != c['loss_history']


def test_benchmark():
    r = client.post('/api/benchmark', json=small(bench_runs=2))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['ok'] is True
    assert out['losses_match'] is True
    assert len(out['cpp']['runs']) == 2 and len(out['python']['runs']) == 2
    assert out['speedup'] > 1
    assert out['cpp']['accuracy'] == out['python']['accuracy']
    assert out['est_python_seconds'] == pytest.approx(40 * 30 * 17 * server.PY_SECONDS_PER_UNIT)
    for key in ('cpu', 'python', 'compiler', 'flags', 'note'):
        assert out['machine'][key]
    assert 'not on your device' in out['machine']['note']


def test_benchmark_over_budget_is_rejected_with_estimate():
    body = small(data=make_moons(400), hidden=[32, 32, 32, 32], epochs=2000)
    r = client.post('/api/benchmark', json=body)
    assert r.status_code == 422
    out = r.json()
    assert out['ok'] is False
    assert out['est_python_seconds'] > server.PY_BUDGET_S
    assert 'Python' in out['error']


def test_benchmark_python_timeout(monkeypatch):
    monkeypatch.setattr(server, 'PY_TIMEOUT_S', 0.01)
    r = client.post('/api/benchmark', json=small(bench_runs=1))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['ok'] is True
    assert out['python'] == {'status': 'timeout', 'timeout_s': 0.01}
    assert out['speedup'] is None
    assert out['cpp']['train_seconds_median'] > 0


def test_busy_server_returns_409():
    assert server.run_lock.acquire(blocking=False)
    try:
        assert client.post('/api/train', json=small()).status_code == 409
        assert client.post('/api/benchmark', json=small()).status_code == 409
    finally:
        server.run_lock.release()


def test_missing_cpp_cli_is_a_clear_500(monkeypatch):
    monkeypatch.setattr(server, 'CPP_CLI', ['/nonexistent/micrograd_cli'])
    r = client.post('/api/train', json=small())
    assert r.status_code == 500
    assert 'make cpp' in r.json()['error']
