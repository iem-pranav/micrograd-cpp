"""The C++ and pure-Python CLIs must agree on identical input (CLAUDE.md, "Training contract").

Run from project/:  make test   (builds the C++ CLI first)
"""

import json
import math
import os
import random
import subprocess
import sys

import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP_CLI = [os.path.join(PROJECT, 'build', 'micrograd_cli')]
PY_CLI = [sys.executable, os.path.join(PROJECT, 'python', 'cli.py')]
CLIS = {'cpp': CPP_CLI, 'python': PY_CLI}

sys.path.insert(0, os.path.join(PROJECT, 'python'))
from micrograd import MLP, Neuron, Value  # noqa: E402


def run_cli(cmd, request):
    """Run a CLI on a request (dict, or a raw string). Returns (exit code, parsed JSON)."""
    stdin = request if isinstance(request, str) else json.dumps(request)
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=120)
    assert proc.stderr == '', proc.stderr
    return proc.returncode, json.loads(proc.stdout)


def make_moons(n, seed, noise=0.1):
    """Two interleaving half circles, labels -1/+1 (a small pure-Python version of sklearn's)."""
    rng = random.Random(seed)
    data = []
    for k in range(n):
        t = math.pi * rng.random()
        if k % 2 == 0:
            x, y, label = math.cos(t), math.sin(t), 1
        else:
            x, y, label = 1 - math.cos(t), 0.5 - math.sin(t), -1
        data.append([x - 0.5 + rng.gauss(0, noise), y - 0.25 + rng.gauss(0, noise), label])
    return data


def param_count(layers):
    return sum((layers[i] + 1) * layers[i + 1] for i in range(len(layers) - 1))


# lr is small because the loss is a sum over points, not a mean: 0.01 already blows up with 40 points.
def make_request(activation, seed, layers=(2, 8, 8, 1), epochs=60, lr=0.002, n=40, **extra):
    layers = list(layers)
    rng = random.Random(1000 + seed)
    req = {
        'activation': activation,
        'layers': layers,
        'epochs': epochs,
        'lr': lr,
        'data': make_moons(n, seed),
        'init_weights': [rng.uniform(-1, 1) for _ in range(param_count(layers))],
        'seed': seed,
        'snapshot_every': 0,
    }
    req.update(extra)
    return req


def rel_close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Python engine, checked directly
# ---------------------------------------------------------------------------

def test_notebook_neuron_example():
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    b = Value(6.8813735870195432)
    o = (x1 * w1 + x2 * w2 + b).tanh()
    o.backward()
    assert o.data == pytest.approx(0.7071067811865476, abs=1e-6)
    assert x1.grad == pytest.approx(-1.5, abs=1e-6)
    assert w1.grad == pytest.approx(1.0, abs=1e-6)
    assert x2.grad == pytest.approx(0.5, abs=1e-6)
    assert w2.grad == pytest.approx(0.0, abs=1e-6)


def test_sigmoid_does_not_overflow():
    assert Value(-1000.0).sigmoid().data == 0.0
    assert Value(1000.0).sigmoid().data == 1.0
    assert Value(0.0).sigmoid().data == 0.5


def test_parameter_order_and_last_layer_linear():
    m = MLP(2, [3, 1], 'relu')
    m.set_parameters(list(range(13)))
    assert m.layers[0].neurons[0].w[1].data == 1
    assert m.layers[0].neurons[0].b.data == 2
    assert m.layers[0].neurons[1].w[0].data == 3
    assert [p.data for p in m.parameters()] == list(range(13))
    assert m.layers[0].neurons[0].activation == 'relu'
    assert m.layers[1].neurons[0].activation == 'linear'


def test_neuron_wrong_input_size_raises():
    with pytest.raises(ValueError):
        Neuron(3)([1.0, 2.0])


# ---------------------------------------------------------------------------
# C++ vs Python, through the CLIs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('activation', ['tanh', 'relu', 'sigmoid'])
def test_cli_equivalence(activation, seed):
    req = make_request(activation, seed)
    code_c, cpp = run_cli(CPP_CLI, req)
    code_p, py = run_cli(PY_CLI, req)
    assert code_c == 0 and cpp['ok'], cpp
    assert code_p == 0 and py['ok'], py
    assert cpp['impl'] == 'cpp' and py['impl'] == 'python'

    assert cpp['param_count'] == py['param_count'] == param_count(req['layers'])
    assert len(cpp['loss_history']) == len(py['loss_history']) == req['epochs']
    for a, b in zip(cpp['loss_history'], py['loss_history']):
        assert rel_close(a, b), (a, b)
    assert rel_close(cpp['final_loss'], py['final_loss']), (cpp['final_loss'], py['final_loss'])
    assert cpp['accuracy'] == py['accuracy']
    assert cpp['frames'] == [] and py['frames'] == []
    # training must actually do something
    assert cpp['final_loss'] < cpp['loss_history'][0]


@pytest.mark.parametrize('impl', ['cpp', 'python'])
def test_frames(impl):
    grid = {'nx': 6, 'ny': 5, 'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2}
    req = make_request('tanh', 1, epochs=12, snapshot_every=5, grid=grid)
    code, out = run_cli(CLIS[impl], req)
    assert code == 0 and out['ok'], out

    # epoch 1, every 5 epochs, and the last epoch
    assert [f['epoch'] for f in out['frames']] == [1, 5, 10, 12]
    for f in out['frames']:
        assert len(f['grid']) == 6 * 5
        assert all(v == round(v, 3) for v in f['grid'])
    # intermediate frames: pre-update loss; last frame: final weights
    for f in out['frames'][:-1]:
        assert f['loss'] == out['loss_history'][f['epoch'] - 1]
    assert out['frames'][-1]['loss'] == out['final_loss']
    assert out['final_loss'] != out['loss_history'][-1]


def test_frames_match_between_cli():
    grid = {'nx': 8, 'ny': 8, 'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2}
    req = make_request('sigmoid', 2, epochs=10, snapshot_every=3, grid=grid)
    _, cpp = run_cli(CPP_CLI, req)
    _, py = run_cli(PY_CLI, req)
    assert len(cpp['frames']) == len(py['frames'])
    for fc, fp in zip(cpp['frames'], py['frames']):
        assert fc['epoch'] == fp['epoch']
        # both round to 3 decimals; allow one step of rounding difference
        assert all(abs(a - b) <= 0.0011 for a, b in zip(fc['grid'], fp['grid']))


def test_frame_grid_layout():
    # A 1-layer linear network with chosen weights: pred = x + 10*y. Checks cell centres and j=0 at ymin.
    grid = {'nx': 2, 'ny': 2, 'xmin': 0, 'xmax': 2, 'ymin': 0, 'ymax': 2}
    req = make_request('tanh', 1, layers=[2, 1], epochs=1, lr=0.0, init_weights=[1.0, 10.0, 0.0],
                       snapshot_every=1, grid=grid)
    for cmd in CLIS.values():
        _, out = run_cli(cmd, req)
        # cells centred at x in (0.5, 1.5), y in (0.5, 1.5); index j*nx + i
        assert out['frames'][-1]['grid'] == [5.5, 6.5, 15.5, 16.5]


@pytest.mark.parametrize('impl', ['cpp', 'python'])
def test_seed_only_run(impl):
    req = make_request('tanh', 4, epochs=5)
    del req['init_weights']
    code, out = run_cli(CLIS[impl], req)
    assert code == 0 and out['ok'], out
    # same seed twice gives the same result (the two implementations differ from each other)
    _, again = run_cli(CLIS[impl], req)
    assert out['loss_history'] == again['loss_history']


@pytest.mark.parametrize('impl', ['cpp', 'python'])
def test_divergence_is_reported(impl):
    req = make_request('relu', 1, lr=5.0, epochs=50)
    code, out = run_cli(CLIS[impl], req)
    assert code != 0
    assert out['ok'] is False
    assert out['error'].startswith('training diverged at epoch ')
    assert out['error'].endswith(', try a lower learning rate')


def bad_requests():
    good = make_request('tanh', 1, epochs=2)

    def changed(**kw):
        req = dict(good)
        req.update(kw)
        return req

    no_lr = dict(good)
    del no_lr['lr']
    return {
        'not json': '{"activation": ',
        'not an object': '[1, 2, 3]',
        'bad activation': changed(activation='softmax'),
        'layers[0] != 2': changed(layers=[3, 8, 1]),
        'layers[-1] != 1': changed(layers=[2, 8, 2]),
        'layers too short': changed(layers=[2]),
        'zero-size layer': changed(layers=[2, 0, 1]),
        'too many points': changed(data=make_moons(401, 1)),
        'no points': changed(data=[]),
        'bad label': changed(data=[[0.0, 0.0, 0]]),
        'bad point': changed(data=[[0.0, 0.0]]),
        'init_weights too short': changed(init_weights=[0.1, 0.2]),
        'missing lr': no_lr,
        'epochs is a float': changed(epochs=2.5),
        'epochs is a bool': changed(epochs=True),
        'frames without grid': changed(snapshot_every=1),
        'empty grid range': changed(snapshot_every=1, grid={'nx': 2, 'ny': 2, 'xmin': 1, 'xmax': 1,
                                                            'ymin': 0, 'ymax': 1}),
    }


@pytest.mark.parametrize('impl', ['cpp', 'python'])
@pytest.mark.parametrize('case', list(bad_requests()))
def test_bad_request_is_rejected(impl, case):
    code, out = run_cli(CLIS[impl], bad_requests()[case])
    assert code != 0
    assert out['ok'] is False
    assert isinstance(out['error'], str) and out['error']
