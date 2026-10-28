"""FastAPI server for the micrograd C++ vs Python demo.

Runs the two CLIs (build/micrograd_cli and python/cli.py) as subprocesses on identical
JSON input, and serves the frontend from web/. See CLAUDE.md for the protocol.

Run from project/:  make run   (binds to 127.0.0.1:8000)
"""

import functools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
from typing import List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP_CLI = [os.path.join(PROJECT, 'build', 'micrograd_cli')]
PY_CLI = [sys.executable, os.path.join(PROJECT, 'python', 'cli.py')]
WEB_DIR = os.path.join(PROJECT, 'web')
CPP_FLAGS = '-std=c++17 -O2'   # keep in sync with CXXFLAGS in the Makefile

# --- Limits (exposed through GET /api/limits) ---
MAX_POINTS = 400
MAX_EPOCHS = 2000
MAX_LAYERS = 5          # including the output layer, so at most 4 hidden layers
MAX_NEURONS = 32        # per layer
MAX_FRAMES = 100
MAX_GRID = 60           # grid cells per side for /api/train frames
PY_TIMEOUT_S = 20
CPP_TIMEOUT_S = 10

# --- Learning rate ---
# The loss is a SUM over points, so a stable step size shrinks like 1/N. The public API takes a
# per-sample rate and the server passes lr / len(data) to both CLIs. Measured on 100 moons
# points, layers 8,8, 200 epochs, 5 seeds: 0.1 never diverges for tanh/relu/sigmoid;
# divergence starts at 0.5, so the cap is 0.3.
LR_DEFAULT = 0.1
LR_MAX = 0.3

# --- Python cost guard ---
# est_python_seconds = points * epochs * params * PY_SECONDS_PER_UNIT.
# Measured on an i7-10750H under WSL2 for 8 configs: 6.6e-6 .. 1.05e-5 per unit
# (100 points, 8,8, 200 epochs took 15.6 s). Using the high end so the estimate rarely
# undershoots: anything under the budget should finish well before PY_TIMEOUT_S.
PY_SECONDS_PER_UNIT = 1.0e-5
PY_BUDGET_S = 15
BENCH_RUNS = 3
MAX_BENCH_RUNS = 5
PY_REPEAT_LIMIT_S = 5   # if one Python run takes longer than this, do not repeat it

# Suggested UI settings: about 2.4 s per Python run.
DEFAULTS = {'points': 60, 'hidden': [8], 'activation': 'tanh', 'lr': LR_DEFAULT,
            'epochs': 120, 'grid': 30}

# One subprocess job at a time: a benchmark must never share the CPU with another run.
run_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class Grid(BaseModel):
    model_config = ConfigDict(extra='forbid')
    nx: int = Field(30, ge=2, le=MAX_GRID)
    ny: int = Field(30, ge=2, le=MAX_GRID)
    xmin: float = -2.0
    xmax: float = 2.0
    ymin: float = -2.0
    ymax: float = 2.0

    @model_validator(mode='after')
    def check_range(self):
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError('grid needs xmin < xmax and ymin < ymax')
        return self


class TrainSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    data: List[List[float]] = Field(min_length=1, max_length=MAX_POINTS)
    hidden: List[int] = Field(default_factory=lambda: list(DEFAULTS['hidden']),
                              min_length=1, max_length=MAX_LAYERS - 1)
    activation: Literal['tanh', 'relu', 'sigmoid'] = 'tanh'
    lr: float = Field(LR_DEFAULT, gt=0, le=LR_MAX, description='per-sample learning rate')
    epochs: int = Field(DEFAULTS['epochs'], ge=1, le=MAX_EPOCHS)
    seed: int = Field(1, ge=0, le=2**31 - 1)

    @field_validator('data')
    @classmethod
    def check_points(cls, data):
        for i, p in enumerate(data):
            if len(p) != 3:
                raise ValueError(f'point {i} must be [x, y, label]')
            if not (math.isfinite(p[0]) and math.isfinite(p[1])):
                raise ValueError(f'point {i} has a non-finite coordinate')
            if p[2] not in (-1, 1):
                raise ValueError(f'point {i} has label {p[2]}; labels must be -1 or 1')
        return data

    @field_validator('hidden')
    @classmethod
    def check_hidden(cls, hidden):
        for n in hidden:
            if n < 1 or n > MAX_NEURONS:
                raise ValueError(f'each hidden layer needs 1..{MAX_NEURONS} neurons')
        return hidden


class TrainRequest(TrainSettings):
    grid: Grid = Field(default_factory=Grid)


class BenchmarkRequest(TrainSettings):
    bench_runs: int = Field(BENCH_RUNS, ge=1, le=MAX_BENCH_RUNS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CliError(Exception):
    """A CLI could not be run or returned garbage (a server-side problem, not bad input)."""


def layer_sizes(s):
    return [2] + s.hidden + [1]


def param_count(layers):
    return sum((layers[i] + 1) * layers[i + 1] for i in range(len(layers) - 1))


def est_python_seconds(points, epochs, params):
    return points * epochs * params * PY_SECONDS_PER_UNIT


def snapshot_every_for(epochs):
    """Frames are taken at epoch 1, every k epochs, and the last epoch: at most MAX_FRAMES."""
    # at most 1 + (epochs - 1) // k + 1 frames, so (epochs - 1) // k must be <= MAX_FRAMES - 2
    return max(1, math.ceil((epochs - 1) / (MAX_FRAMES - 2)))


def cli_request(s, snapshot_every, grid=None):
    """The JSON both CLIs receive. init_weights are generated here, so C++ and Python start equal."""
    layers = layer_sizes(s)
    rng = random.Random(s.seed)
    req = {
        'activation': s.activation,
        'layers': layers,
        'epochs': s.epochs,
        'lr': s.lr / len(s.data),   # per-sample rate -> rate for the summed loss
        'data': s.data,
        'init_weights': [rng.uniform(-1, 1) for _ in range(param_count(layers))],
        'seed': s.seed,
        'snapshot_every': snapshot_every,
    }
    if grid is not None:
        req['grid'] = grid.model_dump()
    return req


def run_cli(cmd, req, timeout):
    """Run one CLI. Returns its parsed JSON. Raises subprocess.TimeoutExpired or CliError."""
    try:
        # A list argument and no shell: nothing in the request can be interpreted as a command.
        # On timeout, subprocess.run kills the child before raising TimeoutExpired.
        proc = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True,
                              timeout=timeout)
    except FileNotFoundError:
        raise CliError(f'{cmd[-1]} not found (run `make cpp` in project/)')
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        raise CliError(f'{os.path.basename(cmd[-1])} exited with {proc.returncode} '
                       f'and no JSON output: {proc.stderr.strip()[:300]}')
    return out


def error(status, message, **extra):
    return JSONResponse(status_code=status, content={'ok': False, 'error': message, **extra})


@functools.lru_cache(maxsize=1)
def machine_info():
    cpu = platform.processor() or platform.machine()
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        first = subprocess.run(['g++', '--version'], capture_output=True, text=True,
                               timeout=5).stdout.splitlines()[0]
        compiler = 'g++ ' + first.split()[-1]   # "g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0"
    except (OSError, IndexError, subprocess.TimeoutExpired):
        compiler = 'g++ (unknown version)'
    try:
        distro = platform.freedesktop_os_release()['PRETTY_NAME'] + ', '   # e.g. "Ubuntu 26.04 LTS"
    except (OSError, KeyError):
        distro = ''
    info = {
        'cpu': cpu,
        'cores': os.cpu_count(),
        'os': distro + platform.system() + ' ' + platform.release(),
        'python': platform.python_version(),
        'compiler': compiler,
        'flags': CPP_FLAGS,
        'note': 'Times are measured on the machine running the server, not on your device.',
    }
    if 'microsoft' in platform.release().lower():
        info['wsl'] = True
        info['note'] += ' The server runs under WSL2, which can be slightly slower than native Linux.'
    return info


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title='micrograd C++ vs Python')


@app.exception_handler(RequestValidationError)
def validation_error(request: Request, exc: RequestValidationError):
    # One readable line per problem, e.g. "epochs: Input should be less than or equal to 2000".
    problems = []
    for e in exc.errors():
        where = '.'.join(str(part) for part in e['loc'] if part != 'body')
        problems.append(f"{where}: {e['msg']}" if where else e['msg'])
    return error(422, '; '.join(problems), details=problems)


@app.get('/api/health')
def health():
    return {'ok': True, 'cpp_cli_built': os.path.exists(CPP_CLI[0]),
            'python': platform.python_version()}


@app.get('/api/limits')
def limits():
    return {
        'max_points': MAX_POINTS, 'max_epochs': MAX_EPOCHS, 'max_layers': MAX_LAYERS,
        'max_neurons': MAX_NEURONS, 'max_frames': MAX_FRAMES, 'max_grid': MAX_GRID,
        'py_timeout_s': PY_TIMEOUT_S, 'cpp_timeout_s': CPP_TIMEOUT_S,
        'lr_default': LR_DEFAULT, 'lr_max': LR_MAX,
        'py_seconds_per_unit': PY_SECONDS_PER_UNIT, 'py_budget_s': PY_BUDGET_S,
        'bench_runs': BENCH_RUNS, 'max_bench_runs': MAX_BENCH_RUNS,
        'defaults': DEFAULTS,
    }


@app.post('/api/train')
def train(body: TrainRequest):
    """C++ only, with frames for the animation."""
    req = cli_request(body, snapshot_every_for(body.epochs), body.grid)
    if not run_lock.acquire(blocking=False):
        return error(409, 'the server is busy with another run, try again in a moment')
    try:
        out = run_cli(CPP_CLI, req, CPP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return error(504, f'C++ training timed out after {CPP_TIMEOUT_S} s')
    except CliError as e:
        return error(500, str(e))
    finally:
        run_lock.release()

    if not out.get('ok'):
        return error(422, out.get('error', 'training failed'))   # e.g. diverged
    out['lr'] = body.lr
    out['lr_effective'] = req['lr']
    out['snapshot_every'] = req['snapshot_every']
    return out


def summarize(runs):
    """Median and all runs from a list of CLI results (all from identical input)."""
    times = [r['train_seconds'] for r in runs]
    return {'train_seconds_median': statistics.median(times), 'runs': times,
            'final_loss': runs[0]['final_loss'], 'accuracy': runs[0]['accuracy']}


@app.post('/api/benchmark')
def benchmark(body: BenchmarkRequest):
    """Run C++ and then Python (never at the same time) on identical input."""
    layers = layer_sizes(body)
    est = est_python_seconds(len(body.data), body.epochs, param_count(layers))
    if est > PY_BUDGET_S:
        return error(422, f'this would take Python about {est:.0f} s (limit {PY_BUDGET_S} s): '
                          'use fewer points, epochs or neurons',
                     est_python_seconds=est, py_budget_s=PY_BUDGET_S)

    req = cli_request(body, snapshot_every=0)
    if not run_lock.acquire(blocking=False):
        return error(409, 'the server is busy with another run, try again in a moment')
    try:
        cpp_runs = []
        for _ in range(body.bench_runs):
            out = run_cli(CPP_CLI, req, CPP_TIMEOUT_S)
            if not out.get('ok'):
                return error(422, out.get('error', 'C++ training failed'), est_python_seconds=est)
            cpp_runs.append(out)

        py_runs = []
        py_timed_out = False
        for _ in range(body.bench_runs):
            try:
                out = run_cli(PY_CLI, req, PY_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                py_timed_out = True
                break
            if not out.get('ok'):
                return error(422, out.get('error', 'Python training failed'), est_python_seconds=est)
            py_runs.append(out)
            if out['train_seconds'] > PY_REPEAT_LIMIT_S:
                break
    except subprocess.TimeoutExpired:
        return error(504, f'C++ benchmark timed out after {CPP_TIMEOUT_S} s')
    except CliError as e:
        return error(500, str(e))
    finally:
        run_lock.release()

    result = {'ok': True, 'cpp': summarize(cpp_runs)}
    if py_timed_out and not py_runs:
        result['python'] = {'status': 'timeout', 'timeout_s': PY_TIMEOUT_S}
        result['speedup'] = None
        result['losses_match'] = None
    else:
        result['python'] = summarize(py_runs)
        if py_timed_out:
            result['python']['note'] = 'a repeat run timed out; median over the finished runs'
        result['speedup'] = (result['python']['train_seconds_median']
                             / result['cpp']['train_seconds_median'])
        result['losses_match'] = math.isclose(result['cpp']['final_loss'],
                                              result['python']['final_loss'],
                                              rel_tol=1e-6, abs_tol=1e-12)
    result['est_python_seconds'] = est
    result['settings'] = {'points': len(body.data), 'layers': layers, 'activation': body.activation,
                          'epochs': body.epochs, 'lr': body.lr, 'lr_effective': req['lr'],
                          'seed': body.seed, 'bench_runs': body.bench_runs}
    result['machine'] = machine_info()
    return result


# Static files last, so /api/* routes win. html=True serves index.html for "/".
app.mount('/', StaticFiles(directory=WEB_DIR, html=True), name='web')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
