"""Pure-Python CLI with the same protocol as the C++ micrograd_cli.

Reads one JSON request on stdin, writes one JSON response on stdout.
See CLAUDE.md ("CLI protocol"). Only the training loop is timed.
"""

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from micrograd import MLP  # noqa: E402

MAX_POINTS = 400  # backward() is recursive: keep the graph shallow

# The loss is a chain of `+` per data point, so build_topo recurses roughly
# (points + network depth) levels deep: above Python's default limit of 1000.
sys.setrecursionlimit(20000)

ACTIVATIONS = ('tanh', 'relu', 'sigmoid')


class Diverged(Exception):
    def __init__(self, epoch):
        super().__init__(f"training diverged at epoch {epoch}, try a lower learning rate")


def field(obj, name):
    if name not in obj:
        raise ValueError(f"missing field '{name}'")
    return obj[name]


def get_number(v, name):
    # bool is a subclass of int in Python, so exclude it explicitly
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"'{name}' must be a number")
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"'{name}' must be finite")
    return v


def get_int(v, name, min_value):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"'{name}' must be an integer")
    if v < min_value:
        raise ValueError(f"'{name}' must be >= {min_value}")
    if v > 1_000_000_000:
        raise ValueError(f"'{name}' is too large")
    return v


def parse_request(req):
    """Validate the request and return a plain dict with defaults filled in."""
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    r = {}

    act = field(req, 'activation')
    if not isinstance(act, str):
        raise ValueError("'activation' must be a string")
    if act not in ACTIVATIONS:
        raise ValueError("'activation' must be one of tanh, relu, sigmoid")
    r['activation'] = act

    layers = field(req, 'layers')
    if not isinstance(layers, list) or len(layers) < 2:
        raise ValueError("'layers' must be an array with at least 2 sizes")
    r['layers'] = [get_int(n, 'layers[i]', 1) for n in layers]
    if r['layers'][0] != 2:
        raise ValueError("'layers' must start with 2 (the data is 2D)")
    if r['layers'][-1] != 1:
        raise ValueError("'layers' must end with 1 (one output)")

    r['epochs'] = get_int(field(req, 'epochs'), 'epochs', 1)
    r['lr'] = get_number(field(req, 'lr'), 'lr')

    data = field(req, 'data')
    if not isinstance(data, list) or not data:
        raise ValueError("'data' must be a non-empty array")
    if len(data) > MAX_POINTS:
        raise ValueError(f"too many points: {len(data)} (max {MAX_POINTS})")
    r['xs'], r['labels'] = [], []
    for p in data:
        if not isinstance(p, list) or len(p) != 3:
            raise ValueError("each data point must be [x, y, label]")
        label = get_number(p[2], 'label')
        if label not in (-1.0, 1.0):
            raise ValueError("labels must be -1 or 1")
        r['xs'].append([get_number(p[0], 'x'), get_number(p[1], 'y')])
        r['labels'].append(label)

    r['init_weights'] = None
    w = req.get('init_weights')
    if w is not None:
        if not isinstance(w, list):
            raise ValueError("'init_weights' must be an array")
        r['init_weights'] = [get_number(v, 'init_weights[i]') for v in w]

    r['seed'] = get_int(req['seed'], 'seed', 0) if 'seed' in req else 1
    r['snapshot_every'] = get_int(req['snapshot_every'], 'snapshot_every', 0) if 'snapshot_every' in req else 0

    if r['snapshot_every'] > 0:
        g = field(req, 'grid')
        if not isinstance(g, dict):
            raise ValueError("'grid' must be an object")
        r['grid'] = {
            'nx': get_int(field(g, 'nx'), 'grid.nx', 1),
            'ny': get_int(field(g, 'ny'), 'grid.ny', 1),
            'xmin': get_number(field(g, 'xmin'), 'grid.xmin'),
            'xmax': get_number(field(g, 'xmax'), 'grid.xmax'),
            'ymin': get_number(field(g, 'ymin'), 'grid.ymin'),
            'ymax': get_number(field(g, 'ymax'), 'grid.ymax'),
        }
        if r['grid']['xmin'] >= r['grid']['xmax'] or r['grid']['ymin'] >= r['grid']['ymax']:
            raise ValueError("grid needs xmin < xmax and ymin < ymax")

    return r


def make_frame(net, grid, epoch, loss):
    """Prediction at the centre of every grid cell, row j = 0 at ymin."""
    dx = (grid['xmax'] - grid['xmin']) / grid['nx']
    dy = (grid['ymax'] - grid['ymin']) / grid['ny']
    values = []
    for j in range(grid['ny']):
        for i in range(grid['nx']):
            x = grid['xmin'] + (i + 0.5) * dx
            y = grid['ymin'] + (j + 0.5) * dy
            values.append(round(net([x, y])[0].data, 3))
    return {'epoch': epoch, 'loss': loss, 'grid': values}


def run(r):
    # Python's `random` differs from C++'s mt19937, so seed-only runs start from
    # different weights than the C++ CLI. Pass init_weights to compare the two.
    random.seed(r['seed'])
    net = MLP(r['layers'][0], r['layers'][1:], r['activation'])
    if r['init_weights'] is not None:
        net.set_parameters(r['init_weights'])

    xs, labels = r['xs'], r['labels']
    params = net.parameters()
    every = r['snapshot_every']
    epochs = r['epochs']
    loss_history = []
    frames = []
    train_seconds = 0.0

    # Only the training steps are timed: the clock is paused while a frame is computed.
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = 0.0
        for x, y in zip(xs, labels):
            diff = net(x)[0] - y
            loss = loss + diff * diff
        if not math.isfinite(loss.data):
            raise Diverged(epoch)
        loss_history.append(loss.data)

        # Intermediate frames use the pre-update weights, so the grid matches this epoch's loss.
        # The last epoch's frame is made after training, from the final weights.
        if every > 0 and epoch != epochs and (epoch == 1 or epoch % every == 0):
            train_seconds += time.perf_counter() - start
            frames.append(make_frame(net, r['grid'], epoch, loss.data))
            start = time.perf_counter()

        for p in params:
            p.grad = 0.0
        loss.backward()

        for p in params:
            p.data -= r['lr'] * p.grad
    train_seconds += time.perf_counter() - start

    # One fresh forward pass with the final weights: final_loss, accuracy and the last frame.
    final_loss = 0.0
    correct = 0
    for x, y in zip(xs, labels):
        pred = net(x)[0].data
        diff = pred - y
        final_loss += diff * diff
        if (1.0 if pred >= 0 else -1.0) == y:
            correct += 1
    if not math.isfinite(final_loss):
        raise Diverged(epochs)
    if every > 0:
        frames.append(make_frame(net, r['grid'], epochs, final_loss))

    return {
        'ok': True,
        'impl': 'python',
        'param_count': len(params),
        'train_seconds': train_seconds,
        'final_loss': final_loss,
        'accuracy': correct / len(xs),
        'loss_history': loss_history,
        'frames': frames,
    }


def fail(message):
    print(json.dumps({'ok': False, 'error': message}))
    return 1


def main():
    try:
        req = json.loads(sys.stdin.read())
    except ValueError:
        return fail("request is not valid JSON")

    try:
        result = run(parse_request(req))
        # allow_nan=False: NaN is not valid JSON, fail loudly instead of emitting it
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as e:  # never show a stack trace: report the error as JSON
        return fail(str(e))


if __name__ == '__main__':
    sys.exit(main())
