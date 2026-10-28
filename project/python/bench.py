"""Benchmark both CLIs without the HTTP server and print a Markdown table (for the README).

Run from project/:  make bench
Calls the same function as POST /api/benchmark (server.benchmark), so the rules are identical:
the 15 s Python budget check, identical init_weights for both, lr scaled by 1/N, 3 runs and the
median, no repeat after a Python run over 5 s, the same timeouts and the same lock.
"""

import math
import random

import server


def make_moons(n, seed=1, noise=0.1):
    """Two interleaving half circles scaled into [-2, 2]^2, labels +1 / -1."""
    rng = random.Random(seed)
    data = []
    for k in range(n):
        t = math.pi * rng.random()
        if k % 2 == 0:
            x, y, label = math.cos(t), math.sin(t), 1
        else:
            x, y, label = 1 - math.cos(t), 0.5 - math.sin(t), -1
        x = 1.3 * (x - 0.5 + rng.gauss(0, noise))
        y = 1.3 * (y - 0.25 + rng.gauss(0, noise))
        data.append([max(-2.0, min(2.0, x)), max(-2.0, min(2.0, y)), label])
    return data


# (name, points, hidden layers, epochs); all tanh, per-sample lr 0.1, weights seed 1
CONFIGS = [
    ('small', 30, [4], 300),
    ('default', 60, [8], 120),
    ('large', 100, [8, 8], 100),   # estimated Python ~10.5 s, inside the 15 s budget
]


def main():
    m = server.machine_info()
    print(f"Machine: {m['cpu']} ({m['cores']} threads), {m['os']}")
    print(f"Python {m['python']}, {m['compiler']} {m['flags']}"
          + (' (WSL2)' if m.get('wsl') else ''))
    print()
    print('| config | data | layers | params | epochs | C++ median (s) | Python median (s) '
          '| speedup | losses match |')
    print('|---|---|---|---|---|---|---|---|---|')
    for name, points, hidden, epochs in CONFIGS:
        body = server.BenchmarkRequest(data=make_moons(points), hidden=hidden,
                                       activation='tanh', epochs=epochs)
        r = server.benchmark(body)
        layers = '-'.join(map(str, [2] + hidden + [1]))
        params = server.param_count([2] + hidden + [1])
        if not isinstance(r, dict):   # an error response
            print(f'| {name} | moons {points} | {layers} | {params} | {epochs} '
                  f'| error: {r.body.decode()} | | | |')
            continue
        py = r['python']
        if py.get('status') == 'timeout':
            py_cell, speedup = f"timeout (> {py['timeout_s']} s)", '-'
        else:
            n = len(py['runs'])
            py_cell = f"{py['train_seconds_median']:.3f} ({n} run{'s' if n > 1 else ''})"
            speedup = f"{r['speedup']:.1f}x"
        print(f"| {name} | moons {points} | {layers} | {params} | {epochs} "
              f"| {r['cpp']['train_seconds_median']:.4f} ({len(r['cpp']['runs'])} runs) "
              f"| {py_cell} | {speedup} | {r['losses_match']} |", flush=True)


if __name__ == '__main__':
    main()
