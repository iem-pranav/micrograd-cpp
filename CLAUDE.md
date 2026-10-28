# micrograd-cpp: C++ vs Python autograd demo

## What this project is

A web demo built on a C++ port of Andrej Karpathy's micrograd (scalar autograd engine + tiny MLP).
A user supplies 2D data, the C++ backend trains a small neural network on it, and the page animates the
learning (decision boundary + loss curve). A second panel runs the **same training** in pure-Python micrograd
and shows which is faster.

The file `micrograd.cpp` at the repo root is the author's own hand-written port. Everything new lives in `project/`.

## Ground rules (hard rules)

- **Never edit, move, reformat or delete `micrograd.cpp`.** It is the author's own work. Read it, build on it, leave it alone.
- The author is **learning C++**. Keep code simple and readable. No templates, macros or metaprogramming.
  Match the style of `micrograd.cpp` (handle-style `Value` wrapping `shared_ptr<Node>`; closures capture the
  children as `an`/`bn` and a **raw** `Node* o` for the output; keep the comment explaining why `o` is raw:
  a `shared_ptr` there would create an ownership cycle and leak the graph).
  When you use a non-obvious C++ feature, add a one-line comment saying why.
- No `using namespace std` in headers. Use `std::` there.
- C++: g++, `-std=c++17 -O2 -Wall -Wextra`, **zero warnings**. A separate debug target uses `-g -O0 -fsanitize=address,undefined`.
- Dependencies, and nothing else without asking first:
  - C++: `nlohmann/json` single header, vendored in `project/cpp/third_party/`.
  - Python: `fastapi`, `uvicorn`, `pydantic`, `pytest`. The micrograd Python code itself is **pure Python, no numpy**.
  - Frontend: plain HTML/CSS/JS + canvas. No framework, no bundler, no build step.
- **Do not `git commit` or `git push` unless asked.** Never commit build outputs or binaries (see `.gitignore`).
- Ask before changing the CLI protocol below, adding dependencies, or raising the limits.
- **Work phase by phase** (see Phases). At the end of each phase: summarize what changed, how to verify it, and
  list the 2-3 files the author should read first to understand it. Then stop and wait for a go-ahead.
- Before writing code for a phase, give a short plan (files, in order) and wait for an OK.

## Repo layout

```
micrograd_cpp/
  micrograd.cpp            author's original. DO NOT TOUCH
  CLAUDE.md                this file
  README.md                written in Phase 4
  .gitignore
  project/
    Makefile
    cpp/
      micrograd.hpp        header-only engine (derived from micrograd.cpp + activation selector)
      main.cpp             CLI: JSON on stdin -> JSON on stdout
      third_party/json.hpp
    python/
      micrograd.py         pure-Python port, same model + training as the C++
      cli.py               same CLI protocol as the C++ one
      server.py            FastAPI app, also serves web/
      bench.py             `make bench`: runs the benchmark code path without HTTP, prints a Markdown table
    requirements.txt       pinned Python deps, installed into project/.venv (gitignored) by the Makefile
    web/
      index.html, style.css
      app.js               DOM wiring, canvas drawing, playback, fetch calls
      data.js              pure logic (presets, CSV, validation, colours, request bodies); no DOM, testable in node
    data/
      moons_sample.csv     60 moons points (the UI hint points to it)
    tests/
      test_equivalence.py
      self_test.cpp        C++ self-checks (run by `make test` and `make asan`)
      test_server.py       server tests (FastAPI TestClient)
      data.test.js         web/data.js tests, `node --test` (make test skips them if node is missing)
    reference/
      micrograd_extended.cpp   tested single-file reference (read-only). The header must behave identically
    build/                 gitignored
```

## Commands (Makefile in `project/`)

- `make cpp`   build `build/micrograd_cli` (`-O2`)
- `make asan`  debug build with sanitizers, runs the C++ self-checks
- `make test`  C++ self-checks + `pytest`
- `make run`   start the server on `127.0.0.1:8000`
- `make bench` run the CLI benchmark without the server and print a table (used for the README)

## Architecture

```
browser (web/) --HTTP/JSON--> server.py (FastAPI)
                                |-- subprocess --> build/micrograd_cli   (C++)
                                |-- subprocess --> python/cli.py         (pure Python)
```

Both CLIs read the **same JSON on stdin** and write JSON on stdout, so the server just runs both on identical input.
Each program times **its own** training loop internally. Process startup and JSON handling are never timed.

## Training contract (must be identical in C++ and Python)

- Network: `layers = [nin, h1, ..., 1]`. A neuron computes `sum(w_i * x_i) + b`, then the activation.
  Hidden layers use the chosen activation (`tanh` | `relu` | `sigmoid`). **The last layer is always linear.**
- **Parameter order** (for `parameters()` and `init_weights`): for each layer, for each neuron, its weights in input order, then its bias.
- Initial values: `init_weights` if given. Otherwise uniform(-1, 1) from `seed`.
- Loss: **sum** over samples of `(pred - label)^2` (not the mean). Labels are -1 or +1.
- One epoch, full batch: forward every sample in data order, build the loss, zero all grads,
  `loss.backward()`, then `p.data -= lr * p.grad` for every parameter. The recorded loss is from the forward pass **before** the update.
- Accuracy: after the last epoch do a fresh forward pass; `pred >= 0` means +1, else -1; compare with the label.
- Sigmoid uses the same overflow-safe form in both: `x >= 0: 1/(1+exp(-x))`, `x < 0: e=exp(x), e/(1+e)`.
  Tanh uses `std::tanh` / `math.tanh` (not the notebook's exp formula).
- Equivalence rule: with identical input (including `init_weights`) the two implementations' `final_loss` must agree within relative 1e-6.

## CLI protocol

Request (stdin, one JSON object):

```json
{
  "activation": "tanh",
  "layers": [2, 8, 8, 1],
  "epochs": 200,
  "lr": 0.05,
  "data": [[x, y, label], ...],
  "init_weights": [ ... ],
  "seed": 1,
  "snapshot_every": 5,
  "grid": {"nx": 30, "ny": 30, "xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2}
}
```

- `init_weights`: optional. Length must equal the parameter count.
- `snapshot_every`: 0 means no frames (benchmark mode).
- `grid`: only used when `snapshot_every > 0`.

Response (stdout):

```json
{
  "ok": true,
  "impl": "cpp",
  "param_count": 105,
  "train_seconds": 0.0123,
  "final_loss": 0.0123,
  "accuracy": 0.97,
  "loss_history": [ ...one value per epoch... ],
  "frames": [ {"epoch": 5, "loss": 0.9, "grid": [ ...nx*ny values... ]} ]
}
```

- `grid[j*nx + i]` is the prediction at `x = xmin + (i+0.5)*dx`, `y = ymin + (j+0.5)*dy`, with `j = 0` at `ymin` (the frontend flips vertically). Round to 3 decimals to keep payloads small.
- A frame is recorded at epoch 1, every `snapshot_every` epochs, and the last epoch.
- `train_seconds` excludes time spent computing frames.
- Frames and final values: `loss_history[k]` is epoch k+1's loss, from the pre-update weights. Intermediate frames
  use the pre-update weights too, so a frame's grid matches its `loss`. The **last** frame, `final_loss` and
  `accuracy` all come from the weights **after** the final update, via one fresh forward pass
  (so `final_loss` is not `loss_history[-1]`).
- Divergence: the loss is checked with `isfinite` every epoch. If it becomes NaN/inf, return
  `{"ok": false, "error": "training diverged at epoch N, try a lower learning rate"}` with a non-zero exit.
- Seed-only runs (no `init_weights`) use different RNGs in C++ (`mt19937`) and Python (`random`), so they start from
  different weights. Tests and the server always pass `init_weights`.
- CLI input checks: points <= 400, `layers[0] == 2`, `layers[-1] == 1`, `init_weights` length. All other limits live in `server.py`.
- Errors: `{"ok": false, "error": "..."}` and a non-zero exit code. Never crash with a stack trace. Validate input first.

## Components

### C++ (`project/cpp`)
- `micrograd.hpp`: header-only, so every function and global is `inline`.
  Based on `/micrograd.cpp`. Keep its op set: `+ - neg * / pow exp relu sigmoid tanh`.
  Add `enum class Activation { Tanh, ReLU, Sigmoid, Linear }` with an `activate(Value, Activation)` function.
  `Neuron` throws `std::invalid_argument` if the input size is wrong. Use a seedable RNG (`seed_rng(unsigned)`), not a global `random_device`.
  Copying a `Neuron`/`Layer`/`MLP` is shallow (shares weights). Keep and document that.
- `main.cpp`: the CLI. Time with `std::chrono::steady_clock` around the training steps only.
- `build_topo` is recursive and its depth grows with the number of data points. Do not raise the point limit without making it iterative.

### Python (`project/python`)
- `micrograd.py`: a faithful port of the author's notebook (`Value` with set-based `_prev` and closures, `Neuron`, `Layer`, `MLP`), plus relu and sigmoid and a linear last layer.
  **Keep it idiomatic Karpathy-style. Do not hand-optimize it**, or the comparison becomes unfair the other way.
- `cli.py`: same protocol as the C++ CLI.
- `server.py` endpoints:
  - `GET /` and static files from `web/`
  - `POST /api/train`: C++ only, with frames (choose `snapshot_every` so frames <= 100)
  - `POST /api/benchmark`: run both on identical input, `snapshot_every = 0`
  - `GET /api/limits`, `GET /api/health`
- The server generates `init_weights` (seeded) and passes the same weights to both implementations.
- Validate with pydantic and return a clear 422 for over-limit requests. Use `subprocess.run` with a list argument (no shell) and timeouts. Kill on timeout. Bind to `127.0.0.1` by default.
- Run benchmarks one at a time (lock). Never run C++ and Python concurrently during a benchmark.

Benchmark response:

```json
{
  "ok": true,
  "cpp":    {"train_seconds_median": 0.01, "runs": [0.01, 0.01, 0.01], "final_loss": 0.012, "accuracy": 0.97},
  "python": {"train_seconds_median": 1.2,  "runs": [1.2, 1.2, 1.2],    "final_loss": 0.012, "accuracy": 0.97},
  "speedup": 120.0,
  "losses_match": true,
  "machine": {"cpu": "...", "python": "3.x", "compiler": "g++ x.y", "flags": "-O2"}
}
```

If Python hits the timeout, return `"python": {"status": "timeout", "timeout_s": 20}` and no speedup.

### Benchmark rules
- Same input, same `init_weights`, same epochs and lr. C++ built with `-O2`.
- `bench_runs` = 3 by default, report the median and all runs. If one Python run takes more than 5 s, do not repeat it.
- `make bench` (README table) calls `server.benchmark` directly, so the same rules and the 15 s budget guard apply.
  Configs, all tanh, per-sample lr 0.1, weights seed 1, moons data (seed 1): small = 30 points, hidden `4`, 300 epochs;
  default = 60, `8`, 120; large = 100, `8,8`, 100 (est. ~10.5 s Python).
  Measure on an otherwise idle machine; run the table twice and investigate if speedups differ by more than 15%.
- The UI must state that times are measured on the machine running the server, not on the visitor's device.

### Web (`project/web`)
- Plain `index.html` + `app.js` + `style.css` + `data.js`, canvas based. No CDNs or external fonts: works offline.
- Every bound and default comes from `GET /api/limits`; the page hard-codes only the data space `[-2, 2]^2`.
- The UI shows the server's error text for 409 / 422 / network errors in the `aria-live` status line.
- Data input: presets (moons, circles, XOR) generated client-side in the data space `[-2, 2]^2` with a noise slider; click to add a point (toggle the class +1 / -1); clear; paste CSV (`x,y,label`, label 0/1 or -1/1, mapped to -1/+1).
- Settings: hidden layer sizes (e.g. `8,8`), activation, learning rate, epochs. Buttons: Train (animate), Benchmark, Re-initialize (new seed). Re-initialize matters because relu can start "dead" and never learn.
- Panels:
  1. Decision boundary: grid frames drawn as coloured cells under the points, with play / pause / scrub.
  2. Loss curve that draws progressively with the frames.
  3. Benchmark: two bars (C++ vs Python), the speedup, a "losses match" badge, the machine note, and a graceful timeout state.
- **Not in scope: any computation-graph / node-level visualization.**
- Keep the page tidy, readable and usable on a laptop screen.

## Limits (constants in `server.py`, exposed through `/api/limits`)

`MAX_POINTS=400`, `MAX_EPOCHS=2000`, `MAX_LAYERS=5` (including the output layer), `MAX_NEURONS=32` per layer,
`MAX_FRAMES=100`, `PY_TIMEOUT_S=20`, `CPP_TIMEOUT_S=10`, `MAX_GRID=60` (cells per side for `/api/train` frames).

### Learning rate (server API)
- The CLIs keep the **sum** loss and take `lr` as-is. In the server's public API, `lr` is a **per-sample** rate:
  the server passes `lr / len(data)` to both CLIs (and returns it as `lr_effective`).
- `LR_DEFAULT=0.1`, `LR_MAX=0.3`. Measured on 100 moons points, layers `8,8`, 200 epochs, 5 seeds:
  at 0.1 nothing diverges (tanh/relu ~0.99 accuracy, sigmoid ~0.87); relu gets unstable from 0.2; divergence starts at 0.5.

### Python cost guard
- `est_python_seconds = points * epochs * params * PY_SECONDS_PER_UNIT`, with `PY_SECONDS_PER_UNIT=1.0e-5`
  (measured 6.6e-6 .. 1.05e-5 over 8 configs on an i7-10750H / WSL2; the high end, so it rarely undershoots).
  Exposed through `/api/limits`.
- `/api/benchmark` returns 422 with `est_python_seconds` when the estimate is over `PY_BUDGET_S=15`,
  and includes `est_python_seconds` in every successful response.
- UI defaults (in `/api/limits` as `defaults`): 60 moons points, hidden `8`, tanh, lr 0.1, 120 epochs
  (est 2.4 s per Python run; 0.87-1.00 accuracy for all three activations).

### Server behaviour
- One lock for all CLI runs; a request that finds it busy gets 409 (never two runs at once).
- 422 for invalid or over-limit input (`{"ok": false, "error": "...", "details": [...]}`), also when training diverges.
  500 if a CLI is missing or misbehaves, 504 if C++ times out.

## Phases

1. **Engines and CLIs, no web.** `micrograd.hpp`, `main.cpp`, `micrograd.py`, `cli.py`, Makefile, `tests/test_equivalence.py`
   (tanh/relu/sigmoid x at least 3 seeds: both CLIs agree within 1e-6; the author's notebook neuron example gives 0.7071 and grads -1.5, 1, 0.5, 0).
2. **Server and benchmark endpoint.** Testable with `curl`. Add `make bench`.
3. **Frontend:** data input, train animation, loss curve, benchmark panel.
4. **Polish, README.** README covers: what it is, build/run, a benchmark table measured on the author's machine (state CPU / OS / compiler / flags),
   and a Credits section: "C++ engine is the author's port of Karpathy's micrograd; the demo around it was built with Claude Code assistance."
5. **Optional:** Dockerfile and deploy notes; a WASM build of the engine for a static, no-server demo.

## Definition of done (every phase)

`make test` passes, zero compiler warnings, `make asan` clean, and the phase's behaviour verified by actually running it
(not only by reading the code).
