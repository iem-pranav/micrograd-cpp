# micrograd-cpp demo. Run `make` targets from this directory (project/).

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
# Debug build: sanitizers catch memory errors, leaks (LeakSanitizer) and undefined behaviour.
# -fno-sanitize-recover makes UBSan stop with an error instead of just printing a warning.
DBGFLAGS := -std=c++17 -g -O0 -Wall -Wextra -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer

BUILD := build
VENV  := .venv
PY    := $(VENV)/bin/python

HEADERS := cpp/micrograd.hpp cpp/third_party/json.hpp

# A small request that exercises training, frames and the error path under the sanitizers.
ASAN_REQUEST := {"activation":"relu","layers":[2,6,6,1],"epochs":20,"lr":0.01,"seed":3,"snapshot_every":5,"data":[[0.5,1.0,1],[-1.0,-0.5,-1],[1.5,-0.2,1],[-0.3,0.8,-1],[0.1,0.1,1]],"grid":{"nx":4,"ny":4,"xmin":-2,"xmax":2,"ymin":-2,"ymax":2}}

.PHONY: all cpp asan test venv run bench clean

all: cpp

cpp: $(BUILD)/micrograd_cli

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/micrograd_cli: cpp/main.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) cpp/main.cpp -o $@

$(BUILD)/self_test: tests/self_test.cpp cpp/micrograd.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/self_test.cpp -o $@

$(BUILD)/micrograd_cli_asan: cpp/main.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(DBGFLAGS) cpp/main.cpp -o $@

$(BUILD)/self_test_asan: tests/self_test.cpp cpp/micrograd.hpp | $(BUILD)
	$(CXX) $(DBGFLAGS) tests/self_test.cpp -o $@

asan: $(BUILD)/self_test_asan $(BUILD)/micrograd_cli_asan
	./$(BUILD)/self_test_asan
	echo '$(ASAN_REQUEST)' | ./$(BUILD)/micrograd_cli_asan > /dev/null
	echo '{not json' | ./$(BUILD)/micrograd_cli_asan > /dev/null; test $$? -eq 1
	@echo "asan: clean"

# Python dependencies in a local virtualenv (Ubuntu's system Python refuses `pip install`).
# The stamp file is re-made whenever requirements.txt changes.
venv: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	test -d $(VENV) || python3 -m venv $(VENV)
	$(VENV)/bin/pip install --quiet -r requirements.txt
	touch $@

test: $(BUILD)/micrograd_cli $(BUILD)/self_test venv
	./$(BUILD)/self_test
	$(PY) -m pytest -q tests
	@if command -v node > /dev/null; then node --test tests/data.test.js; \
	else echo; echo "********************************************************************"; \
	echo "*** SKIPPED: node not found, data.js tests not run               ***"; \
	echo "*** (install node >= 18 to run tests/data.test.js)               ***"; \
	echo "********************************************************************"; fi

# Serves the API and web/ on http://127.0.0.1:8000 (localhost only).
run: $(BUILD)/micrograd_cli venv
	$(PY) -m uvicorn server:app --app-dir python --host 127.0.0.1 --port 8000

# Both CLIs on a few fixed configs, no server; prints a Markdown table.
bench: $(BUILD)/micrograd_cli venv
	$(PY) python/bench.py

clean:
	rm -rf $(BUILD)
