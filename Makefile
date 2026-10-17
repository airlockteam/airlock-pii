# Written for GNU Make 3.81, which is what macOS ships: no .ONESHELL, no grouped targets,
# no $(file ...).

.DEFAULT_GOAL := help

UV      ?= uv
# --locked makes every command fail if uv.lock is out of date, instead of silently updating it.
UV_RUN  := $(UV) run --locked --all-packages
# Where `make models` puts the NER model files, and where the tests and the benchmark look.
MODELS_DIR ?= models
# Two projects with a lock of their own, so that Presidio and spaCy never enter the lock of
# the library: the Presidio baselines of the benchmark, and the tests of the library's
# adapter for Presidio. `uv run --project` builds each environment on first use.
PRESIDIO_PROJECT := bench/presidio
ADAPTER_PROJECT := packages/airlock-pii/tests_presidio
ADAPTER_RUN := $(UV) run --project $(ADAPTER_PROJECT) --locked
# The NER runs with one BLAS thread: numpy's default of one per core makes its small matrices
# several times slower.
NER_ENV := AIRLOCK_PII_MODELS_DIR=$(abspath $(MODELS_DIR)) OPENBLAS_NUM_THREADS=1
# The share of the library's lines and branches that its own tests must run.
COVERAGE_MIN := 99

BENCH_DATA := bench/data/v1.jsonl
# Where the results of the comparison go. The workflow that runs it on a clean machine puts
# them outside the checkout, next to no result of an earlier day.
BENCH_RESULTS ?= bench/results
# Where `make build` puts the wheel and the source distribution.
DIST ?= dist

.PHONY: help setup lint fmt typecheck test test-presidio check deps-audit models bench \
	bench-smoke bench-compare build acceptance

help: ## List the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install the workspace (uv sync)
	$(UV) sync --locked --all-packages

lint: ## ruff check and ruff format --check
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

fmt: ## ruff format, then ruff check --fix
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

typecheck: ## mypy: the library, the benchmark and the tools
	$(UV_RUN) mypy packages/airlock-pii/src packages/airlock-pii/tests
	$(UV_RUN) mypy bench
	$(UV_RUN) mypy tools

# The tests of the library run first and alone, with their coverage of the library measured:
# what the benchmark's tests happen to run is not counted. validators.py, where a wrong branch
# is a wrong check digit, is held to every line and every branch. Then everything else:
# --ignore-glob and not --ignore, which pytest does not apply to a folder that is one of its
# `testpaths`.
test: ## All tests (the NER's need `make models`); the library's own must cover $(COVERAGE_MIN)% of it
	$(NER_ENV) $(UV_RUN) pytest packages/airlock-pii/tests --cov=airlock_pii --cov-branch \
		--cov-report=term-missing:skip-covered --cov-fail-under=$(COVERAGE_MIN)
	$(UV_RUN) coverage report --include='*/airlock_pii/validators.py' --fail-under=100
	$(NER_ENV) $(UV_RUN) pytest --ignore-glob='packages/airlock-pii/tests/*'

# The adapter for Presidio, where Presidio is installed: its tests and the README's example,
# with every line and branch of the adapter run; mypy with Presidio's own types; and the tests
# of the Russian NER once more, on the numpy that Presidio asks for.
test-presidio: ## The adapter for Presidio, in its own environment
	$(NER_ENV) $(ADAPTER_RUN) pytest -c $(ADAPTER_PROJECT)/pyproject.toml --rootdir packages/airlock-pii \
		$(ADAPTER_PROJECT) packages/airlock-pii/tests/recognizers/common/test_ner_ru.py \
		--cov=airlock_pii.presidio_adapter --cov-branch --cov-config=$(ADAPTER_PROJECT)/pyproject.toml \
		--cov-report=term-missing --cov-fail-under=100
	$(ADAPTER_RUN) mypy --config-file $(ADAPTER_PROJECT)/pyproject.toml \
		packages/airlock-pii/src/airlock_pii/presidio_adapter.py $(ADAPTER_PROJECT)

check: lint typecheck test bench-smoke ## Everything a pull request has to pass, but for test-presidio

# Three locks are audited: the workspace's, and those of the two projects that hold Presidio.
# pip-audit says of the two spaCy model wheels of the benchmark that it skips them: they are
# named by URL, hold no code of a package index, and are pinned by their SHA-256 in that lock.
deps-audit: ## pip-audit against every locked dependency set
	@tmp=$$(mktemp) && trap 'rm -f "$$tmp"' EXIT && \
	$(UV) export --locked --all-packages --no-emit-workspace --format requirements-txt --output-file "$$tmp" >/dev/null && \
	$(UV_RUN) pip-audit --require-hashes --disable-pip -r "$$tmp" && \
	$(UV) export --project $(PRESIDIO_PROJECT) --locked --format requirements-txt --output-file "$$tmp" >/dev/null && \
	$(UV_RUN) pip-audit --require-hashes --disable-pip -r "$$tmp" && \
	$(UV) export --project $(ADAPTER_PROJECT) --locked --no-emit-package airlock-pii --format requirements-txt --output-file "$$tmp" >/dev/null && \
	$(UV_RUN) pip-audit --require-hashes --disable-pip -r "$$tmp"

models: ## Download the NER models and verify their SHA-256 from models.lock
	$(UV_RUN) python -m airlock_pii models download --dir $(MODELS_DIR)

# Each benchmark runs the library twice: as it is without extras, and with the Russian NER,
# which needs the model files of `make models` and fails without them.
bench: ## Full benchmark, without and with the NER
	$(UV_RUN) python bench/generate.py --seed 42 --n 2000 --out $(BENCH_DATA)
	$(UV_RUN) python bench/evaluate.py --detector airlock_pii --data $(BENCH_DATA)
	$(NER_ENV) $(UV_RUN) python bench/evaluate.py --detector airlock_pii_ner --data $(BENCH_DATA)

# The first 200 samples of the full run, against the thresholds.
bench-smoke: ## CI-sized benchmark with thresholds, without and with the NER
	$(UV_RUN) python bench/generate.py --seed 42 --n 200 --out bench/data/smoke.jsonl
	$(UV_RUN) python bench/evaluate.py --detector airlock_pii --data bench/data/smoke.jsonl \
		--results-dir bench/results/smoke --check
	$(NER_ENV) $(UV_RUN) python bench/evaluate.py --detector airlock_pii_ner --data bench/data/smoke.jsonl \
		--results-dir bench/results/smoke --check

# One command from a clean checkout: it fetches the NER model files, generates the data,
# builds the environment of the Presidio baselines from its lock (about 940 MB of downloads
# the first time), checks there that Presidio is what the baselines say it is
# (bench/check_presidio.py), and runs all five detectors. The table is printed and written to
# $(BENCH_RESULTS)/<date>_compare.md, with the JSON of every detector next to it.
# `python bench/readme_table.py` then puts the new tables into the README.
bench-compare: models ## Comparison with the baselines: the table, and the JSON results
	$(UV_RUN) python bench/generate.py --seed 42 --n 2000 --out $(BENCH_DATA)
	$(UV) run --project $(PRESIDIO_PROJECT) --locked --no-dev python bench/check_presidio.py
	UV=$(UV) $(NER_ENV) $(UV_RUN) python bench/compare.py --data $(BENCH_DATA) \
		--results-dir $(BENCH_RESULTS)

build: ## Build the wheel and the source distribution into $(DIST)
	rm -rf $(DIST)
	$(UV) build --package airlock-pii --out-dir $(DIST)

# Each built file is installed with pip into an environment that holds nothing else, and the
# quickstart of the README inside that file is run there as it is written. PYTHON chooses
# the interpreter, for example PYTHON=3.14.
acceptance: ## Install the wheel and the sdist in clean environments and run the README's quickstart
	$(UV) run --no-project $(if $(PYTHON),--python $(PYTHON)) python tools/acceptance.py $(DIST)
