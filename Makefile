# Makefile for MCP-Coordinated Swarm Intelligence

.PHONY: help install venv server simulate train experiment \
        dashboard-install dashboard test clean \
        review3-quick review3-full rl-compare slam-demo results \
        final-review final-review-quick gps-denied-test train-agents

# Variables
VENV          = venv
PYTHON        = $(shell [ -f $(VENV)/bin/python3 ] && echo $(VENV)/bin/python3 || echo python3)
PIP           = $(shell [ -f $(VENV)/bin/pip ] && echo $(VENV)/bin/pip || echo pip3)
NPM           = npm
CONFIG_DIR    = config
LOGS_DIR      = logs
SAVED_MODELS  = saved_models
RESULTS_DIR   = results

# Node
NPM_EXISTS := $(shell command -v $(NPM) 2>/dev/null)

# Venv bootstrap
ifeq ($(wildcard $(VENV)/bin/python3),)
    PYTHON_VENV_GEN = python3
else
    PYTHON_VENV_GEN = $(PYTHON)
endif

# ─── Help ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "MCP-Coordinated Swarm Intelligence — Available Commands"
	@echo "════════════════════════════════════════════════════════"
	@echo ""
	@echo "  Setup"
	@echo "    venv               Create Python virtual environment"
	@echo "    install            Install all Python (and optionally dashboard) deps"
	@echo ""
	@echo "  Final Review Experiments  <-- Main thesis experiments"
	@echo "    final-review-quick Run quick demo (5 episodes each, ~1-2 min)"
	@echo "    final-review       Run full publication-quality demo (20/15/12 ep)"
	@echo "    gps-denied-test    Run full demo in GPS-denied / SLAM mode"
	@echo ""
	@echo "  Training"
	@echo "    train-agents       Quick PPO warm-up (200 episodes, shows pipeline)"
	@echo "    train              Full RL training (set EPISODES=N, default 1000)"
	@echo ""
	@echo "  Review III (historical)"
	@echo "    review3-quick      Quick Review III demo"
	@echo "    review3-full       Full Review III demo"
	@echo "    rl-compare         Compare RL algorithms (PPO/SAC/TD3/A2C/DQN)"
	@echo "    slam-demo          SLAM integration demo"
	@echo ""
	@echo "  Utilities"
	@echo "    experiment         Baseline comparison (original)"
	@echo "    server             Start MCP server"
	@echo "    simulate           Run simulation GUI (HEADLESS=true for no GUI)"
	@echo "    dashboard-install  Install web dashboard npm dependencies"
	@echo "    dashboard          Start web dashboard (backend + frontend)"
	@echo "    test               Run all unit tests"
	@echo "    results            Open results directory"
	@echo "    clean              Remove logs, models, and results"
	@echo ""

# ─── Environment ─────────────────────────────────────────────────────────────
venv:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Done.  Run: source $(VENV)/bin/activate"

install:
	@[ -d "$(VENV)" ] || $(MAKE) venv
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	@if [ -z "$(NPM_EXISTS)" ]; then \
		echo "Warning: npm not found — skipping dashboard install."; \
	else \
		$(MAKE) dashboard-install; \
	fi

# ─── Final Review Experiments ─────────────────────────────────────────────────
final-review-quick:
	@echo "Running Final Review quick demo..."
	$(PYTHON) run_final_review_demo.py --mode quick

final-review:
	@echo "Running Final Review full demo (publication quality)..."
	$(PYTHON) run_final_review_demo.py --mode full

gps-denied-test:
	@echo "Running Final Review in GPS-denied / SLAM mode..."
	$(PYTHON) run_final_review_demo.py --mode full --gps-denied

# ─── Training ────────────────────────────────────────────────────────────────
train-agents:
	@echo "Quick PPO warm-up (200 episodes -- shows the training pipeline)..."
	$(PYTHON) -m rl_agents.train --episodes 200

train:
	$(PYTHON) -m rl_agents.train --episodes $(or $(EPISODES),1000) $(if $(CONFIG),--config $(CONFIG))

# ─── Core simulation ─────────────────────────────────────────────────────────
server:
	$(PYTHON) -m mcp_server.server

simulate:
	@if [ "$(HEADLESS)" = "true" ]; then \
		$(PYTHON) -m simulation.main --headless; \
	else \
		$(PYTHON) -m simulation.main; \
	fi

experiment:
	$(PYTHON) -m experiments.baseline_comparison

# ─── Review III (historical) ─────────────────────────────────────────────────
review3-quick:
	@echo "Running Review III quick demo..."
	$(PYTHON) run_review_iii_demo.py --quick

review3-full:
	@echo "Running Review III full demo..."
	$(PYTHON) run_review_iii_demo.py --full

rl-compare:
	@echo "Running RL algorithm comparison..."
	$(PYTHON) experiments/rl_comparison.py --episodes 100 --num_uavs 3

slam-demo:
	@echo "Running SLAM integration demo..."
	$(PYTHON) experiments/slam_comparison.py --episodes 20 --num_uavs 3

# ─── Dashboard ───────────────────────────────────────────────────────────────
dashboard-install:
	@[ -n "$(NPM_EXISTS)" ] || (echo "Error: npm is not installed."; exit 1)
	cd web_dashboard && $(NPM) install

dashboard:
	@[ -n "$(NPM_EXISTS)" ] || (echo "Error: npm is not installed."; exit 1)
	@echo "Starting web dashboard..."
	cd web_dashboard && $(NPM) start

# ─── Tests & Utilities ───────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/

results:
	@if [ -d "$(RESULTS_DIR)/final_review" ]; then \
		open $(RESULTS_DIR)/final_review 2>/dev/null || \
		xdg-open $(RESULTS_DIR)/final_review 2>/dev/null || \
		echo "Results: $(RESULTS_DIR)/final_review"; \
	elif [ -d "$(RESULTS_DIR)/review_iii" ]; then \
		open $(RESULTS_DIR)/review_iii 2>/dev/null || \
		echo "Results: $(RESULTS_DIR)/review_iii"; \
	else \
		echo "No results yet -- run 'make final-review-quick' first."; \
	fi

clean:
	rm -rf $(LOGS_DIR)/* $(SAVED_MODELS)/* $(RESULTS_DIR)/*
	@echo "Cleanup complete."
