# Makefile for MCP-Coordinated Swarm Intelligence

.PHONY: help install server simulate train experiment dashboard dashboard-install test clean

# Variables
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
NPM = npm
CONFIG_DIR = config
LOGS_DIR = logs
SAVED_MODELS_DIR = saved_models
RESULTS_DIR = results

# Check for npm availability
NPM_EXISTS := $(shell command -v $(NPM) 2> /dev/null)

# Check if venv exists, otherwise fallback to system python for venv creation
ifeq ($(wildcard $(VENV)/bin/python3),)
    PYTHON_VENV_GEN = python3
else
    PYTHON_VENV_GEN = $(PYTHON)
endif

# Default target
help:
	@echo "MCP-Coordinated Swarm Intelligence - Available Commands:"
	@echo "  venv               Create a Python virtual environment"
	@echo "  install            Install dependencies (Python and optionally Dashboard)"
	@echo "  server             Start the MCP (Model Context Protocol) Server"
	@echo "  simulate           Run the simulation (use HEADLESS=true for no GUI)"
	@echo "  train              Train RL agents (use EPISODES=1000 to set duration)"
	@echo "  experiment         Run baseline comparison experiments"
	@echo "  dashboard-install  Install web dashboard dependencies"
	@echo "  dashboard          Start the web dashboard (Backend & Frontend)"
	@echo "  test               Run all unit tests"
	@echo "  clean              Remove logs, saved models, and results"
	@echo ""
	@echo "Review III Targets:"
	@echo "  review3-quick      Quick demo for Review III (5-10 min)"
	@echo "  review3-full       Full demo for Review III (30-60 min)"
	@echo "  rl-compare         Compare RL algorithms (PPO, SAC, TD3, A2C, DQN)"
	@echo "  slam-demo          Demonstrate SLAM integration"
	@echo "  results            Open results directory"

venv:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Virtual environment created. Please run 'source $(VENV)/bin/activate' in your shell."

install:
	@if [ ! -d "$(VENV)" ]; then \
		$(MAKE) venv; \
	fi
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	@if [ -z "$(NPM_EXISTS)" ]; then \
		echo "Warning: npm not found. Skipping web dashboard installation."; \
		echo "Please install Node.js and npm to use the web dashboard."; \
	else \
		$(MAKE) dashboard-install; \
	fi

server:
	$(PYTHON) -m mcp_server.server

simulate:
	@if [ "$(HEADLESS)" = "true" ]; then \
		$(PYTHON) -m simulation.main --headless; \
	else \
		$(PYTHON) -m simulation.main; \
	fi

train:
	$(PYTHON) -m rl_agents.train --episodes $(or $(EPISODES), 1000) $(if $(CONFIG), --config $(CONFIG))

experiment:
	$(PYTHON) -m experiments.baseline_comparison

dashboard-install:
	@if [ -z "$(NPM_EXISTS)" ]; then \
		echo "Error: npm is not installed. Cannot install dashboard dependencies."; \
		exit 1; \
	fi
	cd web_dashboard && $(NPM) install

dashboard:
	@if [ -z "$(NPM_EXISTS)" ]; then \
		echo "Error: npm is not installed. Cannot start dashboard."; \
		exit 1; \
	fi
	@echo "Starting web dashboard..."
	@echo "Starting server and client concurrently..."
	cd web_dashboard && $(NPM) start

test:
	$(PYTHON) -m pytest tests/

# Review III Targets
review3-quick:
	@echo "Running Review III Quick Demo (5-10 minutes)..."
	$(PYTHON) run_review_iii_demo.py --quick

review3-full:
	@echo "Running Review III Full Demo (30-60 minutes)..."
	$(PYTHON) run_review_iii_demo.py --full

rl-compare:
	@echo "Running RL Algorithm Comparison..."
	$(PYTHON) experiments/rl_comparison.py --episodes 100 --num_uavs 3

slam-demo:
	@echo "Running SLAM Integration Demo..."
	$(PYTHON) experiments/slam_comparison.py --episodes 20 --num_uavs 3

results:
	@echo "Opening results directory..."
	@if [ -d "results/review_iii" ]; then \
		open results/review_iii 2>/dev/null || xdg-open results/review_iii 2>/dev/null || echo "Results directory: results/review_iii"; \
	else \
		echo "No results found. Run 'make review3-quick' or 'make review3-full' first."; \
	fi


clean:
	rm -rf $(LOGS_DIR)/*
	rm -rf $(SAVED_MODELS_DIR)/*
	rm -rf $(RESULTS_DIR)/*
	@echo "Cleanup complete."
