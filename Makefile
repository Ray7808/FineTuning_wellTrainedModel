# Makefile to run model_modify.py and LoadHiLo.py within a virtual environment

# Determine the current directory (Based on operating system)
ifeq ($(OS),Windows_NT)
	PYTHON := python3
	VENV_PYTHON := .venv\Scripts\python
	CURRENT_DIR := $(shell cd)
else
	PYTHON := python3
	CURRENT_DIR := $(shell pwd)
	VENV_PYTHON := .venv/bin/python
endif

# Define the Python interpreter relative to the current directory

# Define your targets
.PHONY: all run_model_modify run_load_hilo

all: run_model_modify run_load_hilo

# Target to run model_modify.py
run_model_modify:
	$(VENV_PYTHON) "$(CURRENT_DIR)/model_modify.py"

# Target to run LoadHiLo.py
run_load_hilo: run_model_modify
	$(VENV_PYTHON) "$(CURRENT_DIR)/LoadHiLo.py"
