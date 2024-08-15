# Makefile to run model_modify.py and LoadHiLo.py within a virtual environment

# Determine the current directory
CURRENT_DIR := $(shell pwd)

# Define the Python interpreter relative to the current directory
PYTHON := $(CURRENT_DIR)/.venv/bin/python

# Define your targets
.PHONY: all

all: run_model_modify run_load_hilo

# Target to run model_modify.py
run_model_modify:
	$(PYTHON) $(CURRENT_DIR)/model_modify.py

# Target to run LoadHiLo.py
run_load_hilo: run_model_modify
	$(PYTHON) $(CURRENT_DIR)/LoadHiLo.py
