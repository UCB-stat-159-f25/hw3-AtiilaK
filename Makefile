# ---- config ----
SHELL := /bin/bash
ENV_NAME := ligo  # change if your env has a different name

.PHONY: env html clean

# Create or update the conda environment from environment.yml
env:
	conda env update -n $(ENV_NAME) --file environment.yml --prune || \
	conda env create -n $(ENV_NAME) -f environment.yml

# Build local HTML for the MyST site
html:
	myst build --html

# Remove generated artifacts
clean:
	rm -rf figures/* audio/* _build/
