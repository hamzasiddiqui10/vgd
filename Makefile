#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = vbb
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python



#################################################################################
# COMMANDS                                                                      #
#################################################################################

train_contrastive_adapter:
	srun --pty -t 0-05:00:00 -p grete:shared -G A100:1 -C inet -C 80gb -C ssd --cpus-per-gpu=32 --mem=120GB /mnt/vast-nhr/projects/nib00021/micromamba/envs/vbb/bin/python vbb/contrastive_adapter/train.py $(cfg)

interactive_a100:
	srun --pty -t 0-24:00:00 -p grete:shared -G A100:1 --cpus-per-gpu=32 --mem=120GB /bin/bash

interactive_h100:
	srun --pty -t 0-48:00:00 -p grete-h100:shared -G H100:1 --cpus-per-gpu=24 --mem=220GB /bin/bash

interactive_a100_6h:
	srun --pty -t 0-06:00:00 -p grete:shared -G A100:1 --cpus-per-gpu=32 --mem=120GB /bin/bash

interactive_4xa100_48h:
	srun --pty -t 0-48:00:00 -p grete:shared -G A100:4 -c 64 --mem=480GB /bin/bash

interactive_a100_80gb_48ho:
	srun --pty -t 0-48:00:00 -p grete:shared -G A100:1 -C 80gb --cpus-per-gpu=32 --mem=120GB /bin/bash

interactive: 
	srun --pty -t 0-12:00:00 -p grete:interactive -G 1g.10gb  -C inet --cpus-per-task=8 --mem=32GB /bin/bash

interactive_small: 
	srun --pty -t 0-12:00:00 -p grete:interactive -G 1g.10gb  -C inet --cpus-per-task=4 --mem=16GB /bin/bash

interactive_small20gb: 
	srun --pty -t 0-12:00:00 -p grete:interactive -G 2g.20gb  -C inet --cpus-per-task=8 --mem=32GB /bin/bash

interactive_small40gb: 
	srun --pty -t 0-12:00:00 -p grete:interactive -G 3g.40gb  -C inet --cpus-per-task=16 --mem=64GB /bin/bash

interactive_small2g: 
	srun --pty -t 0-12:00:00 -p grete:interactive -G 2g.10gb  -C inet --cpus-per-task=8 --mem=32GB /bin/bash

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 vbb
	isort --check --diff --profile black vbb
	black --check --config pyproject.toml vbb

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml vbb



## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################




#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
