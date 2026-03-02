# helpers
GREEN := \e[0;32m
RESET := \e[0m
FLAKE8_SUCCESS := printf '%b\n' "$(GREEN)Success: flake8$(RESET)"

# structure
SRC_DIRECTORIES := # to do
DIRS := . src $(addprefix src/,$(SRC_DIRECTORIES))
ARGS ?= #

PYCACHES = $(addsuffix /__pycache__,$(DIRS))
MYPYCACHES = $(addsuffix /.mypy_cache,$(DIRS))
EXCLUDE = --exclude .venv

# tools
UV := uv
FLAKE8 := $(UV) run flake8 $(EXCLUDE)
MYPY := $(UV) run mypy $(EXCLUDE)

# flags
MYPY_FLAGS := \
		--check-untyped-defs \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--warn-return-any \
		--disallow-untyped-defs

# rules
install: is_uv
	$(UV) sync

run: install
	mkdir -p data/output
	$(UV) run python -m src $(ARGS)

clean:
	rm -rf $(PYCACHES) $(MYPYCACHES) .venv data/output

debug: install
	$(UV) run python -m pdb -m src $(ARGS)

lint: install
	$(FLAKE8) && $(FLAKE8_SUCCESS)
	$(MYPY) . $(MYPY_FLAGS)

lint-strict: install
	$(FLAKE8) && $(FLAKE8_SUCCESS)
	$(MYPY) . --strict

is_uv:
	command -v $(UV) >/dev/null 2>&1 \
	|| (echo "uv not found, please install it first" && exit 1)

# miscellaneous
.PHONY: install run debug lint lint-strict clean is_uv
