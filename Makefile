# helpers
RED := \e[0;31m
GREEN := \e[0;32m
RESET := \e[0m
FLAKE8_SUCCESS := printf '%b\n' "$(GREEN)Success: flake8$(RESET)"

# structure
DIRS := . src llm_sdk
ARGS ?=

PYCACHES := $(addsuffix /__pycache__,$(DIRS))
MYPYCACHES := $(addsuffix /.mypy_cache,$(DIRS))

# tools
UV := uv
FLAKE8 := $(UV) run flake8 --exclude=.venv,llm_sdk,__pycache__,.mypy_cache
MYPY := $(UV) run mypy --exclude '(.*cache.*)' --follow-imports=skip

# flags
MYPY_FLAGS := \
		--check-untyped-defs \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--warn-return-any \
		--disallow-untyped-defs

# rules
install: is_uv
	@$(UV) sync

run: install
	@$(UV) run python -m src $(ARGS)

clean:
	@rm -rf $(PYCACHES) $(MYPYCACHES) .venv data/output

debug: install
	@$(UV) run python -m pdb -m src $(ARGS)

lint: install
	@$(FLAKE8) && $(FLAKE8_SUCCESS)
	@$(MYPY) src $(MYPY_FLAGS)

lint-strict: install
	@$(FLAKE8) && $(FLAKE8_SUCCESS)
	@$(MYPY) src --strict

is_uv:
	@command -v $(UV) >/dev/null 2>&1 \
	|| (printf "$(RED)uv not found, please install it first$(RESET)\n" \
	&& exit 1)

# miscellaneous
.PHONY: install run debug lint lint-strict clean is_uv
