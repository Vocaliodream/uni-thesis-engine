# uni-thesis-engine -- one-command entry points
#   make init                                   install the package (editable) + register the pre-commit hook
#   make test                                   pytest with a 90 % coverage floor
#   make run PROFILE=profiles/demo_biosignal.yaml  full pipeline for one topic profile
#   make audit                                  anti-hallucination audit of every generated thesis in results/
#   make all                                    both built-in profiles + tests + audit
#   make pdf PROFILE=...                        compile results/<id>/thesis.pdf (needs a TeX engine)

PYTHON      ?= python3
PROFILE     ?= profiles/demo_biosignal.yaml
RESULTS     ?= results
TEMPLATE    ?= templates/thesis_tex
RUN_FLAGS   ?=
COV_MIN     ?= 90
PROFILES    := profiles/demo_biosignal.yaml profiles/demo_rsm_opt.yaml
UNI         := $(PYTHON) -m uni_thesis.cli

.PHONY: init install hooks test run audit audit-fresh pdf all clean help

help:
	@sed -n '2,8p' Makefile

init: install hooks

install:
	$(PYTHON) -m pip install -q -e .

hooks:
	bash scripts/install_hooks.sh

test:
	$(PYTHON) -m pytest -q -W error --cov=src/uni_thesis --cov-report=term --cov-fail-under=$(COV_MIN)

run:
	$(UNI) run --profile $(PROFILE) --results-dir $(RESULTS) --template-dir $(TEMPLATE) $(RUN_FLAGS)

audit:
	$(PYTHON) scripts/audit_anti_hallucination.py --results-dir $(RESULTS)

audit-fresh:
	$(PYTHON) scripts/audit_anti_hallucination.py

pdf:
	$(UNI) pdf --profile $(PROFILE) --results-dir $(RESULTS)

all:
	@for p in $(PROFILES); do \
		echo "==> run $$p"; \
		$(UNI) run --profile $$p --results-dir $(RESULTS) --template-dir $(TEMPLATE) $(RUN_FLAGS) || exit 1; \
	done
	$(MAKE) --no-print-directory test
	$(MAKE) --no-print-directory audit
	@echo "make all: profiles, tests and anti-hallucination audit are green"

clean:
	rm -rf $(RESULTS)/*/ .pytest_cache .coverage htmlcov
