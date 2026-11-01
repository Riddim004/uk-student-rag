PYTHON ?= python

.PHONY: setup data data-live test eval calibrate measure serve demo smoke

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt -r requirements-local.txt -r requirements-evidence.txt

data:
	$(PYTHON) scripts/build_data.py --offline

data-live:
	$(PYTHON) scripts/build_data.py

test:
	$(PYTHON) -m pytest -q

eval:
	$(PYTHON) scripts/run_eval.py --rerank

calibrate:
	$(PYTHON) scripts/calibrate_gate.py --rerank

measure:
	$(PYTHON) scripts/measure.py

serve:
	$(PYTHON) scripts/serve.py

demo:
	$(PYTHON) scripts/serve.py --demo

smoke:
	$(PYTHON) -m ragkit "How many hours can I work during term time?" --embedding-backend hashing --no-rerank --no-llm --json
