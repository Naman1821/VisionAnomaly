.PHONY: install download train eval compare demo api test

PYTHON ?= python3
ROOT := $(shell pwd)

install:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_mvtec.py --category bottle

train:
	$(PYTHON) scripts/train.py -c configs/default.yaml

eval:
	$(PYTHON) scripts/evaluate.py -c configs/default.yaml

compare:
	$(PYTHON) scripts/compare.py -c configs/default.yaml --category bottle

demo:
	CHECKPOINT=outputs/bottle/patchcore $(PYTHON) app/gradio_demo.py

api:
	CHECKPOINT=outputs/bottle/patchcore uvicorn app.fastapi_server:app --reload --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest tests/ -q

pipeline: download train eval compare
