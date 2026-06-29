.PHONY: install run api test lint typecheck check

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m streamlit run app.py

api:
	$(PYTHON) -m uvicorn api.main:app --reload

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src api

check: lint typecheck test
