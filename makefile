install:
	pip install --upgrade pip &&\pip install -r requirements/requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C bikeshare_model/processing/features.py

test:
	python -m pytest tests/test_*.py

all: install test format lint