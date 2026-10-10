.PHONY: install data train index api test eval bias

install:
	pip install -r requirements.txt
	pip install -e .

data:
	python scripts/make_data.py

train:
	python scripts/train_model.py

index:
	python scripts/build_index.py

api:
	uvicorn aml_copilot.api.main:app --reload

test:
	python -m pytest -q

eval:
	python scripts/run_eval.py

bias:
	python scripts/bias_check.py
