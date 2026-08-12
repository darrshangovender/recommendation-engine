.PHONY: install data test eval bench clean

install:
	pip install -e ".[dev]"

data:
	python -c "from recommender.data.loader import load_movielens_100k; load_movielens_100k()"

test:
	pytest -v

eval:
	python -m recommender.pipeline

bench:
	python benchmarks/run.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
