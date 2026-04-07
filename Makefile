.PHONY: setup seed run test clean docker-up docker-down

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	cp .env.example .env
	@echo "Edit .env with your API keys and DATABASE_URL"

seed:
	PYTHONPATH=. python data/seed.py

run:
	PYTHONPATH=. streamlit run ui/app.py

test:
	PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	rm -rf __pycache__ .pytest_cache

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
