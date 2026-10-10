FROM python:3.11-slim

# xgboost needs the OpenMP runtime, which the slim image doesn't ship
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY prompts ./prompts
RUN pip install --no-cache-dir -e .

# data and models are mounted as volumes so the image stays small
ENV DATA_DIR=/app/data MODEL_DIR=/app/models
EXPOSE 8000
CMD ["uvicorn", "aml_copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
