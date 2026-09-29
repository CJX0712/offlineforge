FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY offlineforge ./offlineforge
COPY examples ./examples
COPY tests ./tests
COPY pyproject.toml README.md ./

CMD ["python", "-m", "offlineforge.cli"]
