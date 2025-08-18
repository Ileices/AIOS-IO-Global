# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY aios_io ./aios_io
RUN pip install --no-cache-dir .[science]

CMD ["python", "-m", "aios_io.cli"]
