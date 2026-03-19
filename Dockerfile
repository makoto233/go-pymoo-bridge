FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./

RUN python -m pip install --upgrade pip && \
    python -m pip install \
        fastapi>=0.115.12 \
        pydantic>=2.12.5 \
        pymoo>=0.6.1.6 \
        numpy>=2.3.3 \
        uvicorn>=0.34.0

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
