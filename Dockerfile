FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# LightGBM (imported by the pickled bundle) needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY deploy/requirements.txt .
RUN pip install -r requirements.txt
COPY deploy/app ./app
COPY deploy/model ./model
RUN useradd -r -u 10001 api && chown -R api /srv
USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
