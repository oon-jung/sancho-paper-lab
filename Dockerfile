FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV HF_HOME=/app/.cache \
    TRANSFORMERS_VERBOSITY=error \
    OMP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Docling 레이아웃·표 모델을 이미지에 미리 받는다. 첫 요청이 몇 분씩 걸리는 것을 막는다.
RUN python -c "from docling.utils.model_downloader import download_models; download_models(with_easyocr=False)" || true

COPY app.py .
COPY static ./static
RUN mkdir -p gold_store && chmod -R 777 gold_store /app/.cache

EXPOSE 7860
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","7860"]
