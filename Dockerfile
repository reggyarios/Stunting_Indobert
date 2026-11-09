# Backend Flask (CURE + IndoBERT-RCNN) untuk Hugging Face Spaces (SDK Docker, CPU). Lihat DEPLOY.md.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
# torch versi CPU (jauh lebih kecil dari build CUDA), lalu dependensi lain
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Unduh backbone IndoBERT + tokenizer saat build agar server tidak mengunduh saat permintaan pertama
RUN python -c "from transformers import AutoModel, AutoTokenizer; m='indobenchmark/indobert-base-p1'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

COPY *.py ./
COPY templates/ templates/
COPY checkpoints/ checkpoints/

RUN useradd -m -u 1000 user && mkdir -p uploads outputs && chown -R user /app
USER user

EXPOSE 7860
# 1 worker: checkpoint dimuat sekali di memori; timeout panjang karena CURE hierarkis O(n^2) per lembar
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 1800 app:app
