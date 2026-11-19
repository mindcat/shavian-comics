# Shavian comics web tool, for a free CPU Hugging Face Space.
# OCR uses Groq's API when the GROQ_API_KEY secret is set, and falls back to
# Florence-2-base on the CPU otherwise. Everything else runs locally.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY --chown=user src/ src/
# Models, dictionaries and fonts (checksums pinned in init.py; YOLO isn't used here),
# plus the Florence-2 fallback OCR model, all baked into the image
RUN python src/init.py --skip model.pt < /dev/null \
    && python -c "from transformers import AutoProcessor, Florence2ForConditionalGeneration as F; \
m='florence-community/Florence-2-base'; F.from_pretrained(m); AutoProcessor.from_pretrained(m)"

ENV HF_HUB_OFFLINE=1 SHAVIAN_DEVICE=cpu \
    SHAVIAN_OCR=groq SHAVIAN_OCR_FALLBACK=florence \
    SHAVIAN_FLORENCE=florence-community/Florence-2-base SHAVIAN_FLORENCE_BEAMS=1 \
    SHAVIAN_MAX_JOBS=8 SHAVIAN_MAX_SIDE=2000 SHAVIAN_MAX_UPLOAD_MB=10 SHAVIAN_UPLOADS_PER_HOUR=30 \
    SHAVIAN_HOST=0.0.0.0 SHAVIAN_PORT=7860

EXPOSE 7860
CMD ["python", "src/server.py"]
