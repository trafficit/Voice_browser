FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium chromium-driver \
        portaudio19-dev libasound2 libsndfile1 ffmpeg libgomp1 \
        pulseaudio-utils espeak-ng \
        fonts-liberation libnss3 libgbm1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

CMD ["python", "main.py"]
