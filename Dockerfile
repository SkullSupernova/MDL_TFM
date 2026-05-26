FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE "^(torch|torchvision)==" requirements.txt > /tmp/req-api.txt \
    && pip install --no-cache-dir -r /tmp/req-api.txt

COPY src/ src/
COPY config/ config/

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
