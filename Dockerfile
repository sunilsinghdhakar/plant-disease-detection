
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api:app --host 127.0.0.1 --port 8000 & streamlit run app.py --server.address 0.0.0.0 --server.port 7860"]
