FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY worker.py /app/worker.py
CMD ["python", "/app/worker.py"]
