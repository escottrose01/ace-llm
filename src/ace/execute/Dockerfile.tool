FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY tool_runner.py /app/tool_runner.py
CMD ["python", "/app/tool_runner.py"]
