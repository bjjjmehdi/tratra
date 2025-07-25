FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "src.main"]