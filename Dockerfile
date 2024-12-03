FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

WORKDIR /app

RUN apt -y update && \
  apt install -y --no-install-recommends git python3-pip python3-dev python-is-python3 && \
  rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements.txt

RUN --mount=type=cache,mode=0755,target=/root/.cache \
  pip install -r requirements.txt

COPY ./src ./src

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
