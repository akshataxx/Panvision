# One image for both services; docker-compose.yml picks the command (FastAPI or Streamlit).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/panvision

# CPU-only torch/torchvision first so nnunetv2/monai don't pull the multi-GB CUDA build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./requirements.txt
COPY app/requirements.txt app/inference-requirements.txt ./app/
RUN pip install -r app/inference-requirements.txt

COPY .streamlit ./.streamlit
COPY src ./src
COPY app ./app

RUN useradd --create-home --uid 10001 panvision
USER panvision

EXPOSE 8001 8501
