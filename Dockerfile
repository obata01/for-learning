FROM python:3.10-slim

ENV APP_DIR /workspace
WORKDIR ${APP_DIR}

COPY requirements.txt ${APP_DIR}

RUN apt-get update && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# COPY ./src/ ${APP_DIR}/src/

# 環境変数設定
ENV PYTHONPATH="/workspace/src"

# PyTorch
ARG PYTORCH_URL=https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html 
RUN pip install --pre torch torchvision -f ${PYTORCH_URL}

