FROM python:3.10-slim

ENV APP_DIR /workspace
WORKDIR ${APP_DIR}

COPY requirements.txt ${APP_DIR}

RUN apt-get update && \
    apt-get install -y nodejs npm && \  # 開発時のみ
    pip install --no-cache-dir -r requirements.txt

COPY ./src/ ${APP_DIR}/src/

# 環境変数設定
ENV PYTHONPATH="/workspace/src"

