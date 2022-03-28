FROM python:3.10-slim

ENV APP_DIR /workspace
WORKDIR ${APP_DIR}

COPY requirements.txt ${APP_DIR}

RUN apt-get update && \
    apt-get install -y nodejs npm && \
    pip install --no-cache-dir -r requirements.txt

RUN apt-key adv --keyserver hkps://keyserver.ubuntu.com:443 \
                --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN apt-get install -y r-base r-base-dev

# 環境変数設定
ENV PYTHONPATH /workspace
