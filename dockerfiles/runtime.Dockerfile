ARG BASE_IMAGE=nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.10
ARG MODEL_DIR=/opt/models
ARG APP_DIR=/opt/app
ARG ENGINE_DIR=/opt/engine
ARG MODEL_NAME=vtol_hover
ARG METADATA_FILE=observations_metadata.yaml

ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_DIR=${MODEL_DIR}
ENV APP_DIR=${APP_DIR}
ENV ENGINE_DIR=${ENGINE_DIR}

RUN apt-get update && apt-get install -y --no-install-recommends \
  python${PYTHON_VERSION} \
  python3-pip \
  python${PYTHON_VERSION}-venv \
  python3-numpy \
  python3-yaml \
  python3-libnvinfer \
  && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
  ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --no-cache-dir --default-timeout=300 cuda-python

RUN mkdir -p ${MODEL_DIR}/${MODEL_NAME} ${APP_DIR} ${ENGINE_DIR}

COPY dockerfiles/run_inference.py ${APP_DIR}/
COPY engine-output/ ${ENGINE_DIR}/
COPY models/${MODEL_NAME}/${METADATA_FILE} ${MODEL_DIR}/${MODEL_NAME}/${METADATA_FILE}

WORKDIR ${APP_DIR}
CMD ["python3", "run_inference.py"]
