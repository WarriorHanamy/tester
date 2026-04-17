ARG BASE_IMAGE=nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  libnvinfer-bin \
  tensorrt-dev \
  && rm -rf /var/lib/apt/lists/*

RUN if [ -x /usr/src/tensorrt/bin/trtexec ]; then \
      ln -sf /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec; \
    elif [ -x /usr/bin/trtexec ]; then \
      ln -sf /usr/bin/trtexec /usr/local/bin/trtexec; \
    else \
      echo "trtexec not found after TensorRT installation" >&2; \
      exit 1; \
    fi

ENTRYPOINT ["trtexec"]
