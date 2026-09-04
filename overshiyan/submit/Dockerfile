FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    PIP_INDEX_URL=https://mirrors.sustech.edu.cn/pypi/web/simple \
    PIP_NO_DEPS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash bzip2 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Download to disk and let curl fail before extraction. This avoids the corrupt
# pipe failure seen when a proxy returned a partial micromamba response.
RUN set -eux; \
    curl -fL --retry 8 --retry-all-errors --retry-delay 5 \
        --connect-timeout 30 --max-time 300 \
        -o /tmp/micromamba.tar.bz2 \
        https://micro.mamba.pm/api/micromamba/linux-64/latest; \
    tar -tjf /tmp/micromamba.tar.bz2 >/dev/null; \
    tar -xjf /tmp/micromamba.tar.bz2 \
        -C /usr/local/bin --strip-components=1 bin/micromamba; \
    rm -f /tmp/micromamba.tar.bz2

COPY submit/environment.yml /tmp/environment.yml
RUN micromamba create -y -n detector -f /tmp/environment.yml \
    && micromamba clean --all --yes

ENV PATH=/opt/micromamba/envs/detector/bin:/opt/micromamba/bin:$PATH

COPY src/inference /app/inference
COPY submit/app/main.py /app/main.py
COPY submit/models /app/models
COPY submit/model_manifest.json /app/model_manifest.json

RUN test -s /app/models/best.pt \
    && mkdir -p /input /output /tmp/Ultralytics

ENTRYPOINT ["python", "/app/main.py"]
