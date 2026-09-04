FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    PIP_INDEX_URL=https://mirrors.sustech.edu.cn/pypi/web/simple \
    PIP_NO_DEPS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl -fL --retry 8 --retry-all-errors --retry-delay 5 \
        --connect-timeout 30 --max-time 300 \
        -o /tmp/micromamba.tar.bz2 \
        https://micro.mamba.pm/api/micromamba/linux-64/latest; \
    tar -xvjf /tmp/micromamba.tar.bz2 \
        -C /usr/local/bin --strip-components=1 bin/micromamba; \
    rm -f /tmp/micromamba.tar.bz2

COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n detector -f /tmp/environment.yml \
    && micromamba clean --all --yes

ENV PATH=/opt/micromamba/envs/detector/bin:/opt/micromamba/bin:$PATH

COPY app /app
COPY models /app/models

RUN mkdir -p /input /output

ENTRYPOINT ["python", "/app/main.py"]
