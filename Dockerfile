FROM node:22-bookworm-slim AS frontend-build
WORKDIR /build/frontend
ARG VITE_MIN_FACE_LUMA=45
ENV VITE_MIN_FACE_LUMA=${VITE_MIN_FACE_LUMA}
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=secret,id=cloud_ca \
    if [ -f /run/secrets/cloud_ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/cloud_ca; fi; \
    npm ci --no-audit --no-fund
COPY frontend/ ./
RUN test -s public/models/face_landmarker.task && npm run build

FROM python:3.11-slim-bookworm AS cpu-runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ATTENDANCE_MODEL_DIR=/app/.local/models ATTENDANCE_INFERENCE_DEVICE=cpu
WORKDIR /app
COPY requirements.txt ./
RUN --mount=type=secret,id=cloud_ca \
    if [ -f /run/secrets/cloud_ca ]; then export PIP_CERT=/run/secrets/cloud_ca; fi; \
    pip install --no-cache-dir -r requirements.txt
COPY attendance/ attendance/
COPY main.py ./
COPY scripts/ scripts/
COPY --from=frontend-build /build/frontend/dist/ frontend/dist/
RUN useradd --create-home --uid 10001 attendance && mkdir -p /app/.local/models \
    && chown -R attendance:attendance /app/.local && chmod -R a+rX /app
USER attendance
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]
ENTRYPOINT ["sh", "scripts/start_container.sh"]

# Optional target: CUDA 12 + cuDNN 9; requires NVIDIA Container Toolkit/GPU.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cuda
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ATTENDANCE_MODEL_DIR=/app/.local/models ATTENDANCE_INFERENCE_DEVICE=cuda
WORKDIR /app
RUN --mount=type=secret,id=cloud_ca \
    sed -i 's|http://|https://|g' /etc/apt/sources.list \
    && if [ -f /run/secrets/cloud_ca ]; then \
      apt-get -o Acquire::https::CaInfo=/run/secrets/cloud_ca update \
      && apt-get -o Acquire::https::CaInfo=/run/secrets/cloud_ca install -y --no-install-recommends python3 python3-pip libglib2.0-0 libgomp1; \
      else apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libglib2.0-0 libgomp1; fi \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-cuda.txt ./
RUN --mount=type=secret,id=cloud_ca \
    if [ -f /run/secrets/cloud_ca ]; then export PIP_CERT=/run/secrets/cloud_ca; fi; \
    python -m pip install --no-cache-dir -r requirements-cuda.txt
COPY attendance/ attendance/
COPY main.py ./
COPY scripts/ scripts/
COPY --from=frontend-build /build/frontend/dist/ frontend/dist/
RUN useradd --create-home --uid 10001 attendance && mkdir -p /app/.local/models \
    && chown -R attendance:attendance /app/.local && chmod -R a+rX /app
USER attendance
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]
ENTRYPOINT ["sh", "scripts/start_container.sh"]

# CPU is the default final build target.
FROM cpu-runtime AS cpu
