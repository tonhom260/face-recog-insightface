FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# เพิ่ม build-essential และ python3-dev สำหรับ compile InsightFace
# และ libgl Libraries สำหรับ OpenCV
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    python3-dev 
    # libgl1-mesa-glx \
    # libglib2.0-0 \
    # libsm6 \
    # libxext6 \
    # libxrender-dev \
    # && apt-get clean \
    # && rm -rf /var/lib/apt/lists/*

# (Optional) Copy requirements ถ้าต้องการ
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp/