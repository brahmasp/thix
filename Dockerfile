# 1. Use the Clean NVIDIA Base (Ubuntu 22.04)
# This has the drivers but NO Python yet.
FROM tensorflow/tensorflow:2.18.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl

# 2. Install System Utilities + Python PPA Prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    git \
    vim \
    unrar \
    ffmpeg \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libosmesa6 \
    freeglut3-dev \
    libglu1-mesa \
    xvfb 

# 6. Install JAX (Clean install on fresh Python)
RUN pip install --upgrade pip setuptools wheel

# 8. Install Your Code
COPY embodied/ ./embodied/
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied
RUN pip install -r embodied/requirements.txt

COPY dreamerv3/ ./dreamerv3/
RUN chown -R 1000:root /dreamerv3 && chmod -R 775 /dreamerv3
# IMPORTANT: Ensure dreamerv3/requirements.txt does NOT contain 'jax'
RUN pip install -r dreamerv3/requirements.txt 

COPY utils/ ./utils/
RUN chown -R 1000:root /utils && chmod -R 775 /utils

RUN pip install --upgrade --force-reinstall \
    "jax==0.4.28" \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "jaxlib==0.4.28+cuda12.cudnn89"

# 2. Install TF and TFP with the legacy Keras fix
RUN pip install  "tensorflow-probability==0.25.0" "tf-keras"

# 7. Verify Installation (Build-time check)
RUN python3 -c "import jax; print('JAX Version:', jax.__version__)"




