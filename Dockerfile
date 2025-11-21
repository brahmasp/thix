# 1. Use the Clean NVIDIA Base (Ubuntu 22.04)
# This has the drivers but NO Python yet.
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

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
    xvfb \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# 3. Install Python 3.11 and Critical Headers
# 'python3.11-dev' and 'distutils' are REQUIRED for building JAX/pip dependencies
RUN apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils

# 4. Make Python 3.11 the default "python" and "python3"
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    ln -s /usr/bin/python3.11 /usr/bin/python

# 5. Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# 6. Install JAX (Clean install on fresh Python)
RUN pip install --upgrade pip setuptools wheel
#RUN pip install "jax[cuda12]==0.4.35"
# 1. Install JAX for CUDA 12 first (as discussed)
#RUN pip install --upgrade "jax==0.4.35" "jaxlib==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jax[cuda12]"
RUN pip install --upgrade "jax==0.4.35"  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jax[cuda12]"

# 2. Install TF and TFP with the legacy Keras fix
RUN pip install "tensorflow[and-cuda]==2.18.0" "tensorflow-probability==0.25.0" "tf-keras"
#RUN pip install "jax[cuda12]"

# 7. Verify Installation (Build-time check)
RUN python3 -c "import jax; print('JAX Version:', jax.__version__)"

# 8. Install Your Code
COPY embodied/ ./embodied/
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied
RUN pip install -U -r embodied/requirements.txt

COPY dreamerv3/ ./dreamerv3/
RUN chown -R 1000:root /dreamerv3 && chmod -R 775 /dreamerv3
# IMPORTANT: Ensure dreamerv3/requirements.txt does NOT contain 'jax'
RUN pip install -U -r dreamerv3/requirements.txt 

COPY utils/ ./utils/
RUN chown -R 1000:root /utils && chmod -R 775 /utils


