# 1. Test setup:
# docker run -it --rm --gpus all tensorflow/tensorflow:2.8.0-gpu nvidia-smi
#
# 2. Start training:
# docker build -f  agents/director/Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#   sh xvfb_run.sh python3 agents/director/train.py \
#   --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs dmc_vision --task dmc_walker_walk
#
# 3. See results:
# tensorboard --logdir ~/logdir

# System
FROM tensorflow/tensorflow:2.20.0-gpu
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONUNBUFFERED 1
#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y \
  apt ffmpeg git python3-pip vim wget unrar libglib2.0-0 libsm6 libxext6 libxrender1 libosmesa6 freeglut3-dev libglu1-mesa libgl1-mesa-dri xvfb\
  && apt-get clean

# Envs
RUN pip3 install --no-cache-dir crafter
RUN pip3 install --no-cache-dir robodesk
RUN pip3 install --no-cache-dir dm_control
RUN pip3 install --no-cache-dir miniworld 
RUN pip3 install --no-cache-dir minihack 
RUN pip3 install --no-cache-dir nle 
ENV MUJOCO_GL egl

# Agent
RUN pip3 install --no-cache-dir dm-sonnet
ENV TF_FUNCTION_JIT_COMPILE_DEFAULT 1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8


# Embodied
COPY embodied/ ./embodied/
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied
RUN pip install -U -r embodied/requirements.txt
COPY dreamerv3/ ./dreamerv3/
RUN chown -R 1000:root /dreamerv3 && chmod -R 775 /dreamerv3
RUN pip install -U -r dreamerv3/requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
COPY utils/ ./utils/
RUN chown -R 1000:root /utils && chmod -R 775 /utils
#WORKDIR /embodied



CMD [ \
    "python3", "agents/director/train.py" \
    "--logdir=/logdir/$(date +%Y%m%d-%H%M%S)" \
    "--configs=dmc_vision", \
    "--task=dmc_walker_walk" \
]
