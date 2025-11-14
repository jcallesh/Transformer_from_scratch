# Dockerfile
# Builds Docker images for installing and configuring pythorch with CUDA support.
# Make sure to set the right arguments for CUDA, cuDNN, and pytorch versions below.
# To use this file rename it to `Dockerfile` and run the build command.


# ---- Base image with CUDA and cuDNN ----
ARG CUDA="12.8.0"
ARG TAG="devel"
ARG OS="ubuntu22.04"
FROM nvidia/cuda:${CUDA}-cudnn-${TAG}-${OS}

# this is needed for texlive-fonts-extra and wget
ENV DEBIAN_FRONTEND=noninteractive

# ---- System packages ----
RUN apt-get update && \
    apt-get install -y \
        git \
        vim \
        python3 \
        python3-pip


# ---- Python setup ----
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade pip

# ---- Create non-root user ----
ARG USER_ID
ARG GROUP_ID
ARG NAME
RUN groupadd --gid ${GROUP_ID} ${NAME} && \
    useradd --no-log-init --create-home --uid ${USER_ID} --gid ${GROUP_ID} -s /bin/sh ${NAME}
USER ${USER_ID}:${GROUP_ID}

# ---- Working directory ----
ARG WORKDIR_PATH
WORKDIR ${WORKDIR_PATH}

# ---- Install PyTorch with CUDA support ----
# check your CUDA version and select the appropriate PyTorch version
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN python3 -m pip install tensorboard
RUN python3 -m pip install matplotlib
RUN python3 -m pip install datasets tokenizers
RUN python3 -m pip install tqdm

# ---- Clean up pip cache (frees space) ----
RUN python3 -m pip cache purge


# ---- Set default shell ----
CMD ["/bin/bash"]

####################################################################

# ----Build exmple command ----

# docker build --tag transformer-cuda \
# --build-arg CUDA="12.8.0" \
# --build-arg TAG="devel" \
# --build-arg OS="ubuntu22.04" \
# --build-arg USER_ID=$(id -u) \
# --build-arg GROUP_ID=$(id -g) \
# --build-arg NAME="user" \
# --build-arg WORKDIR_PATH=$(pwd) .

# ---- Run example command ----
# make sure to include the `--gpus all` flag to enable GPU support

# docker run --gpus all -d -it\
#     --name transformer_container \
#     -u $(id -u):$(id -g) \
#     -v $(pwd):$(pwd):rw \
#     transformer-cuda