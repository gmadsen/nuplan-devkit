FROM ubuntu:20.04

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
    curl \
    gnupg2 \
    software-properties-common \
    default-jdk \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn

# Install NVIDIA Docker support and Bazel
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | apt-key add - \
    && curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list \
    && add-apt-repository "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
    && apt-get update \
    && apt-get install -y \
    bazel \
    file \
    zip \
    nvidia-container-toolkit \
    software-properties-common \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.9 python3.9-dev python3.9-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Setup working environment
ARG NUPLAN_HOME=/nuplan_devkit
WORKDIR $NUPLAN_HOME

# Copy dependency files
COPY pyproject.toml uv.lock .python-version /nuplan_devkit/

# Copy source code
RUN mkdir -p $NUPLAN_HOME/nuplan
COPY nuplan $NUPLAN_HOME/nuplan

# Install dependencies with uv (CUDA enabled)
# Use --system to install in the container's Python instead of creating a venv
RUN uv sync --frozen --all-extras --no-dev

# Install nuplan-devkit in editable mode
RUN uv pip install --system -e .

ENV NUPLAN_MAPS_ROOT=/data/sets/nuplan/maps \
    NUPLAN_DATA_ROOT=/data/sets/nuplan \
    NUPLAN_EXP_ROOT=/data/exp/nuplan

RUN bash -c 'mkdir -p {$NUPLAN_MAPS_ROOT,$NUPLAN_DATA_ROOT,$NUPLAN_EXP_ROOT}'

ARG NUPLAN_CHALLENGE_DATA_ROOT_S3_URL
ARG NUPLAN_CHALLENGE_MAPS_ROOT_S3_URL
ARG NUPLAN_SERVER_S3_ROOT_URL
ARG S3_TOKEN_DIR
ARG NUPLAN_DATA_STORE

ENV NUPLAN_DATA_ROOT $NUPLAN_DATA_ROOT
ENV NUPLAN_MAPS_ROOT $NUPLAN_MAPS_ROOT
ENV NUPLAN_DB_FILES  /data/sets/nuplan/nuplan-v1.1/splits/mini
ENV NUPLAN_MAP_VERSION "nuplan-maps-v1.0"
ENV NUPLAN_DATA_STORE $NUPLAN_DATA_STORE
ENV NUPLAN_S3_PROFILE "default"
ENV NUPLAN_DATA_ROOT_S3_URL $NUPLAN_CHALLENGE_DATA_ROOT_S3_URL
ENV NUPLAN_MAPS_ROOT_S3_URL $NUPLAN_CHALLENGE_MAPS_ROOT_S3_URL
ENV NUPLAN_SERVER_S3_ROOT_URL $NUPLAN_SERVER_S3_ROOT_URL
ENV S3_TOKEN_DIR $S3_TOKEN_DIR

RUN bash -c 'mkdir -p $NUPLAN_DB_FILES'

CMD ["/nuplan_devkit/nuplan/entrypoint_simulation.sh"]
