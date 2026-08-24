# SubCellSpace — spatial data processing environment.
#
# Builds the cross-platform pyxel environment (yamls/pyxel_env_crossplatform.yml) so the
# raw Xenium/MERFISH processing pipeline can be run without solving any conda environment
# locally. Using the YAML here rather than a bare pip install is deliberate: it means this
# image also serves as a build-time check that the published environment file resolves.
#
# Prebuilt images are published at https://hub.docker.com/r/woutdavid/subcellspace
# (tags: latest / cpu / cuda, plus immutable 1.0-cpu and 1.0-cuda), so most users do not
# need to build this at all:
#   docker run -it --rm -p 8888:8888 -v /path/to/your/data:/data woutdavid/subcellspace:latest
#
# Build:
#   docker build -t woutdavid/subcellspace:cuda .
#
# Run (mount the data you want to process):
#   docker run -it --rm -p 8888:8888 \
#       -v /path/to/your/data:/data \
#       woutdavid/subcellspace:latest
#   ...then open the http://127.0.0.1:8888/?token=... URL printed in the terminal.
#
# Push (`latest` should point at the CPU build — see the README tag table):
#   docker login
#   docker tag subcellspace:cpu woutdavid/subcellspace:latest
#   docker push woutdavid/subcellspace:latest
#
# Build a smaller CPU-only image (~3.2 GB less, no CUDA runtime) — recommended for
# reviewers and for anyone running the image on macOS/Windows, where the bundled CUDA
# wheels are dead weight:
#   docker build --build-arg TORCH_VARIANT=cpu -t <dockerhub-user>/subcellspace:cpu .
#
# Notes
#   * By default pip installs the CUDA-enabled torch wheels, so `gpu=True` in the
#     segmentation calls works when the container is started with `--gpus all`.
#     Without a GPU everything still runs on CPU (set gpu=False), just slower.
#   * classifiers/ (2.1 GB of pickled random forests) is excluded via .dockerignore.
#     Those are only needed for the paper-figure notebooks, not for processing; mount
#     them with `-v $(pwd)/classifiers:/opt/subcellspace/classifiers` if you need them.
FROM continuumio/miniconda3:24.9.2-0

LABEL org.opencontainers.image.title="SubCellSpace processing environment" \
      org.opencontainers.image.description="Environment for processing raw Xenium/MERFISH data into SubCellSpace embeddings" \
      org.opencontainers.image.source="https://github.com/WoutDavid/SubCellSpace" \
      org.opencontainers.image.licenses="MIT"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# libgl/libglib are needed by opencv, which pyxelperfect pulls in.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# conda-forge only; the environment file deliberately does not use the defaults channel.
RUN conda config --system --remove channels defaults 2>/dev/null || true \
 && conda config --system --add channels conda-forge \
 && conda config --system --set channel_priority strict

# "cuda" (default) keeps the GPU-capable torch wheels; "cpu" swaps in the CPU build and
# drops the nvidia runtime. The swap happens inside this same RUN so the space is actually
# reclaimed rather than merely shadowed by a later layer.
ARG TORCH_VARIANT=cuda

# PIP_NO_CACHE_DIR matters: conda clean does not touch pip's wheel cache, which otherwise
# leaves ~2.2 GB of downloaded wheels baked into this layer.
ENV PIP_NO_CACHE_DIR=1

COPY yamls/pyxel_env_crossplatform.yml /tmp/pyxel_env.yml
RUN conda env create -f /tmp/pyxel_env.yml \
 && if [ "$TORCH_VARIANT" = "cpu" ]; then \
        : "--no-deps is essential here: without it --force-reinstall also reinstalls" \
        : "torch's dependencies and drags in numpy 2.x over the pinned 1.26.4, which" \
        : "breaks torch's numpy bridge and scanpy's compiled ABI." ; \
        /opt/conda/envs/pyxel_env/bin/pip install --no-cache-dir --force-reinstall --no-deps \
            --index-url https://download.pytorch.org/whl/cpu \
            torch==1.13.1+cpu torchvision==0.14.1+cpu \
     && /opt/conda/envs/pyxel_env/bin/pip uninstall -y \
            nvidia-cublas-cu11 nvidia-cuda-nvrtc-cu11 \
            nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11 2>/dev/null || true; \
    fi \
 && conda clean -afy \
 && find /opt/conda/envs/pyxel_env -follow -type f -name '*.pyc' -delete \
 && rm -rf /root/.cache /tmp/pyxel_env.yml

# Put the environment first on PATH so no `conda activate` is needed.
ENV PATH=/opt/conda/envs/pyxel_env/bin:$PATH \
    CONDA_DEFAULT_ENV=pyxel_env \
    PYTHONDONTWRITEBYTECODE=1

# The pipeline code and the trained model.
COPY . /opt/subcellspace
ENV PYTHONPATH=/opt/subcellspace

WORKDIR /data
EXPOSE 8888

# Token stays enabled; copy the printed URL from the container logs.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.root_dir=/", "--ServerApp.preferred_dir=/opt/subcellspace"]
