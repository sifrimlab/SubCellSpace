# SubCellSpace — spatial data processing environment.
#
# This image ships the ENVIRONMENT ONLY (yamls/pyxel_env_crossplatform.yml); the pipeline
# code and the trained model come from the repository you mount at run time. That keeps the
# image from going stale every time the code changes — it only needs rebuilding when a
# dependency moves. Building from the YAML is also deliberate: it makes the image a
# build-time check that the published environment file actually resolves.
#
# Prebuilt images: https://hub.docker.com/r/woutdavid/subcellspace
#
# Run — mount the repo (which carries the code and the trained model) and your data:
#   docker run -it --rm -p 8888:8888 \
#       -v $(pwd):/workspace \
#       -v /path/to/your/data:/data \
#       woutdavid/subcellspace:latest
#   ...then open the http://127.0.0.1:8888/?token=... URL printed in the terminal.
#
# Run the pipeline inside the container (from /workspace, which is the repo root):
#   docker run -it --rm -v $(pwd):/workspace -v /path/to/your/data:/data \
#       woutdavid/subcellspace:latest \
#       python -m raw_spatial_data_processing.run raw_spatial_data_processing/configs/mine.yaml
#
# Build:
#   docker build -t woutdavid/subcellspace:cuda .                          # GPU-capable
#   docker build --build-arg TORCH_VARIANT=cpu -t woutdavid/subcellspace:cpu .   # ~3.2 GB smaller
#
# Push (`latest` should point at the CPU build — see the README tag table):
#   docker login
#   docker tag subcellspace:cpu woutdavid/subcellspace:latest
#   docker push woutdavid/subcellspace:latest
#
# Notes
#   * By default pip installs the CUDA-enabled torch wheels, so GPU segmentation works when
#     the container is started with `--gpus all`. Without a GPU everything still runs on CPU
#     (set gpu=False / use a CPU-only image), just slower.
#   * Because no code is baked in, nothing needs excluding from the build context beyond the
#     conda YAML — see .dockerignore.
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

# No code is copied in: mount the repository at /workspace instead. PYTHONPATH points
# there so `raw_spatial_data_processing` and the top-level `CVAE` module (needed to
# unpickle the trained model) are importable from any working directory.
ENV PYTHONPATH=/workspace

WORKDIR /workspace
EXPOSE 8888

# Token stays enabled; copy the printed URL from the container logs.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.root_dir=/", "--ServerApp.preferred_dir=/workspace"]
