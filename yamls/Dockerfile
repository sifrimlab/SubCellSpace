# SubCellSpace — analysis environment: the paper figures and the additional analyses.
#
# This image ships the ENVIRONMENT ONLY (yamls/analysis_figures_crossplatform.yml); the
# notebooks, the trained model and the classifiers come from the repository you mount at run
# time, and the data from the folder you mount at /data. Like the processing image (the
# Dockerfile at the repository root) it only needs rebuilding when a dependency moves, and
# building it from the YAML makes the image a build-time check that the file still resolves.
#
# Build from the repository root (BuildKit picks up yamls/Dockerfile.dockerignore):
#   docker build -f yamls/Dockerfile -t woutdavid/subcellspace:analysis-cuda .
#   docker build -f yamls/Dockerfile --build-arg TORCH_VARIANT=cpu \
#       -t woutdavid/subcellspace:analysis-cpu .
#
# Run — mount the repo and your data, then open the http://127.0.0.1:8888/?token=... URL
# printed in the terminal:
#   docker run -it --rm -p 8888:8888 \
#       -v $(pwd):/workspace \
#       -v /path/to/your/data:/data \
#       woutdavid/subcellspace:analysis-cpu
#
# The additional analyses read their cached inputs from /data/additional_analysis (override
# with -e ADDITIONAL_ANALYSIS_DATA=...). The figure notebooks read the processed data from the
# main data record; set data_dir in each notebook to its location under /data.
#
# Notes
#   * Fig. 1B simulates its examples with simfish and needs the simulation environment
#     (yamls/make_simulations.yml), which this image does not contain.
#   * The benchmark has its own, smaller environment (benchmark/envs/benchmark.yml).
#   * The classic notebook interface is used because the interactive figure features
#     (clicking observations, drawing patterns) work best there.
FROM continuumio/miniconda3:24.9.2-0

LABEL org.opencontainers.image.title="SubCellSpace analysis environment" \
      org.opencontainers.image.description="Environment for reproducing the SubCellSpace figures and additional analyses" \
      org.opencontainers.image.source="https://github.com/WoutDavid/SubCellSpace" \
      org.opencontainers.image.licenses="MIT"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# conda-forge only; the environment file deliberately does not use the defaults channel.
RUN conda config --system --remove channels defaults 2>/dev/null || true \
 && conda config --system --add channels conda-forge \
 && conda config --system --set channel_priority strict

# "cuda" (default) installs the GPU-capable torch wheels; "cpu" adds PyTorch's CPU index, so
# pip resolves torch==1.13.1 / torchvision==0.14.1 to their +cpu builds directly and the
# nvidia runtime is never downloaded.
ARG TORCH_VARIANT=cuda

# conda clean does not touch pip's wheel cache, which would otherwise stay in this layer.
ENV PIP_NO_CACHE_DIR=1

COPY yamls/analysis_figures_crossplatform.yml /tmp/analysis_env.yml
RUN if [ "$TORCH_VARIANT" = "cpu" ]; then \
        export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu; \
    fi \
 && conda env create -f /tmp/analysis_env.yml \
 && conda clean -afy \
 && find /opt/conda/envs/subcellspace_env -follow -type f -name '*.pyc' -delete \
 && rm -rf /root/.cache /tmp/analysis_env.yml

# Put the environment first on PATH so no `conda activate` is needed.
ENV PATH=/opt/conda/envs/subcellspace_env/bin:$PATH \
    CONDA_DEFAULT_ENV=subcellspace_env \
    PYTHONDONTWRITEBYTECODE=1 \
    ADDITIONAL_ANALYSIS_DATA=/data/additional_analysis \
    MPLCONFIGDIR=/tmp/matplotlib

# No code is copied in: mount the repository at /workspace. PYTHONPATH points there so the
# top-level CVAE module (needed to unpickle the trained model) imports from any notebook.
ENV PYTHONPATH=/workspace

WORKDIR /workspace
EXPOSE 8888

# Token stays enabled; copy the printed URL from the container logs.
CMD ["jupyter", "nbclassic", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--notebook-dir=/workspace"]
