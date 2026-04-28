# Use Miniconda base image
FROM continuumio/miniconda3:22.11.1

# Set working directory inside container
WORKDIR /app

# Copy the environment.yml and patch file into the container
COPY analysis_figures.yml .

# Create the environment
RUN conda env create -f analysis_figures.yml

# Install Jupyter if it's not already in your environment
RUN conda run -n subcellspace_env conda install jupyter -y

# Expose Jupyter port
EXPOSE 8888

# Default shell to ensure environment is activated
SHELL ["conda", "run", "-n", "subcellspace_env", "/bin/bash", "-c"]

# Start Jupyter Notebook on container start
CMD ["bash", "-c", "conda run -n subcellspace_env jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/app"]
