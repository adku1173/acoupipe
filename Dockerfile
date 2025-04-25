# Stage 1: Base image with Miniconda
FROM continuumio/miniconda3:latest AS base

# Set environment variables
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV PATH="/opt/conda/envs/acoupipe/bin:$PATH"

# Set the working directory
WORKDIR /src

# Create and activate the `acoupipe` environment, and install Python
RUN conda install -y -n base python=3.12 && \
    conda clean --all --yes

# Activate the `acoupipe` environment and install essential tools
RUN conda install -y -n base -c conda-forge git && \
    conda clean --all --yes

# Clone and install Acoular in the `acoupipe` environment
RUN git clone --branch pickle_fix --single-branch https://github.com/acoular/acoular.git /tmp/acoular && \
    pip install /tmp/acoular && \
    rm -rf /tmp/acoular

# Copy project files
COPY . /src

# Install project dependencies in the `acoupipe` environment
RUN pip install .


# Default command to run the application
CMD ["python", "/src/app/main.py"]

# Stage 2: Full build with optional dependencies
FROM base AS full

# Install optional dependencies in the `acoupipe` environment
RUN pip install ".[full]"

# Stage 3: Development build
FROM full AS dev

# Install development dependencies in the `acoupipe` environment
RUN pip install ".[dev]"