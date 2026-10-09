FROM python:3.11-slim

WORKDIR /workspace

# Install system deps for matplotlib, networkx
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install package (runtime only, no dev extras)
RUN pip install --no-cache-dir -e .

# Default: run one sequence-classification smoke epoch to verify install
CMD ["python", "-c", \
     "from graft_net.train.trainer import Trainer; \
      m = Trainer.for_smoke_test().run_smoke_epoch(); \
      print('Smoke test passed:', m)"]
