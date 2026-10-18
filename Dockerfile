# syntax=docker/dockerfile:1
# Multi-stage image for URPS.
#   docker build --target test .   -> runs lint and the full test suite inside the container
#   docker build -t urps .         -> slim runtime image with the CLI and the Streamlit dashboard
#   docker run -p 8501:8501 urps   -> dashboard on http://localhost:8501
#   docker run urps run --data data/sample/the_rankings_sample.csv --out /tmp/reports

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# ---- build: package the project as a wheel ----
FROM base AS build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /dist .

# ---- test: same sources plus dev tools; the build fails if any check fails ----
FROM base AS test
COPY pyproject.toml constraints.txt README.md LICENSE ./
COPY src ./src
# The pip cache lives in a BuildKit cache mount: downloads are reused between builds, not stored in the image
RUN --mount=type=cache,target=/root/.cache/pip pip install -e ".[dev,app]" -c constraints.txt
COPY tests ./tests
COPY data/sample ./data/sample
RUN ruff check . && pytest -p no:cacheprovider

# ---- runtime: only the wheel, its dependencies and the sample data ----
FROM base AS runtime
LABEL org.opencontainers.image.title="urps" \
      org.opencontainers.image.description="University Ranking Prediction System: CLI and Streamlit dashboard" \
      org.opencontainers.image.source="https://github.com/merxv/university-ranking-prediction" \
      org.opencontainers.image.licenses="MIT"
COPY constraints.txt /tmp/constraints.txt
COPY --from=build /dist /tmp/dist
RUN --mount=type=cache,target=/root/.cache/pip WHEEL=$(ls /tmp/dist/*.whl) \
    && pip install "${WHEEL}[app]" -c /tmp/constraints.txt \
    && rm -rf /tmp/dist /tmp/constraints.txt \
    && useradd --create-home --uid 1000 urps
COPY data/sample ./data/sample
USER urps

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["urps"]
CMD ["app"]
