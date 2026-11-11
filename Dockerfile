# RFP Intelligence: REST API by default, Streamlit UI on request.
#   docker run -p 8000:8000 ghcr.io/mayanksinghmewari/bid-intel                    -> API, docs at /docs
#   docker run -p 8501:8501 ghcr.io/mayanksinghmewari/bid-intel \
#       streamlit run app.py --server.address 0.0.0.0                                -> UI
# Optional at run time: -e GEMINI_API_KEY=... (LLM answers), -e LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL (tracing).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    BID_INTEL_ENV=production

RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app

WORKDIR /app
RUN chown app:app /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app . .
USER app

# Build the search index into the image: the embedding model is downloaded and every chunk embedded
# once, at build time, so containers answer immediately.
RUN BID_INTEL_LLM=off BID_INTEL_TRACING=off python -m bid_intel index --bid data/bid_folders/Bid1 \
 && BID_INTEL_LLM=off BID_INTEL_TRACING=off python -m bid_intel index --bid data/bid_folders/Bid2

ARG GIT_SHA=unknown
ENV BID_INTEL_RELEASE=$GIT_SHA

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD curl -fs http://localhost:8000/health || curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["python", "-m", "bid_intel", "serve", "--host", "0.0.0.0", "--port", "8000"]
