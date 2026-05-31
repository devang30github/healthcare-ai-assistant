# ── Stage 1: Builder ─────────────────────────────────────────
# Install dependencies in a separate stage to keep final image lean
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Create required directories with correct ownership
RUN mkdir -p data vector_store logs frontend && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser app/        ./app/
COPY --chown=appuser:appuser frontend/   ./frontend/
COPY --chown=appuser:appuser data/       ./data/

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check — Docker will mark container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]