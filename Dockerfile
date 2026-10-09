# ── BeatSync-Engine Docker image (Render / any cloud) ──────────────
# CPU-only build: no CUDA, no cupy.  FFmpeg + libass installed from
# the system package manager so lyrics burn-in works out of the box.

FROM python:3.13-slim

# ── System deps ────────────────────────────────────────────────────
#   ffmpeg           — video encoding/decoding (H.264, audio mux)
#   libass / fonts   — ASS subtitle rendering for lyrics overlay
#   fonts-dejavu     — fallback sans-serif glyphs
#   fontconfig       — font discovery for libass
#   mesa-utils       — headless GL stubs (avoids cv2 GPU init errors)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libass9 \
        libfreetype6 \
        libfontconfig1 \
        fonts-dejavu-core \
        fonts-dejavu-extra \
        fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

# Refresh the font cache so libass can find installed fonts.
RUN fc-cache -f

WORKDIR /app

# ── Python deps (cached layer) ─────────────────────────────────────
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# ── Application code ───────────────────────────────────────────────
COPY src/ ./src/
COPY assets/ ./assets/

# Output directory for rendered videos (Render uses ephemeral disk
# on the free plan — files survive across requests but not redeploys).
RUN mkdir -p output input

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1
ENV GRADIO_SERVER_NAME=0.0.0.0

# Render injects $PORT at runtime (default 10000 for local testing).
EXPOSE ${PORT:-10000}

# Gradio reads $PORT via the find_launch_port() logic in gui.py.
CMD ["python", "src/gui.py"]
