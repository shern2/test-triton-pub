FROM nvcr.io/nvidia/tritonserver:24.12-py3
# FROM nvcr.io/nvidia/tritonserver:24.12-pyt-python-py3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=UTF-8

RUN apt-get update
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     xxx1 \
#     xxx2 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# [ss] Consider uncommenting for actual deployment OR pull the models from cloud storage / mounted volume
# COPY ./models /models

COPY ./app /app
RUN chmod +x start.sh

EXPOSE 8000 8001 8002 8005

# # Use a non-root user for better security
# RUN useradd -ms /bin/bash appuser
# USER appuser

# Start both Triton Server and FastAPI Proxy
CMD ["/bin/bash", "start.sh"]
