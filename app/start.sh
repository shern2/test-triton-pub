#!/bin/bash
set -e

# echo "Download models from cloud storage"
# download

echo "Starting Triton Inference Server..."

# --load-model=simple_net \
tritonserver \
--model-control-mode=explicit \
--load-model=finbert-model \
--load-model=finbert-tokenizer \
--load-model=finbert \
--model-repository=/models \
--http-port=8000 \
--grpc-port=8001 \
--metrics-port=8002 &
TRITON_PID=$!

# Wait for Triton to be ready
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v2/health/ready | grep -q "200"; do
    echo "Waiting for Triton Inference Server to be ready..."
    sleep 1
done
echo "Triton Inference Server is ready!"

# Start FastAPI Proxy
echo "Starting FastAPI Proxy..."
uvicorn main:app --host 0.0.0.0 --port 8005 --workers 2 --log-config log_config_uvicorn.yml &
FASTAPI_PID=$!

# Handle SIGTERM and SIGINT to shut down services gracefully
trap "echo 'Stopping services...'; kill $TRITON_PID; kill $FASTAPI_PID; exit" SIGTERM SIGINT

# Wait for both processes
wait $TRITON_PID
wait $FASTAPI_PID