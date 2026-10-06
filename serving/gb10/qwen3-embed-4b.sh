#!/usr/bin/env bash
# Embeddings for the vector arms: Qwen3-Embedding-4B on hubed-dgx, next to Qwen3.8-27B (capped at 0.62).
set -euo pipefail

IMAGE="vllm/vllm-openai@sha256:541e0e475418de6178b45c0d9ef420fb6be79bf43130a4d552cb668e425f4d27"  # tag qwen38-arm64-cu130
MODEL="Qwen/Qwen3-Embedding-4B"
REVISION="5cf2132abc99cad020ac570b19d031efec650f2b"
NAME="aex-qwen3-embed-4b"
PORT="${PORT:-8001}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped --gpus all --ipc host \
  -p "$PORT:8000" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  --entrypoint python3 "$IMAGE" -m vllm.entrypoints.openai.api_server \
  --model="$MODEL" --revision="$REVISION" \
  --served-model-name=qwen3-embedding-4b \
  --host=0.0.0.0 --port=8000 \
  --runner=pooling \
  --gpu-memory-utilization=0.12 \
  --max-model-len=8192
# This vLLM build has no --task; --runner=pooling serves /v1/embeddings.

echo "started $NAME on :$PORT — follow with: docker logs -f $NAME"
