#!/usr/bin/env bash
# Cross-encoder reranker for hybrid_rerank: Qwen3-Reranker-4B on hubed-dgx.
set -euo pipefail

IMAGE="vllm/vllm-openai@sha256:541e0e475418de6178b45c0d9ef420fb6be79bf43130a4d552cb668e425f4d27"  # tag qwen38-arm64-cu130
MODEL="Qwen/Qwen3-Reranker-4B"
REVISION="22e683669bc0f0bd69640a1354a6d0aebcfeede5"
NAME="aex-qwen3-rerank-4b"
PORT="${PORT:-8002}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped --gpus all --ipc host \
  -p "$PORT:8000" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  --entrypoint python3 "$IMAGE" -m vllm.entrypoints.openai.api_server \
  --model="$MODEL" --revision="$REVISION" \
  --served-model-name=qwen3-reranker-4b \
  --host=0.0.0.0 --port=8000 \
  --runner=pooling \
  '--hf-overrides={"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --gpu-memory-utilization=0.12 \
  --max-model-len=8192
# The hf-overrides turn the generative checkpoint into a yes/no classifier, as the model card's vLLM section shows.

echo "started $NAME on :$PORT — follow with: docker logs -f $NAME"
