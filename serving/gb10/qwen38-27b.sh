#!/usr/bin/env bash
# Primary navigator + answerer: Qwen3.8-27B (unsloth NVFP4) on hubed-dgx (GB10).
# Image and model revision are pinned so a rerun serves the identical stack.
# Flags follow the configuration already proven on this box (~/inference-server).
set -euo pipefail

IMAGE="vllm/vllm-openai@sha256:541e0e475418de6178b45c0d9ef420fb6be79bf43130a4d552cb668e425f4d27"  # tag qwen38-arm64-cu130
MODEL="unsloth/Qwen3.8-27B-NVFP4"
REVISION="7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
NAME="aex-qwen38-27b"
PORT="${PORT:-8000}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped --gpus all --ipc host \
  -p "$PORT:8000" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  --entrypoint python3 "$IMAGE" -m vllm.entrypoints.openai.api_server \
  --model="$MODEL" --revision="$REVISION" \
  --served-model-name=qwen3.8-27b \
  --host=0.0.0.0 --port=8000 \
  --seed=0 \
  --reasoning-parser=qwen3 \
  --enable-auto-tool-choice --tool-call-parser=qwen3_coder \
  '--speculative-config={"method":"mtp","num_speculative_tokens":2}' \
  --max-num-seqs=16 \
  --gpu-memory-utilization=0.62 \
  --max-model-len=262144
# 0.62 of 121 GiB: 22 GiB weights + ~50 GiB KV (4 x 262k-token contexts need ~36 GiB); the default 0.92
# reserved 85 GiB of KV and left nothing for the embedding and reranker servers.
# No --tensor-parallel-size (single GB10). No --kv-cache-dtype fp8: measured slower than f16 on Spark unified memory.
# Thinking is chosen per request (chat_template_kwargs.enable_thinking); the template defaults it ON.

echo "started $NAME on :$PORT — follow with: docker logs -f $NAME"
