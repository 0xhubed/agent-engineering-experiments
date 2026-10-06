#!/usr/bin/env bash
# Health + speed check for an OpenAI-compatible endpoint. Usage: check.sh http://hubed-dgx:8000/v1 qwen3.8-27b
set -euo pipefail
BASE="${1:-http://hubed-dgx:8000/v1}"; MODEL="${2:-qwen3.8-27b}"
curl -sf "$BASE/models" | python3 -c "import sys,json; print('models:', [m['id'] for m in json.load(sys.stdin)['data']])"
for thinking in false true; do
  start=$(date +%s.%N)
  out=$(curl -sf "$BASE/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\": \"$MODEL\", \"temperature\": 0, \"seed\": 0, \"max_tokens\": 1024,
    \"chat_template_kwargs\": {\"enable_thinking\": $thinking},
    \"messages\": [{\"role\": \"user\", \"content\": \"A note pays a 4.5% annual coupon on CHF 5,000 nominal, semi-annually. What is each coupon payment? End with: ANSWER: <amount>\"}]}")
  end=$(date +%s.%N)
  echo "$out" | python3 -c "
import sys, json
d = json.load(sys.stdin); m = d['choices'][0]['message']; u = d['usage']; s = $end - $start
print(f'thinking=$thinking  {s:.1f}s  completion_tokens={u[\"completion_tokens\"]}  {u[\"completion_tokens\"]/s:.1f} tok/s')
print('  content:', (m.get('content') or '').strip().splitlines()[-1][:120])
print('  reasoning chars:', len(m.get('reasoning_content') or m.get('reasoning') or ''))"
done
