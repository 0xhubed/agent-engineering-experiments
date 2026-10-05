# GB10 serving

| Script | Box | Model | Endpoint |
|---|---|---|---|
| `qwen38-27b.sh` | `hubed-dgx` | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4` (served as `qwen3.8-27b`) | `http://hubed-dgx:8000/v1` |
| `qwen3-embed-4b.sh` | `hubed-dgx` | `Qwen/Qwen3-Embedding-4B` @ `5cf2132` (served as `qwen3-embedding-4b`, dim 2560) | `http://hubed-dgx:8001/v1` |
| `qwen3-rerank-4b.sh` | `hubed-dgx` | `Qwen/Qwen3-Reranker-4B` @ `22e6836` (served as `qwen3-reranker-4b`) | `http://hubed-dgx:8002/v1` |

Run the launch script on the box (`bash qwen38-27b.sh`), then from the ThinkCentre:

    serving/gb10/check.sh http://hubed-dgx:8000/v1 qwen3.8-27b

MTP speculative decoding follows unsloth's Qwen3.8 guide (`{"method":"mtp","num_speculative_tokens":2}`).
Confirm it is active with `docker logs aex-qwen38-27b | grep SpecDecoding` — mean acceptance length near 3 means both drafted tokens are usually accepted.

Thinking is per request: `chat_template_kwargs: {"enable_thinking": false}` or `{"reasoning_effort": "xhigh" | "medium" | "low"}` (template default: thinking on, `xhigh`).

## Measured 2026-10-05 (single stream, temperature 0)

| Mode | Decode | Notes |
|---|---|---|
| thinking off | 22.1 tok/s | 249 tokens |
| thinking on (xhigh) | 23.2 tok/s | 143 tokens + 359 reasoning chars |

MTP: mean acceptance length 2.85–2.92, per-position acceptance 0.97 / 0.94.

## Memory on hubed-dgx (121 GiB unified)

| Server | `--gpu-memory-utilization` |
|---|---|
| Qwen3.8-27B | 0.62 (22 GiB weights, 48 GiB KV = 1.38M tokens) |
| Qwen3-Embedding-4B | 0.12 |
| Qwen3-Reranker-4B | 0.12 |

All three running leaves ~6 GiB available — fine for serving, not for other jobs on the box.

## Reranker prompt template (applied client-side)

This vLLM build does not apply Qwen3-Reranker's prompt format on `/v1/rerank`. Without it the scores barely separate
relevant from irrelevant (0.61 vs 0.68 on a barrier/coupon pair); with it, 0.999 vs 0.004 (and 0.997 for the German
passage). The harness reranker client wraps query and documents (`query_template` / `document_template` in the run config):

    query:    <|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n
    document: <Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
