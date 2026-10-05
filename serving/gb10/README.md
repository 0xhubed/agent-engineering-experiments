# GB10 serving

| Script | Box | Model | Endpoint |
|---|---|---|---|
| `qwen38-27b.sh` | `hubed-dgx` | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4` (served as `qwen3.8-27b`) | `http://hubed-dgx:8000/v1` |

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
