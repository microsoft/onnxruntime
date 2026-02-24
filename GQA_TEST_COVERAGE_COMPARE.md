# GQA Test Coverage Comparison

Comparison of feature coverage between the CPU Python test (`test_gqa_cpu.py`) and the WebGPU JSONC test (`group-query-attention.jsonc`). The "WebGPU impl supports?" column indicates whether the WebGPU GroupQueryAttention kernel actually implements the feature, regardless of test coverage.

| Feature | test_gqa_cpu.py (CPU) | group-query-attention.jsonc (WebGPU) | WebGPU impl supports? |
|---|---|---|---|
| Basic GQA (Q, K, V, past_key, past_value) | Yes | Yes | Yes |
| num_heads != kv_num_heads (GQA grouping) | Yes | Yes | Yes |
| Past KV cache (non-empty past) | Yes | Yes | Yes |
| PackedQKV (key/value = null) | Yes | Yes (tests 14, 16) | Yes |
| Batch size > 1 | Yes | Yes (tests 8, 13) | Yes |
| Multi-sequence length (seq_len > 1) | Yes | Yes (tests 5–13) | Yes |
| head_sink | Yes | **No** | Yes |
| smooth_softmax | Yes | **No** | Yes |
| softcap | Yes | **No** | Yes |
| do_rotary | Yes | **No** | Yes |
| rotary_interleaved | Yes | **No** | Yes |
| local_window_size (sliding window) | Yes | **No** | Yes |
| position_ids | Yes | **No** | Yes |
| attention_bias | Yes | **No** | Yes |
| QK output (debug output) | Yes | **No** | N/A |
| float16 | Yes | **No** (all float32) | Yes |
