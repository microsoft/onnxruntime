import torch
import os
import sys

# sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
os.environ['ORT_DISABLE_FLASH_ATTENTION'] = '0'

from test_gqa import GQAConfig, gqa_prompt_func, apply_rotary_embedding, attention_ref
from einops import rearrange
import math
from onnx import TensorProto
import numpy as np

# Exact config from the failing test
# test_gqa_prompt_flash_attention_35_b1_sq240_skv240_nh3_1_h160_w_1_rotTrueTrue_pkdFalse_sbTrue_sc0_0_smFalse_True_pidTrue
config = GQAConfig(
    batch_size=16,
    q_sequence_length=1,
    kv_sequence_length=1,
    past_kv_sequence_length=2047,
    buffer_sequence_length=2048,
    num_heads=64,
    kv_num_heads=1,
    head_size=128,
    local_window_size=-1,
    rotary=True,
    rotary_interleaved=True,
    packed=False,
    share_buffer=True,
    softcap=0.0,
    use_smooth_softmax=False,
    has_head_sink=False,
    has_position_ids=True,
)

print(f"Config: {config}", flush=True)

device = 'cpu'
# The failure showed mismatches in floats, test uses float16 usually?
# Error message: "0.022735595703125 (ACTUAL), 0.2333984375 (DESIRED)"
# These look like float values.
# Let's check test_gqa.py for dtype. Default is likely float16.
torch_type = torch.float16
ort_type = TensorProto.FLOAT16

torch.manual_seed(0)
std = 0.2
q = torch.randn(config.batch_size, config.q_sequence_length, config.num_heads, config.head_size, device=device, dtype=torch_type) * std
k = torch.randn(config.batch_size, config.kv_num_heads, config.buffer_sequence_length, config.head_size, device=device, dtype=torch_type) * std
v = torch.randn_like(k) * std
new_k = torch.randn(config.batch_size, config.kv_sequence_length, config.kv_num_heads, config.head_size, device=device, dtype=torch_type) * std
new_v = torch.randn_like(new_k) * std

head_sink = torch.rand(config.num_heads, dtype=torch_type, device=device) if config.has_head_sink else None

# Reference path calculation
print("Updating reference cache...")
k_cache_ref = k.clone().transpose(1, 2)
v_cache_ref = v.clone().transpose(1, 2)
cache_seqlens = torch.full((config.batch_size,), config.kv_sequence_length, device=device, dtype=torch.int32)
rotary_seqlens = torch.zeros(config.batch_size, device=device, dtype=torch.long)

rotary_dim = math.floor(config.head_size / 16) * 16
angle = torch.rand(config.buffer_sequence_length, rotary_dim // 2, device=device) * 2 * math.pi
cos = torch.cos(angle).to(dtype=torch_type)
sin = torch.sin(angle).to(dtype=torch_type)

# Important: Test uses position_ids for rotation if has_position_ids is True
position_ids = None
if config.has_position_ids:
    position_ids = torch.arange(config.q_sequence_length, device=device).unsqueeze(0).expand(config.batch_size, -1).contiguous()

q_ro = q.clone()
k_ro = new_k.clone()
if config.rotary:
    q_ro = apply_rotary_embedding(q.clone(), cos, sin, rotary_seqlens, config.rotary_interleaved, device)
    k_ro = apply_rotary_embedding(new_k.clone(), cos, sin, rotary_seqlens, config.rotary_interleaved, device)

# Update reference cache
arange = rearrange(torch.arange(config.buffer_sequence_length, device=device), 's -> 1 s')
kv_seqlens_expanded = rearrange(cache_seqlens, 'b -> b 1')
update_mask = arange < kv_seqlens_expanded
source_k = rearrange(k_ro, 'b s ... -> (b s) ...')
k_cache_ref[update_mask] = source_k.to(k_cache_ref.dtype)
source_v = rearrange(new_v, 'b s ... -> (b s) ...')
v_cache_ref[update_mask] = source_v.to(v_cache_ref.dtype)

key_padding_mask = arange < kv_seqlens_expanded

# Window size conversion
window_size = (-1, -1)
if config.local_window_size > 0:
    window_size = (config.local_window_size, 0)
elif True: # causal
    window_size = (-1, 0)

print("Calling attention_ref...")
out_ref, _ = attention_ref(
    q=q_ro, k=k_cache_ref, v=v_cache_ref,
    key_padding_mask=key_padding_mask,
    attention_bias=None, causal=True,
    window_size=window_size, softcap=config.softcap,
    use_smooth_softmax=config.use_smooth_softmax, head_sink=head_sink,
)

# ORT Path
ort_seqlens = cache_seqlens - 1
all_outs = []
all_k_caches = []
print("Running GQA 100 times to check for non-determinism...")
for i in range(100):
    out, present_k, present_v = gqa_prompt_func(
        q=q, k=k, v=v, config=config,
        new_k=new_k, new_v=new_v,
        cos=cos if config.rotary else None,
        sin=sin if config.rotary else None,
        seqlens_k=ort_seqlens,
        position_ids=position_ids,
        attention_bias=None, head_sink=head_sink,
        ep='CPUExecutionProvider', device=device,
        share_buffer=config.share_buffer, ort_type=ort_type,
    )
    all_outs.append(out.cpu().numpy())
    all_k_caches.append(present_k.cpu().numpy())
    if i > 0:
        if not np.array_equal(all_outs[0], all_outs[i]):
            print(f"NON-DETERMINISM DETECTED in output at run {i}!")
            diff = np.abs(all_outs[0] - all_outs[i])
            print(f"Max Diff between run 0 and {i}: {diff.max()}")
            # sys.exit(1) # Don't exit yet, check K cache too
        if not np.array_equal(all_k_caches[0], all_k_caches[i]):
            print(f"NON-DETERMINISM DETECTED in K CACHE at run {i}!")
            k_diff = np.abs(all_k_caches[0] - all_k_caches[i])
            print(f"Max Diff in K cache between run 0 and {i}: {k_diff.max()}")
            sys.exit(1)

print("Determinism check passed over 100 runs.")

out = torch.from_numpy(all_outs[0]).to(device)
out = out.reshape(config.batch_size, config.q_sequence_length, config.num_heads, config.head_size)

# Comparison
out_ref_np = out_ref.cpu().numpy()
out_np = all_outs[0].reshape(config.batch_size, config.q_sequence_length, config.num_heads, config.head_size)

diff = np.abs(out_np - out_ref_np)
max_diff = diff.max()
print(f"Max Diff vs Reference: {max_diff}")

mismatch_cnt = (diff > 0.005).sum()
print(f"Mismatched count > 0.005: {mismatch_cnt} / {out.numel()}")

print("Head-wise max diff:")
for h in range(config.num_heads):
    h_diff = np.abs(out_np[:,:,h,:] - out_ref_np[:,:,h,:]).max()
    print(f"Head {h}: {h_diff}")

# Verify K/V Cache
print("\n=== K/V Cache Verification ===")
k_cache_np = present_k.cpu().numpy()
# Reference k_cache_ref is updated with k_ro
k_cache_ref_np = k_cache_ref.cpu().numpy()
# present_k shape [B, num_kv_heads, max_seq_len, head_size]
# compare only filled part
valid_len = config.kv_sequence_length
k_diff = np.abs(k_cache_np[:, :, :valid_len, :] - k_cache_ref_np[:, :, :valid_len, :])
print(f"K Cache Max Diff: {k_diff.max()}")

# Run again without Head Sink
print("\n=== Running without Head Sink ===")
out_ref_no_sink, _ = attention_ref(
    q=q_ro, k=k_cache_ref, v=v_cache_ref,
    key_padding_mask=key_padding_mask,
    attention_bias=None, causal=True,
    window_size=window_size, softcap=config.softcap,
    use_smooth_softmax=False, head_sink=None,
)

out_no_sink, _, _ = gqa_prompt_func(
    q=q, k=k, v=v, config=config,
    new_k=new_k, new_v=new_v,
    cos=cos if config.rotary else None,
    sin=sin if config.rotary else None,
    seqlens_k=ort_seqlens,
    position_ids=position_ids,
    attention_bias=None, head_sink=None, # DISABLE head sink
    ep='CUDAExecutionProvider', device=device,
    share_buffer=config.share_buffer, ort_type=ort_type,
)
out_no_sink = out_no_sink.reshape(config.batch_size, config.q_sequence_length, config.num_heads, config.head_size)
diff_no_sink = np.abs(out_no_sink.cpu().numpy() - out_ref_no_sink.cpu().numpy()).max()
print(f"Max Diff without Head Sink: {diff_no_sink}")
