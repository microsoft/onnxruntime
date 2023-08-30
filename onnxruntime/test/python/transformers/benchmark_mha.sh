echo "flash attention v2"
ORT_DISABLE_FLASH_ATTENTION=0  ORT_MIN_SEQ_LEN_FLASH_ATTENTION_PACKED_QKV=0 python benchmark_mha.py | tee result.txt

echo "==="
echo "TensorRT attention kernels - cross attention (when kv_seq_len <= 128) or fused attention (when seq_len <= 384) or flash attention (seq_len > 384)"
ORT_DISABLE_FLASH_ATTENTION=1  python benchmark_mha.py | tee -a result.txt

echo "==="
echo "Memory Efficient attention"
ORT_DISABLE_FLASH_ATTENTION=1 ORT_DISABLE_TRT_FLASH_ATTENTION=1 ORT_DISABLE_FUSED_ATTENTION=1 ORT_DISABLE_FUSED_CROSS_ATTENTION=1 python benchmark_mha.py | tee -a result.txt

echo "==="
echo "Unfused Attention (some configurations might fail)"
ORT_DISABLE_FLASH_ATTENTION=1 ORT_DISABLE_TRT_FLASH_ATTENTION=1 ORT_DISABLE_FUSED_ATTENTION=1 ORT_DISABLE_FUSED_CROSS_ATTENTION=1 ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION=1 python benchmark_mha.py | tee -a result.txt
