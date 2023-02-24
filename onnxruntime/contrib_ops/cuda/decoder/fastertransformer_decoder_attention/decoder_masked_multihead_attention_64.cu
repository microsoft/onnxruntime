#include "decoder_masked_multihead_attention_impl.h"
#include "decoder_masked_multihead_attention_impl_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace decoder_masked_multihead_attention_details;

#define MMHA_LAUNCH_KERNEL(                                                                        \
    T, head_size, THDS_PER_KEY, THDS_PER_VALUE, THDS_PER_BLOCK)                                    \
  size_t dynamic_block_memory = CalcDynamicBlockMemory<T>(params, THDS_PER_VALUE, THDS_PER_BLOCK); \
  dim3 grid(params.num_heads, params.batch_size);                                                  \
  masked_multihead_attention_kernel<T,                                                             \
                                    head_size,                                                     \
                                    THDS_PER_KEY,                                                  \
                                    THDS_PER_VALUE,                                                \
                                    THDS_PER_BLOCK>                                                \
      <<<grid, THDS_PER_BLOCK, dynamic_block_memory, stream>>>(params)

template <typename T, int head_size>
void mmha_launch_kernel(const DecoderMaskedMultiheadAttentionParams& params, cudaStream_t stream) {
  constexpr int THREADS_PER_VALUE = ThreadsPerValue<T, head_size>::value;
  int total_sequence_length = params.total_sequence_length;

  if (total_sequence_length < 32) {
    MMHA_LAUNCH_KERNEL(T, head_size, 4, THREADS_PER_VALUE, 64);
  } else if (total_sequence_length < 2048) {
    MMHA_LAUNCH_KERNEL(T, head_size, 2, THREADS_PER_VALUE, 128);
  } else {
    MMHA_LAUNCH_KERNEL(T, head_size, 1, THREADS_PER_VALUE, 256);
  }
}

template void mmha_launch_kernel<float, 64>(const DecoderMaskedMultiheadAttentionParams& params, cudaStream_t stream);

template void mmha_launch_kernel<uint16_t, 64>(const DecoderMaskedMultiheadAttentionParams& params, cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime