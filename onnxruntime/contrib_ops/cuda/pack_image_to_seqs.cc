#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/pack_image_to_seqs.h"

using namespace onnxruntime;
using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    PackImageToSeqs,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>()),
    PackImageToSeqs);

template <typename T>
void PackImageToSeqsCudaImpl(
    cudaStream_t stream,
    const Tensor* input_tensor,
    Tensor* output_tensor,
    int* seq_offset_gpu,
    int* seq_output_indexs_gpu);

template <typename T>
void PackImageToSeqsCuda<T>::operator()(
    cudaStream_t stream,
    const Tensor* input_tensor,
    Tensor* output_tensor,
    int* seq_offset_gpu,
    int* seq_output_indexs_gpu)
{
  PackImageToSeqsCudaImpl<T>(
      stream,
      input_tensor,
      output_tensor,
      seq_offset_gpu,
      seq_output_indexs_gpu);
}

Status PackImageToSeqs::ComputeInternal(OpKernelContext* ctx) const {
    const auto* input_tensor = ctx->Input<Tensor>(0);
    const auto& input_shape = input_tensor->Shape();
    const auto& input_dim = input_shape.NumDimensions();

    const auto* seq_len_tensor = ctx->Input<Tensor>(1);
    const auto& seq_len_shape = seq_len_tensor->Shape();
    const auto seq_len_dim = seq_len_shape.NumDimensions();

    ORT_RETURN_IF_NOT(input_dim == 4, "Input tensor dim should be 4 1CHW.");
    ORT_RETURN_IF_NOT(seq_len_dim == 1, "Mask tensor dim should be 1.");
    ORT_RETURN_IF_NOT(1 == input_shape.GetDims()[0], "wrong input tensor.");

    auto seq_len_nums = seq_len_shape.GetDims()[0];
    const int* data = seq_len_tensor->template Data<int>();
    std::vector<int> seq_len_vector_host(seq_len_nums);
    seq_len_vector_host.assign(data, data + seq_len_nums);
    int max_seq_width = *std::max_element(seq_len_vector_host.begin(), seq_len_vector_host.end());

    std::vector<int> seq_offset_vector_host(seq_len_nums);
    std::vector<int> seq_output_batch_index_host(input_shape.GetDims()[3]);
    memset(seq_output_batch_index_host.data(), 0, sizeof(int) * seq_output_batch_index_host.size());
    int offset = 0;
    auto index_iterator = seq_output_batch_index_host.begin();
    int batch_id = 1;
    for(auto seq_len:seq_len_vector_host)
    {
        std::transform(index_iterator + offset, index_iterator + offset + seq_len, index_iterator + offset, [batch_id](int i) -> int { return batch_id; });
        batch_id ++;
        seq_offset_vector_host.push_back(offset);
        offset += seq_len + margin_;
    }

    CudaAsyncBuffer<int> seq_offset_gpu(this, seq_offset_vector_host);
    ORT_RETURN_IF_ERROR(seq_offset_gpu.CopyToGpu());

    CudaAsyncBuffer<int> seq_output_indexs_gpu(this, seq_output_batch_index_host);
    ORT_RETURN_IF_ERROR(seq_output_indexs_gpu.CopyToGpu());

    // N * C * H * max_width
    std::vector<int64_t> output_dimensions{seq_len_nums,
                                            input_shape.GetDims()[1],
                                            input_shape.GetDims()[2],
                                            max_seq_width};
    Tensor* output_tensor = ctx->Output(0, TensorShape{output_dimensions});
    return Status::OK();
    utils::MLTypeCallDispatcher<MLFloat16, float, double, BFloat16> t_disp(input_tensor->GetElementType());

    t_disp.Invoke<PackImageToSeqsCuda>(Stream(), input_tensor, output_tensor, seq_offset_gpu.GpuPtr(), seq_output_indexs_gpu.GpuPtr());

    return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
