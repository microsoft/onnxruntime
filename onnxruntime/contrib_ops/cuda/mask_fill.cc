#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/mask_fill.h"

using namespace onnxruntime;
using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    MaskFill,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(0, 0)
        .TypeConstraint("T", BuildKernelDefConstraints<BFloat16, float, double, MLFloat16>()),
    MaskFill);

template <typename T>
void MaskFillCudaImpl(
    cudaStream_t stream,
    Tensor* output_tensor,
    const Tensor* mask_tensor,
    int axis);

template <typename T>
void MaskFillCuda<T>::operator()(
    cudaStream_t stream,
    Tensor* output_tensor,
    const Tensor* mask_tensor,
    int axis) {
  MaskFillCudaImpl<T>(
      stream,
      output_tensor,
      mask_tensor,
      axis);
}

Status MaskFill::ComputeInternal(OpKernelContext* ctx) const {
    const auto* input_tensor = ctx->Input<Tensor>(0);
    const auto& input_shape = input_tensor->Shape();

    const auto* mask_tensor = ctx->Input<Tensor>(1);
    auto* output_tensor = ctx->Output(0, input_shape);

    const void* input_data = input_tensor->DataRaw();
    void* output_data = output_tensor->MutableDataRaw();

    if (input_data != output_data)
    {
        CUDA_CALL(cudaMemcpyAsync(output_data, input_data, input_tensor->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream()));
    }

    const auto& mask_shape = mask_tensor->Shape();
    const auto mask_dim = mask_shape.NumDimensions();

    ORT_RETURN_IF_NOT(mask_dim == 1, "Mask tensor dim should be 1.");
    const int axis = static_cast<int>(HandleNegativeAxis(axis_, input_shape.NumDimensions()));
    ORT_RETURN_IF_NOT(mask_shape.GetDims()[0] == input_shape.GetDims()[axis], "mask tensor mismatch.");

    utils::MLTypeCallDispatcher<float, double, MLFloat16> t_disp(input_tensor->GetElementType());

    t_disp.Invoke<MaskFillCuda>(Stream(), output_tensor, mask_tensor, axis);

    return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
