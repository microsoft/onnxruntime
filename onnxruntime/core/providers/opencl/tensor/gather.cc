// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opencl/tensor/gather.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/cpu/tensor/gatherbase.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME gather_kernel_src
#include "opencl_generated/tensor/kernels/gather.cl.inc"
}  // namespace

class Gather : public OpenCLKernel, public GatherBase {
 public:
  explicit Gather(const OpKernelInfo& info)
      : OpenCLKernel(info), GatherBase(info) {
    LoadProgram(gather_kernel_src, gather_kernel_src_len);
    LoadKernel("Gather");
  };

  Status Compute(OpKernelContext* context) const override;
};

// template <typename Tin>
// Status Gather(const Tensor* indices_tensor, const uint8_t* src_base, uint8_t* dst_base, bool is_string_type,
//                       const size_t element_bytes, const int64_t block_size, const int64_t M,
//                       const int64_t N, const int64_t data_batch_bytes, const int64_t gathered_batch_bytes,
//                       const TensorShape& input_data_shape, const int64_t axis,OpKernelContext* context) {
//   const Tin* indices_data = indices_tensor->Data<Tin>();

//   // Check the indices first in case there's a out of bound index.
//   auto axis_dim_limit = input_data_shape[narrow<size_t>(axis)];

//   for (int64_t i = 0; i < N; ++i) {
//     Tin idx = indices_data[i];
//     if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
//       return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
//                              "indices element out of data bounds, idx=", idx,
//                              " must be within the inclusive range [", -axis_dim_limit, ",", axis_dim_limit - 1, "]");
//     }
//   }

//   return Status::OK();
// }
Status Gather::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));

  const TensorShape& input_data_shape = p.input_tensor->Shape();

  bool is_string_type = p.input_tensor->IsDataTypeString();

  const size_t element_bytes = p.input_tensor->DataType()->Size();
  const int64_t block = input_data_shape.SizeFromDimension(SafeInt<size_t>(p.axis) + 1);
  const int64_t block_size = SafeInt<int64_t>(element_bytes) * block;
  const int64_t M = input_data_shape.SizeToDimension(narrow<size_t>(p.axis));
  const int64_t N = p.indices_tensor->Shape().Size();
  const int64_t data_batch_bytes = input_data_shape.SizeFromDimension(narrow<size_t>(p.axis)) * element_bytes;
  const int64_t gathered_batch_bytes = N * block * SafeInt<int64_t>(element_bytes);

  const auto* src_base = static_cast<const uint8_t*>(p.input_tensor->DataRaw());
  auto* dst_base = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw());
  auto Input = p.input_tensor;
  auto Output = p.output_tensor;
  if (/*utils::HasType<EnabledIndexTypes, int32_t>() &&*/
      p.indices_tensor->IsDataType<int32_t>()) {
    // return Gather<int32_t>(p.indices_tensor, src_base, dst_base, is_string_type, element_bytes,
    //                                block_size, M, N, data_batch_bytes, gathered_batch_bytes, input_data_shape, p.axis, context);
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "int32_t indices tensor not supported yet in Gather.");
  }
  if (p.indices_tensor->IsDataType<int64_t>()) {
    size_t input_shape_size = input_data_shape.NumDimensions() * sizeof(int64_t);

    cl_mem Input_shape = exec_->GetScratchBufferTmp(input_shape_size);
    exec_->WriteToCLBuffer(Input_shape, input_data_shape.GetDims().data(), input_shape_size);

    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("Gather")}
            .SetBuffers(*(p.indices_tensor), *Input, *Output)
            .SetArg<cl_int>((cl_int)(is_string_type ? 1 : 0))
            .SetArg<cl_long>(element_bytes)
            .SetArg<cl_long>(block_size)
            .SetArg<cl_long>(M)
            .SetArg<cl_long>(N)
            .SetArg<cl_long>(data_batch_bytes)
            .SetArg<cl_long>(gathered_batch_bytes)
            .SetBuffer(Input_shape)
            .SetArg<cl_long>(p.axis)
            .Launch(*exec_, {M * N, 1, 1}));

    exec_->ReleaseCLBuffer(Input_shape);
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tind not supported yet in Gather.");
}

ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kOnnxDomain,
    13,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

}  // namespace opencl
}  // namespace onnxruntime
