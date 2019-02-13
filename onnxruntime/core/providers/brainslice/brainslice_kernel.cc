#include "core/providers/brainslice/brainslice_kernel.h"
#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/providers/brainslice/brainslice_mem_planner.h"
#include "core/providers/brainslice/fpga_handle.h"

namespace onnxruntime {
namespace brainslice {
BrainSliceOpKernel::BrainSliceOpKernel(const OpKernelInfo& info) : OpKernel(info),
                                                                   provider_(const_cast<BrainSliceExecutionProvider*>(dynamic_cast<const BrainSliceExecutionProvider*>(info.GetExecutionProvider()))),
                                                                   native_dim_(provider_->GetFPGAHandle().GetParameters().NATIVE_DIM) {
}

template <>
Status BrainSliceOpKernel::UploadBrainSliceParameter<float>(BrainSliceParameterInitPlan& plan, BrainSliceExecutionProvider* provider, uint32_t* num_tiles) {
  if (!plan.tensor)
    return Status(common::ONNXRUNTIME, common::FAIL, "can't have empty tensor in BrainSlice parameter initialization");
  auto& shape = plan.tensor->Shape();
  const float* data = plan.tensor->Data<float>();
  size_t native_dim = provider->GetBrainSliceNativeDim();
  if (plan.need_transpose)
    ORT_NOT_IMPLEMENTED("Transponse BrainSlice matrix before uploading is not supported yet");
  std::vector<BS_Half> bs_data;
  for (int64_t i = 0; i < shape.Size(); i++) {
    bs_data.push_back(BS_Half(data[i]));
  }

  auto* mem_planner = provider->GetBrainSliceMemoryPlanner(plan.mem_type, plan.usage);
  if (!mem_planner)
    return Status(common::ONNXRUNTIME, common::FAIL, "Failed to find memory planner for memory type : " + std::to_string(plan.mem_type));
  size_t tiles;
  if (plan.usage == ParameterUsage::USE_AS_VECTOR) {
    auto pad_size = (shape.Size() + native_dim - 1) / native_dim;
    auto address = mem_planner->Alloc(pad_size);
    if (address < 0)
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to allocate resource in BrainSlice, memory type: " + std::to_string(plan.mem_type) + ", size: " + std::to_string(pad_size));
    ORT_RETURN_IF_ERROR(provider->GetFPGAHandle().LoadVector(bs_data, address, plan.mem_type, &tiles));
    plan.address = address;
  } else {
    auto rows = shape.SizeToDimension(plan.axis);
    auto cols = shape.SizeFromDimension(plan.axis);
    auto pad_rows = (rows + native_dim - 1) / native_dim;
    auto pad_cols = (cols + native_dim - 1) / native_dim;
    auto address = mem_planner->Alloc(pad_rows * pad_cols);
    if (address < 0)
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to allocate resource in BrainSlice, memory type: " + std::to_string(plan.mem_type) + ", size: " + std::to_string(pad_rows * pad_cols));
    ORT_RETURN_IF_ERROR(provider->GetFPGAHandle().LoadMatrix(bs_data, static_cast<int>(rows), static_cast<int>(cols), address, true, plan.mem_type, &tiles));
    plan.address = address;
  }
  *num_tiles = static_cast<uint32_t>(tiles);
  return Status::OK();
}

}  // namespace brainslice
}  // namespace onnxruntime
