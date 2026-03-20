// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Plugin EP control flow kernel implementations for If, Loop, and Scan.
// These delegate to OrtEpApi::CreateIfKernel/CreateLoopKernel/CreateScanKernel
// instead of inheriting from CPU base classes.

#include "core/providers/cuda/plugin/cuda_controlflow_plugin.h"
#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {
namespace plugin {

// ===================================================================
// If kernel
// ===================================================================

Status PluginIfKernel::CreateControlFlowKernelImpl(const OrtKernelInfo* info, OrtKernelImpl** impl) {
  OrtStatus* status = Ort::GetEpApi().CreateIfKernel(info, impl);
  if (status) {
    std::string msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, msg);
  }
  return Status::OK();
}

// ===================================================================
// Loop kernel helper
// ===================================================================

PluginLoopHelper::PluginLoopHelper() : OrtLoopKernelHelper{} {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  ConcatOutput = ConcatOutputImpl;
}

/*static*/
void ORT_API_CALL PluginLoopHelper::ReleaseImpl(_In_ OrtLoopKernelHelper* this_ptr) noexcept {
  delete static_cast<PluginLoopHelper*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL PluginLoopHelper::ConcatOutputImpl(
    _In_ OrtLoopKernelHelper* /*this_ptr*/,
    _In_opt_ void* stream_handle,
    _In_reads_(num_per_iteration_outputs) const OrtValue* const* per_iteration_outputs,
    _In_ size_t num_per_iteration_outputs,
    _Out_writes_bytes_all_(output_size_in_bytes) void* output,
    _In_ size_t output_size_in_bytes) noexcept {
  try {
    if (num_per_iteration_outputs == 0) return nullptr;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream_handle);

    Ort::ConstValue first_output(per_iteration_outputs[0]);
    size_t bytes_per_iteration = first_output.GetTensorSizeInBytes();

    char* cur = static_cast<char*>(output);
    for (size_t i = 0; i < num_per_iteration_outputs; i++) {
      Ort::ConstValue val(per_iteration_outputs[i]);
      size_t cur_bytes = val.GetTensorSizeInBytes();
      if (cur_bytes != bytes_per_iteration) {
        return Ort::Status("Inconsistent size in loop output iteration", ORT_FAIL).release();
      }
      cudaError_t err = cudaMemcpyAsync(cur, val.GetTensorRawData(), bytes_per_iteration,
                                        cudaMemcpyDeviceToDevice, cuda_stream);
      if (err != cudaSuccess) {
        return Ort::Status((std::string("cudaMemcpyAsync failed in Loop ConcatOutput: ") +
                            cudaGetErrorString(err))
                               .c_str(),
                           ORT_FAIL)
            .release();
      }
      cur += bytes_per_iteration;
    }

    if (static_cast<size_t>(cur - static_cast<char*>(output)) != output_size_in_bytes) {
      return Ort::Status("Loop ConcatOutput: output buffer not fully filled", ORT_FAIL).release();
    }

    return nullptr;
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_RUNTIME_EXCEPTION).release();
  }
}

// ===================================================================
// Loop kernel
// ===================================================================

Status PluginLoopKernel::CreateControlFlowKernelImpl(const OrtKernelInfo* info, OrtKernelImpl** impl) {
  auto helper = std::make_unique<PluginLoopHelper>();
  OrtStatus* status = Ort::GetEpApi().CreateLoopKernel(info, helper.get(), impl);
  if (status) {
    std::string msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, msg);
  }
  helper.release();  // ORT takes ownership on success
  return Status::OK();
}

// ===================================================================
// Scan kernel helper
// ===================================================================

PluginScanHelper::PluginScanHelper() : OrtScanKernelHelper{} {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  Transpose = TransposeImpl;
}

/*static*/
void ORT_API_CALL PluginScanHelper::ReleaseImpl(_In_ OrtScanKernelHelper* this_ptr) noexcept {
  delete static_cast<PluginScanHelper*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL PluginScanHelper::TransposeImpl(
    _In_ OrtScanKernelHelper* /*this_ptr*/,
    _In_reads_(num_permutation_elems) const size_t* permutation,
    _In_ size_t num_permutation_elems,
    _In_ const OrtValue* ort_input,
    _In_opt_ OrtSyncStream* stream,
    _Inout_ OrtValue* ort_output) noexcept {
  try {
    // Get the CUDA stream from the OrtSyncStream
    cudaStream_t cuda_stream = nullptr;
    if (stream) {
      const OrtSyncStreamImpl* impl = Ort::GetEpApi().SyncStream_GetImpl(stream);
      if (impl) {
        // GetHandle is a function pointer on OrtSyncStreamImpl
        cuda_stream = static_cast<cudaStream_t>(
            const_cast<OrtSyncStreamImpl*>(impl)->GetHandle(const_cast<OrtSyncStreamImpl*>(impl)));
      }
    }

    Ort::ConstValue input(ort_input);
    Ort::UnownedValue output(ort_output);

    Ort::TensorTypeAndShapeInfo input_info = input.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = input_info.GetShape();
    size_t num_dims = input_shape.size();
    size_t total_elements = input_info.GetElementCount();

    if (num_dims != num_permutation_elems) {
      return Ort::Status("Scan Transpose: permutation size does not match input rank", ORT_FAIL).release();
    }

    if (total_elements == 0) return nullptr;

    // Determine element size from the data type
    ONNXTensorElementDataType elem_type = input_info.GetElementType();
    size_t element_size = 0;
    switch (elem_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        element_size = sizeof(float);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        element_size = sizeof(double);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        element_size = 2;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        element_size = 1;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        element_size = 2;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        element_size = 4;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        element_size = 8;
        break;
      default:
        return Ort::Status("Scan Transpose: unsupported element type", ORT_FAIL).release();
    }

    const void* input_data = input.GetTensorRawData();
    void* output_data = output.GetTensorMutableData<void>();

    // Launch the GPU transpose kernel
    LaunchTransposeKernel(input_data, output_data,
                          input_shape.data(), permutation,
                          num_dims, element_size, total_elements,
                          cuda_stream);

    return nullptr;
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_RUNTIME_EXCEPTION).release();
  }
}

// ===================================================================
// Scan kernel
// ===================================================================

Status PluginScanKernel::CreateControlFlowKernelImpl(const OrtKernelInfo* info, OrtKernelImpl** impl) {
  auto helper = std::make_unique<PluginScanHelper>();
  OrtStatus* status = Ort::GetEpApi().CreateScanKernel(info, helper.get(), impl);
  if (status) {
    std::string msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, msg);
  }
  helper.release();  // ORT takes ownership on success
  return Status::OK();
}

}  // namespace plugin
}  // namespace cuda
}  // namespace onnxruntime

// ===================================================================
// Kernel Registrations — same opset versions as the framework CUDA EP
// ===================================================================

using namespace onnxruntime::cuda::plugin;

namespace onnxruntime {
namespace cuda {

// --- If ---

ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  1, 10,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  PluginIfKernel);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  11, 12,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  PluginIfKernel);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(If,
                                  kOnnxDomain,
                                  13, 18,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  PluginIfKernel);

ONNX_OPERATOR_KERNEL_EX(If,
                        kOnnxDomain,
                        19,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 0)
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypesIRv9()),
                        PluginIfKernel);

// --- Loop ---

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Loop,
                                  kOnnxDomain,
                                  1, 10,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)
                                      .InputMemoryType(OrtMemTypeCPUInput, 1)
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  PluginLoopKernel);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Loop,
                                  kOnnxDomain,
                                  11, 12,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)
                                      .InputMemoryType(OrtMemTypeCPUInput, 1)
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  PluginLoopKernel);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Loop,
                                  kOnnxDomain,
                                  13, 18,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)
                                      .InputMemoryType(OrtMemTypeCPUInput, 1)
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  PluginLoopKernel);

ONNX_OPERATOR_KERNEL_EX(Loop,
                        kOnnxDomain,
                        19,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 0)
                            .InputMemoryType(OrtMemTypeCPUInput, 1)
                            .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypesIRv9()),
                        PluginLoopKernel);

// --- Scan (opset 8) ---

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scan,
                                  kOnnxDomain,
                                  8, 8,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .InputMemoryType(OrtMemTypeCPUInput, 0)
                                      .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                                      .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                                  PluginScanKernel);

// --- Scan (opset 9+) ---

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scan,
                                  kOnnxDomain,
                                  9, 10,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  PluginScanKernel);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scan,
                                  kOnnxDomain,
                                  11, 15,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  PluginScanKernel);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scan,
                                  kOnnxDomain,
                                  16, 18,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
                                  PluginScanKernel);

ONNX_OPERATOR_KERNEL_EX(Scan,
                        kOnnxDomain,
                        19,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypesIRv9()),
                        PluginScanKernel);

}  // namespace cuda
}  // namespace onnxruntime
