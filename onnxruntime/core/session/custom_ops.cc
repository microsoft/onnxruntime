// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include "core/session/inference_session.h"
#include "core/framework/customregistry.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/tensor_type_and_shape.h"

ONNXTensorElementDataType MLDataTypeToOnnxRuntimeTensorElementDataType(const onnxruntime::DataTypeImpl* cpp_type);

ORT_API_STATUS_IMPL(OrtKernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out) {
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<float>(name, out);
  if (status.IsOK())
    return nullptr;
  return onnxruntime::ToOrtStatus(status);
}

ORT_API_STATUS_IMPL(OrtKernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out) {
  auto status = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info)->GetAttr<int64_t>(name, out);
  if (status.IsOK())
    return nullptr;
  return onnxruntime::ToOrtStatus(status);
}

ORT_API(const OrtValue*, OrtKernelContext_GetInput, const OrtKernelContext* context, _In_ size_t index) {
  return reinterpret_cast<const OrtValue*>(reinterpret_cast<const onnxruntime::OpKernelContextInternal*>(context)->GetInputMLValue(index));
};

ORT_API(OrtValue*, OrtKernelContext_GetOutput, OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count) {
  onnxruntime::TensorShape shape(dim_values, dim_count);
  return reinterpret_cast<OrtValue*>(reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context)->OutputMLValue(index, shape));
};

constexpr OrtCustomOpApi g_custom_op_api = {
    &OrtKernelInfoGetAttribute_float,
    &OrtKernelInfoGetAttribute_int64,

    &OrtGetTensorShapeAndType,

    &OrtGetTensorShapeElementCount,
    &OrtGetNumOfDimensions,
    &OrtGetDimensions,
    &OrtSetDims,

    &OrtGetTensorMutableData,

    &OrtReleaseTensorTypeAndShapeInfo,

    &OrtKernelContext_GetInput,
    &OrtKernelContext_GetOutput,
};

namespace onnxruntime {
struct CustomOpKernel : OpKernel {
  CustomOpKernel(const OpKernelInfo& info, OrtCustomOp& op) : OpKernel(info), op_(op) {
    if (op_.version != 1)
      throw std::invalid_argument("Unsupported version '" + std::to_string(op_.version) + "' in custom op '" + op.GetName(&op));
    op_kernel_ = op_.CreateKernel(&op_, &g_custom_op_api, reinterpret_cast<OrtKernelInfo*>(const_cast<OpKernelInfo*>(&info)));
  }

  ~CustomOpKernel() {
    op_.KernelDestroy(op_kernel_);
  }

  Status Compute(OpKernelContext* ctx) const override {
    auto* ictx = static_cast<OpKernelContextInternal*>(ctx);
    op_.KernelCompute(op_kernel_, reinterpret_cast<OrtKernelContext*>(ictx));
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpKernel);

  OrtCustomOp& op_;
  void* op_kernel_;
};

common::Status CreateCustomRegistry(const std::vector<OrtCustomOpDomain*>& op_domains, std::shared_ptr<CustomRegistry>& output) {
  output = std::make_shared<CustomRegistry>();

  for (auto& domain : op_domains) {
    if (domain->domain_[0])
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(domain->domain_, 1, 1000);

    std::vector<ONNX_NAMESPACE::OpSchema> schemas_list;

    for (auto& op : domain->custom_ops_) {
      ONNX_NAMESPACE::OpSchema schema(op->GetName(op), "unknown", 0);

      auto input_count = op->GetInputTypeCount(op);
      for (size_t i = 0; i < input_count; i++) {
        auto type = op->GetInputType(op, i);

        schema.Input(i, "A", "Description",
                     DataTypeImpl::ToString(onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(type)));
      }

      auto output_count = op->GetOutputTypeCount(op);
      for (size_t i = 0; i < output_count; i++) {
        auto type = op->GetOutputType(op, i);

        schema.Output(i, "A", "Description",
                      DataTypeImpl::ToString(onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(type)));
      }

      schema.SetDomain(domain->domain_);
      schema.SinceVersion(1);
      schema.AllowUncheckedAttributes();
      schemas_list.push_back(schema);

      KernelDefBuilder def_builder;
      def_builder.SetName(op->GetName(op))
          .SetDomain(domain->domain_)
          .SinceVersion(1)
          .Provider(onnxruntime::kCpuExecutionProvider);
      KernelCreateFn kernel_create_fn = [&op](const OpKernelInfo& info) -> OpKernel* { return new CustomOpKernel(info, *op); };
      KernelCreateInfo create_info(def_builder.Build(), kernel_create_fn);

      output->RegisterCustomKernel(create_info);
    }

    ORT_RETURN_IF_ERROR(output->RegisterOpSet(schemas_list,
                                              domain->domain_,
                                              1 /* baseline opset version */,
                                              1000 /* opset version */));
  }
  return Status::OK();
}

}  // namespace onnxruntime
