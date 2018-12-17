// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// example custom op

#include "core/framework/custom_ops_author.h"
#include "core/session/onnxruntime_c_api.h"

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;

class FooKernel : public OpKernel {
 public:
  FooKernel(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    const Tensor* X = ctx->Input<Tensor>(0);
    const Tensor* W = ctx->Input<Tensor>(1);
    auto* X_data = X->template Data<float>();
    auto* W_data = W->template Data<float>();
    Tensor* Y = ctx->Output(0, X->Shape());
    auto* Y_data = Y->template MutableData<float>();

    for (int64_t i = 0; i < X->Shape().Size(); i++) {
      Y_data[i] = X_data[i] + W_data[i];
    }

    return Status::OK();
  }
};

ORT_EXPORT KernelsContainer* GetAllKernels() {
  KernelsContainer* kc = new KernelsContainer;

  KernelDefBuilder def_builder;
  def_builder.SetName("Foo")
      .SetDomain(onnxruntime::kOnnxDomain)
      .SinceVersion(7)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  KernelCreateFn kernel_create_fn = [](const OpKernelInfo& info) -> OpKernel* { return new FooKernel(info); };
  KernelCreateInfo create_info(def_builder.Build(), kernel_create_fn);
  kc->kernels_list.push_back(std::move(create_info));
  return kc;
}

ORT_EXPORT SchemasContainer* GetAllSchemas() {
  SchemasContainer* sc = new SchemasContainer;
  sc->domain = onnxruntime::kOnnxDomain;
  sc->baseline_opset_version = 5;
  sc->opset_version = 7;

  ONNX_NAMESPACE::OpSchema schema("Foo", "unknown", 0);
  schema.Input(0,
               "A",
               "First operand, should share the type with the second operand.",
               "T");
  schema.Input(
      1,
      "B",
      "Second operand. With broadcasting can be of smaller size than A. "
      "If broadcasting is disabled it should be of the same size.",
      "T");
  schema.Output(0, "C", "Result, has same dimensions and type as A", "T");
  schema.TypeConstraint(
      "T",
      OpSchema::numeric_types_for_math_reduction(),
      "Constrain input and output types to high-precision numeric tensors.");
  schema.SinceVersion(7);

  sc->schemas_list.push_back(schema);
  return sc;
}

ORT_EXPORT void FreeKernelsContainer(KernelsContainer* kc) {
  delete kc;
}

ORT_EXPORT void FreeSchemasContainer(SchemasContainer* sc) {
  delete sc;
}
