// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_DML

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "core/providers/dml/dml_context.h"
#include "onnxruntime_lite_custom_op.h"

using namespace Ort::Custom;

#define CUSTOM_ENFORCE(cond, msg)  \
  if (!(cond)) {                   \
    throw std::runtime_error(msg); \
  }

namespace Dml {

void KernelOne(const Ort::Custom::DmlContext& dml_ctx,
               const Ort::Custom::Tensor<float>& X,
               const Ort::Custom::Tensor<float>& /*Y*/,
               Ort::Custom::Tensor<float>& Z) {
  auto input_shape = X.Shape();
  CUSTOM_ENFORCE(dml_ctx.dml_device, "failed to get dml device");
  CUSTOM_ENFORCE(dml_ctx.d3d12_device, "failed to get dml device");
  CUSTOM_ENFORCE(dml_ctx.cmd_list, "failed to get cmd list");
  CUSTOM_ENFORCE(dml_ctx.cmd_recorder, "failed to get cmd recorder");
  auto z_raw = Z.Allocate(input_shape);
  CUSTOM_ENFORCE(z_raw, "failed to allocate output");
  // todo - implement element-wise add for DML
}

const std::unique_ptr<OrtLiteCustomOp> c_CustomOpOne{Ort::Custom::CreateLiteCustomOp("CustomOpOne", "DMLExecutionProvider", KernelOne)};

void RegisterOps(Ort::CustomOpDomain& domain) {
  domain.Add(c_CustomOpOne.get());
}

}  // namespace Dml

#else

namespace Dml {

void RegisterOps(Ort::CustomOpDomain& domain) {}

}  // namespace Dml

#endif