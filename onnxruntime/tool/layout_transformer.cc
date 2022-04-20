// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/session/onnxruntime_c_api.h>
#include <core/graph/model.h>
#include <core/common/common.h>
#include <core/graph/model.h>
#include <core/platform/logging/make_platform_default_log_sink.h>
#include <onnx/defs/operator_sets.h>
#include <core/graph/contrib_ops/ms_opset.h>
#include <core/graph/contrib_ops/onnx_deprecated_opset.h>
#include <onnx/shape_inference/implementation.h>
#include <core/xnnpack/schema/xnnpack_opset.h>
#include <core/framework/op_node_proto_helper.h>
#include <core/providers/common.h>
#include <core/optimizer/selectors_actions/helpers.h>
#include <core/framework/tensorprotoutils.h>
#include <core/optimizer/utils.h>
#include <core/xnnpack/optimizer/xnnpack_transformer.h>

using namespace onnxruntime;

#define ORT_RETURN_NEG_ONE_IF_ERROR(expr)                               \
  do {                                                                  \
    auto _status = (expr);                                              \
    if ((!_status.IsOK())) {                                            \
      LOGS_USER(*logger, ERROR) << _status.ErrorMessage() << std::endl; \
      return -1;                                                        \
    }                                                                   \
  } while (0)

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  if (argc < 3) return -1;
  setlocale(LC_ALL, "");
  auto& domainToVersionRangeInstance = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  if (domainToVersionRangeInstance.Map().find(onnxruntime::kMSDomain) == domainToVersionRangeInstance.Map().end()) {
    domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSDomain, 1, 1);
  }
  domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSNchwcDomain, 1, 1);

  ::ONNX_NAMESPACE::RegisterOnnxOperatorSetSchema();
  ::ONNX_NAMESPACE::RegisterOpSetSchema<contrib::OpSet_Microsoft_ver1>();
  ::ONNX_NAMESPACE::RegisterOpSetSchema<contrib::OpSet_ONNX_Deprecated>();
  for (auto& schema : xnnpack::GetSchemas()) {
    ::ONNX_NAMESPACE::RegisterSchema(schema);
  }
  std::string default_logger_id = "XNNPack";
  auto lmgr = std::make_unique<logging::LoggingManager>(logging::MakePlatformDefaultLogSink(),
                                                        logging::Severity::kINFO,
                                                        false,
                                                        logging::LoggingManager::InstanceType::Default,
                                                        &default_logger_id);

  const ORTCHAR_T* input_model_path = argv[1];
  const ORTCHAR_T* output_model_path = argv[2];
  auto logger = lmgr->CreateLogger("xnnpack_converter");
  std::shared_ptr<onnxruntime::Model> m;
  ORT_RETURN_NEG_ONE_IF_ERROR(Model::Load(input_model_path, m, nullptr, *logger));
  std::shared_ptr<CPUAllocator> cpu_allocator = std::make_shared<CPUAllocator>();
  bool modified = false;
  XNNPackTransformer xnnpack_trans(cpu_allocator);
  ORT_RETURN_NEG_ONE_IF_ERROR(xnnpack_trans.Apply(m->MainGraph(), modified, *logger));
  if (!modified) {
    LOGS_USER(*logger, ERROR) << "The graph is not changed.";
    return 0;
  }
  auto model_proto = m->ToProto();
  try {
    ::ONNX_NAMESPACE::ShapeInferenceOptions options{true, 1, true};
    ::ONNX_NAMESPACE::shape_inference::InferShapes(model_proto,
                                                   ::ONNX_NAMESPACE::OpSchemaRegistry::Instance(),
                                                   options);
  } catch (const std::exception& ex) {
    LOGS_USER(*logger, ERROR) << ex.what();
    return -1;
  }
  int fd = -1;
  ORT_RETURN_NEG_ONE_IF_ERROR(Env::Default().FileOpenWr(output_model_path, fd));
  if (!model_proto.SerializeToFileDescriptor(fd)) {
    return -1;
  }
  return 0;
}
