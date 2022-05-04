// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)
#include <thread>

#include "gtest/gtest.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"
// #include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "core/common/path_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/models/runner/training_runner.h"
// #include "orttraining/test/session/training_session_test_utils.h"
#include "orttraining/training_api/interfaces.h"
#include "core/framework/tensorprotoutils.h"

// includes to load parameters.of
#include "orttraining/onnxflow/csrc/load_parameters.h"
#include <filesystem>
#include <iostream>
#include <onnx/onnx_pb.h>

using namespace onnxruntime::logging;
using namespace onnxruntime::training;
using namespace onnxruntime::training::api_test;
using namespace google::protobuf::util;
using namespace onnxruntime::path_utils;

#ifdef USE_CUDA
namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options);
std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name);

}  // namespace onnxruntime
#endif

namespace onnxruntime {
namespace training {

namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/training_api/")

template <typename T>
static void CreateInputOrtValue(std::vector<int64_t> dims,
                                const std::vector<T>& value,
                                OrtValue* p_ortvalue,
                                AllocatorPtr alloc = nullptr) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.GetAllocator(0, OrtMemTypeDefault);

  TensorShape shape(dims);
  assert(shape.Size() == static_cast<int64_t>(value.size()));
  auto element_type = DataTypeImpl::GetType<T>();
  auto allocator = alloc ? alloc : cpu_allocator;
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);

  if (value.size() > 0) {
    memcpy(p_tensor->MutableDataRaw(), value.data(), p_tensor->SizeInBytes());
  }

  p_ortvalue->Init(p_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

template <typename T>
static void OrtValueToVec(OrtValue& val, std::vector<T>& output) {
  const Tensor& tensor = val.Get<Tensor>();
  int64_t num_ele = tensor.Shape().Size();
  const float* val_ptr = tensor.template Data<float>();
  output.assign(val_ptr, val_ptr + num_ele);
}
// static Status CreateOrtValuesFromTensorProtos(
//     const std::vector<const ONNX_NAMESPACE::TensorProto*>& tensor_protos,
//     NameMLValMap& name_to_ort_value) {
//   static CPUExecutionProviderInfo info;
//   static CPUExecutionProvider cpu_provider(info);
//   static AllocatorPtr cpu_allocator = cpu_provider.GetAllocator(0, OrtMemTypeDefault);

//   for (const auto tensor_proto : tensor_protos) {
//     TensorShape tensor_shape{utils::GetTensorShapeFromTensorProto(*tensor_proto)};
//     const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto->data_type())->GetElementType();
//     auto p_tensor = std::make_unique<Tensor>(tensor_dtype, tensor_shape, cpu_allocator);
//     ORT_RETURN_IF_ERROR(utils::TensorProtoToTensor(Env::Default(), nullptr, *tensor_proto, *p_tensor));

//     OrtValue ort_value;
//     ort_value.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
//     name_to_ort_value.emplace(tensor_proto->name(), ort_value);
//   }

//   return Status::OK();
// }

TEST(TrainingApiTest, ModuleTrainStep) {
  auto model_uri = MODEL_FOLDER "gradient_graph.onnx";

  std::map<std::string, std::shared_ptr<Parameter>> named_parameters;

  std::map<std::string, std::vector<int64_t>> param_names_shapes{{"_original_module.fc1.weight", {500, 784}},
                                                                 {"_original_module.fc1.bias", {500}},
                                                                 {"_original_module.fc2.weight", {10, 500}},
                                                                 {"_original_module.fc2.bias", {10}}};
  OrtValue param;
  for (auto& it : param_names_shapes) {
    int64_t num_ele = std::accumulate(it.second.begin(), it.second.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    std::vector<float_t> data(num_ele, 1);
    CreateInputOrtValue(it.second, data, &param);
    named_parameters.insert({it.first, std::make_shared<Parameter>(it.first, std::move(param))});
  }

  /*
  // This path works, but the parameters.of file doesnt have any data right now
   auto path_to_parameters_proto = MODEL_FOLDER "parameters.of";
   auto parameters = onnxflow::load_parameters(path_to_parameters_proto);

   std::vector<std::string> param_names;
   std::vector<const ONNX_NAMESPACE::TensorProto*> param_tensor_proto_ptrs{};
   for (const auto& param : parameters.parameters()) {
     // if (param.is_parameter()) {
       onnx::TensorProto tensor_proto;
       param.data().UnpackTo(&tensor_proto);
       param_tensor_proto_ptrs.emplace_back(&tensor_proto);
       std::cout << "param<" << tensor_proto.name() << ", requires_grad=" << (param.requires_grad() ? "True" : "False") << ">" << std::endl;
     // } else {
     //   onnx::ValueInfoProto valueinfo;
     //   param.data().UnpackTo(&valueinfo);
     //   std::cout << "valinfo<" << valueinfo.name() << ", requires_grad=" << (param.requires_grad() ? "True" : "False") << ">" << std::endl;
     // }
   }
   std::unordered_map<std::string, OrtValue> name_to_ort_values;
   ORT_ENFORCE(CreateOrtValuesFromTensorProtos(param_tensor_proto_ptrs, name_to_ort_values).IsOK());
   for (auto it = name_to_ort_values.begin(); it != name_to_ort_values.end(); ++it) {
     named_parameters.insert({it->first, std::make_shared<Parameter>(it->first, it->second)});
   }
   */

  auto module_sess = std::make_unique<Module>(model_uri, named_parameters);

  // #ifdef USE_CUDA
  //   OrtCUDAProviderOptions provider_options{};
  //     auto factory = CreateExecutionProviderFactory_Cuda(&provider_options);
  //     auto provider = std::move(factory->CreateProvider());

  //     auto input_allocator = CreateCUDAPinnedAllocator(provider_options.device_id, CUDA_PINNED);
  // #endif

  OrtValue input, target;
  // hard coded each sample to have 4 elements so far.
  // todo: we can make it support more generic once we are clear what our offline process graph needed.
  CreateInputOrtValue({2, 784}, std::vector<float_t>(1568, 1), &input);
  CreateInputOrtValue({
                          2,
                      },
                      std::vector<int32_t>(2, 1), &target);
  auto data_loader = std::vector<std::vector<OrtValue>>(4, std::vector<OrtValue>{input, target});

  size_t step = 0;
  std::vector<float> before_train_vec, single_bias_grad_vec, current_bias_grad_vec, single_grad_vec, accumulated_grad_vec;
  std::string param_name = "_original_module.fc2.weight";
  std::shared_ptr<Parameter> bias_param = module_sess->named_parameters()[param_name];
  OrtValue& bias_grad = bias_param->gradient();
  OrtValueToVec(bias_grad, before_train_vec);

  for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
    step += 1;
    std::vector<OrtValue>& inputs = *it;
    std::vector<OrtValue> fetches;
    ORT_ENFORCE(module_sess->TrainStep(inputs, fetches).IsOK());

    OrtValueToVec(fetches[1], single_grad_vec);
    OrtValueToVec(fetches[5], accumulated_grad_vec);

    bias_grad = bias_param->gradient();

    if (step > 1) {
      OrtValueToVec(bias_grad, current_bias_grad_vec);
      for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
        ORT_ENFORCE(current_bias_grad_vec[i] == single_bias_grad_vec[i] * step);
      }
    } else {
      OrtValueToVec(bias_grad, single_bias_grad_vec);
    }
  }
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
#endif