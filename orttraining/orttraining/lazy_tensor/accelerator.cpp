#include "accelerator.h"
#include <string>
#include <stack>
#include <iostream>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/onnx.h>
#include <ATen/core/functional.h>
#include "core/framework/session_options.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ort_value.h"

namespace py = pybind11;
namespace aten = torch::jit::aten;
namespace prim = torch::jit::prim;

// static variable used to create inference session and training session.
static std::unique_ptr<onnxruntime::Environment> ltc_env;

void InitializeLtcEnv() {
  auto initialize = [&]() {
    std::cout << "InitializeLtcEnv" << std::endl;
    // Initialization of the module
    static std::string name = std::string("LTC");
    ORT_THROW_IF_ERROR(onnxruntime::Environment::Create(std::make_unique<onnxruntime::logging::LoggingManager>(
                                                  std::make_unique<onnxruntime::logging::CLogSink>(),
                                                  onnxruntime::logging::Severity::kWARNING, false, onnxruntime::logging::LoggingManager::InstanceType::Temporal,
                                                  &name),
                                              ltc_env));
    static bool initialized = false;
    if (initialized) {
      return;
    }
    initialized = true;
  };
  initialize();
}

onnxruntime::Environment& GetLtcEnv() {
  if (!ltc_env) {
    InitializeLtcEnv();
  }
  return *ltc_env;
}

onnxruntime::MLDataType to_ort_scalar_type(
  at::ScalarType dtype) {
  switch (dtype){
    case at::kFloat:
      return onnxruntime::DataTypeImpl::GetType<float>();
    case at::kDouble:
      return onnxruntime::DataTypeImpl::GetType<double>();
    case at::kHalf:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>();
    case at::kBFloat16:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>();
    case at::kInt:
      return onnxruntime::DataTypeImpl::GetType<int>();
    case at::kShort:
      return onnxruntime::DataTypeImpl::GetType<int16_t>();
    case at::kLong:
      return onnxruntime::DataTypeImpl::GetType<int64_t>();
    case at::kBool:
      return onnxruntime::DataTypeImpl::GetType<bool>();
    default:
      ORT_THROW("Unsupport aten scalar type: ", dtype);
  }
}

OrtDevice create_ort_device(const c10::DeviceType device_type, const c10::DeviceIndex device_id) {
    // Assume ID is the same in ORT and Pytorch.
    OrtDevice::DeviceId ort_device_id = static_cast<OrtDevice::DeviceId>(device_id);
    // Translate Pytorch device type to ORT device type.
    OrtDevice::DeviceType ort_device_type;
    switch (device_type) {
        case c10::DeviceType::CPU:
            ort_device_type = OrtDevice::CPU;
            break;
        case c10::DeviceType::CUDA:
            ort_device_type = OrtDevice::GPU;
            break;
        default:
        ORT_THROW(
          "Unsupport Pytorch device.",
          " Type: ", c10::DeviceTypeName(device_type), ",",
          " ID: ", device_id);
    };

    // TODO: check if we should always do OrtAllocatorType::OrtDeviceAllocator.
    return OrtDevice(ort_device_type, OrtDevice::MemType::DEFAULT, ort_device_id);
}


OrtMemoryInfo create_ort_memory_info(const c10::Device device) {
    std::string ort_device_name = device.str();
    OrtDevice ort_device = create_ort_device(device.type(), device.index());
    return OrtMemoryInfo(
        ort_device_name.c_str(), OrtAllocatorType::OrtDeviceAllocator, ort_device, ort_device.Id());
}

OrtMemoryInfo create_ort_cpu_memory_info(const char* name) {
  OrtDevice device(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0);
  return OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator, device, device.Id());
}


OrtValue create_ort_tensor_value(const at::Tensor& tensor) {
  onnxruntime::MLDataType element_type = to_ort_scalar_type(tensor.scalar_type());
  onnxruntime::TensorShape shape(tensor.sizes().vec());
  OrtMemoryInfo memory_info = create_ort_memory_info(tensor.device());
  // This tensor's life time is controlled by Pytorch.
  // TODO: consider to let ORT also own that tensor.
  std::unique_ptr<onnxruntime::Tensor> ort_tensor = std::make_unique<onnxruntime::Tensor>(
      element_type, shape,
      tensor.data_ptr(), memory_info);

  OrtValue ort_value;
  ort_value.Init(
      ort_tensor.release(),
      onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
      onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());
  return ort_value;
}

OrtValue create_ort_scalar_value(const at::Scalar& scalar) {
  onnxruntime::MLDataType element_type = to_ort_scalar_type(scalar.type());
  onnxruntime::TensorShape shape({});
  // This tensor's life time is controlled by Pytorch.
  // TODO: consider to let ORT also own that tensor.
  void* data_ptr = nullptr;
  std::function<void()> data_deleter;

  switch (scalar.type()) {
    case at::kFloat: {
      data_ptr = new float;
      *reinterpret_cast<float*>(data_ptr) = scalar.toFloat();
      data_deleter = [=]() {
        delete reinterpret_cast<float*>(data_ptr);
      };
      break;
    }
    case at::kDouble: {
      data_ptr = new double;
      *reinterpret_cast<double*>(data_ptr) = scalar.toDouble();
      data_deleter = [=]() {
        delete reinterpret_cast<double*>(data_ptr);
      };
      break;
    }
    case at::kBFloat16: {
      at::BFloat16 valBFloat16 = scalar.toBFloat16();
      Ort::BFloat16_t *valOrtBFloat16 = reinterpret_cast<Ort::BFloat16_t *>(&valBFloat16);
      data_ptr = new Ort::BFloat16_t;
      *reinterpret_cast<Ort::BFloat16_t*>(data_ptr) = *valOrtBFloat16;
      data_deleter = [=]() {
        delete reinterpret_cast<Ort::BFloat16_t*>(data_ptr);
      };
      break;
    }
    default:
      ORT_THROW("Unsupport aten scalar type: ", scalar.type());
  }

  OrtMemoryInfo memory_info = create_ort_cpu_memory_info("at::Scalar on CPU");
  std::unique_ptr<onnxruntime::Tensor> ort_tensor = std::make_unique<onnxruntime::Tensor>(
      element_type, shape,
      data_ptr, memory_info);

  std::function<void(void*)> deleter = [=](void* p) {
   data_deleter();
   onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc()(p);
  };

  OrtValue ort_value;
  ort_value.Init(
      ort_tensor.release(),
      onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
      deleter);
  return ort_value;
}


c10::ScalarType create_torch_element_type(const onnxruntime::PrimitiveDataTypeBase* elem_type) {
  ORT_ENFORCE(elem_type, "Element type pointer cannot be NULL.");
  switch (static_cast<ONNX_NAMESPACE::TensorProto_DataType>(elem_type->GetDataType())) {
    case onnxruntime::data_types_internal::ToTensorDataType<float>() : {
      return c10::kFloat;  
    }
    case onnxruntime::data_types_internal::ToTensorDataType<double>() : {
      return c10::kDouble;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<onnxruntime::BFloat16>() : {
      return c10::kBFloat16;
    }
    default:
      ORT_THROW("Unsupport ORT scalar type.");
  }
}

c10::DeviceType create_torch_device_type(const OrtDevice::DeviceType& device_type) {
  switch (device_type) {
    case OrtDevice::CPU:
      return c10::DeviceType::CPU;
    case OrtDevice::GPU:
      return c10::DeviceType::CUDA;
    default:
      ORT_THROW("Unsupport ORT device.");
  }
}

c10::DeviceIndex create_torch_device_index(const OrtDevice::DeviceId& device_id) {
  return static_cast<c10::DeviceIndex>(device_id);
}

at::Tensor create_at_tensor(const onnxruntime::Tensor& tensor) {
  const OrtDevice& device = tensor.Location().device;
  auto options = torch::TensorOptions()
    .dtype(create_torch_element_type(tensor.DataType()->AsPrimitiveDataType()))
    .layout(torch::kStrided)
    .device(create_torch_device_type(device.Type()), create_torch_device_index(device.Type()))
    .requires_grad(false);

  at::Tensor new_tensor = torch::empty(tensor.Shape().GetDimsAsVector(), options);

  switch (device.Type()) {
    case OrtDevice::CPU:
      //auto tensor = torch::from_blob(
      //  v.data(),
      //  v.size(),
      //  /*deleter=*/[&called](void* data) { called = true; },
      //  torch::kInt32);
      std::memcpy(new_tensor.data_ptr(), tensor.DataRaw(), tensor.SizeInBytes());
      break;
    // TODO: Add GPU.
    default:
      ORT_THROW("Unsupport ORT device.");
  }
  return new_tensor;
}

//OrtValue create_ort_tensor_value() {
//  OrtDevice device;
//  float* data_ptr = new float[2];
//  onnxruntime::TensorShape shape(std::vector<int64_t>{2});
//  onnxruntime::MLDataType data_type = onnxruntime::DataTypeImpl::GetType<float>();
//  OrtMemoryInfo info("CPU", OrtAllocatorType::OrtDeviceAllocator, device, device.Id());
//  std::unique_ptr<onnxruntime::Tensor> p_tensor = std::make_unique<onnxruntime::Tensor>(
//      data_type, shape,
//      data_ptr, info);
//
//  OrtValue ort_value;
//  std::function<void(void*)> deleter = [=](void* p) {
//    delete[] data_ptr;
//   onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc()(p);
//  };
//
//  ort_value.Init(p_tensor.release(), onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(), deleter);
//  return ort_value;
//}
//
//template<typename T>
//OrtValue to_ort_value(const at::Tensor& tensor) {
//  auto shape = tensor.sizes().vec();
//  auto device = to_ort_device(tensor.device());
//  OrtDevice device;
//  OrtMemoryInfo info("CPU", OrtAllocatorType::OrtDeviceAllocator, device, device.Id());
//  std::unique_ptr<onnxruntime::Tensor> p_tensor = std::make_unique<onnxruntime::Tensor>(
//      to_ort_scalar_type(tensor.scalar_type()), shape,
//      tensor.data_ptr(), info);
//
//  OrtValue ort_value;
//  std::function<void(void*)> deleter = [&](void* p) {
//   std::cout << "Release at::Tensor's data_ptr " << tensor.data_ptr() << std::endl;
//   onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc()(p);
//  };
//
//  ort_value.Init(p_tensor.release(), onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(), deleter);
//  return ort_value;
//}
//
OrtValue create_2() {
  OrtDevice device;
  float* data_ptr = new float[2];
  onnxruntime::TensorShape shape(std::vector<int64_t>{2});
  onnxruntime::MLDataType data_type = onnxruntime::DataTypeImpl::GetType<float>();
  OrtMemoryInfo info("CPU", OrtAllocatorType::OrtDeviceAllocator, device, device.Id());
  std::unique_ptr<onnxruntime::Tensor> p_tensor = std::make_unique<onnxruntime::Tensor>(
      data_type, shape,
      data_ptr, info);

  OrtValue ort_value;
  std::function<void(void*)> deleter = [=](void* p) {
    delete[] data_ptr;
   onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc()(p);
  };

  ort_value.Init(p_tensor.release(), onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(), deleter);
  return ort_value;
}
//
//OrtValue create_ort_bool_tensor_value() {
//  OrtDevice device;
//  bool* data_ptr = new bool[2];
//  onnxruntime::TensorShape shape(std::vector<int64_t>{2});
//  onnxruntime::MLDataType data_type = onnxruntime::DataTypeImpl::GetType<bool>();
//  OrtMemoryInfo info("CPU", OrtAllocatorType::OrtDeviceAllocator, device, device.Id());
//  std::unique_ptr<onnxruntime::Tensor> p_tensor = std::make_unique<onnxruntime::Tensor>(
//      data_type, shape,
//      data_ptr, info);
//
//  OrtValue ort_value;
//  std::function<void(void*)> deleter = [=](void* p) {
//    delete[] data_ptr;
//   onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc()(p);
//  };
//
//  ort_value.Init(p_tensor.release(), onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(), deleter);
//  return ort_value;
//}

bool Accelerator::supported(const torch::jit::Node* node) {
  switch (node->kind()) {
    case aten::relu:
    case aten::mul:
    //case aten::gt:
    //case aten::eq:
    case prim::Constant:
    //case aten::threshold_backward:
      std::cout << "[compiler.cc] Support " << *node;  //<< std::endl;
      return true;
    default:
      std::cout << "[compiler.cc] Not support " << *node; //<< std::endl;
      return false;
  }
}

void Accelerator::run(torch::jit::Stack& stack) {
  // Get the number of expected inputs to the graph we are compiling
  const at::ArrayRef<torch::jit::Value*>& graph_inputs = subgraph_->inputs();
  const auto num_inputs = graph_inputs.size();

  // Pop these inputs from the stack.
  at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, num_inputs);

  std::cout << "JIT sub-graph: " << std::endl;
  std::cout << *subgraph_ << std::endl;
  // If we haven't compiled for the shape/device of these inputs before,
  // do so now.
  torch::jit::CompleteArgumentSpec spec{false, at::ArrayRef<c10::IValue>(inputs)};
  if (cache_.find(spec) == cache_.end()) {
    cache_[spec] = compile(inputs);
  }

  // Run the compiled function!
  auto outputs = cache_[spec](inputs);

  torch::jit::drop(stack, num_inputs);

  for (auto& output : outputs) {
    auto var = torch::autograd::make_variable(output.toTensor());
    stack.push_back(c10::IValue(var));
  }
}

CompiledCode Accelerator::compile(
    at::ArrayRef<c10::IValue>& inputs) {
  // First we run through some checks to make sure the inputs are Tensors and
  // that the implied semantics are pointwise.

  TORCH_CHECK(inputs.size(), "Need at least one input.");
  for (const auto& input : inputs) {
    TORCH_CHECK(input.isTensor(), "Compiler can only handle Tensor inputs.");
  }
  auto size = inputs[0].toTensor().numel();
  for (const auto& input : inputs) {
    TORCH_CHECK(
        input.toTensor().numel() == size,
        "Compiler can only handle pointwise operations without broadcasting.");
  }

  //onnxruntime::Environment& pybind_default_env = GetLtcEnv();
  //onnxruntime::SessionOptions sess_opts;
  //onnxruntime::InferenceSession sess(sess_opts, pybind_default_env);
  //std::string model_path;
  //{
  //  //const at::ArrayRef<torch::jit::Value*>& graph_inputs = subgraph_->inputs();
  //  const auto num_inputs = subgraph_->inputs().size();
  //  for (size_t i = 0; i < num_inputs; ++i) {
  //      auto input_symbol = subgraph_->inputs()[i];
  //      auto input_value = inputs[i];
  //      input_symbol->setType(input_value.type());
  //  }

  //  std::cout << "[ext] gil_scoped_acquire" << std::endl;
  //  pybind11::gil_scoped_acquire guard{};
  //  std::cout << "[ext] create to_onnx" << std::endl;
  //  pybind11::function to_onnx =
  //      pybind11::reinterpret_borrow<pybind11::function>(   // cast from 'object' to 'function - use `borrow` (copy) or `steal` (move)
  //          pybind11::module::import("torch.onnx.utils").attr("_optimize_graph_1")  // import method "min_rosen" from python "module"
  //      );
  //  std::cout << "[ext] print JIT graph in Python:" << std::endl;
  //  pybind11::print(subgraph_);
  //  std::cout << "[ext] call to_onnx" << std::endl;
  //  auto result = to_onnx(subgraph_, ::torch::onnx::OperatorExportTypes::ONNX_ATEN_FALLBACK);
  //  std::cout << "[ext] print ONNX graph:" << std::endl;
  //  pybind11::print(subgraph_);

  //  //std::stringstream ss;
  //  //ss << "Oops-" << n;
  //  //++n;
  //  //Ort::SessionOptions sessionOptions;
  //  //Ort::Session session(env, "/bert_ort/wechi/model_QI4C.onnx", sessionOptions);

  //  ORT_THROW_IF_ERROR(sess.Load(result.cast<std::string>()));
  //  ORT_THROW_IF_ERROR(sess.Initialize());
  //  auto a = create_2();
  //  // //auto b = create_ort_bool_tensor_value();

  //  onnxruntime::RunOptions run_options;
  //  std::vector<std::string> feed_names{"0"};
  //  std::vector<OrtValue> feeds{a};
  //  std::vector<std::string> output_names{"2", "1"};
  //  std::vector<OrtValue> fetches;
  //  std::cout << "[ORT] run sess" << std::endl;
  //  ORT_THROW_IF_ERROR(sess.Run(run_options, feed_names, feeds, output_names, &fetches));
  //  std::cout << "[ORT] run sess done" << std::endl;
  //  model_path = result.cast<std::string>();
  //}

  //const at::ArrayRef<torch::jit::Value*>& graph_inputs = subgraph_->inputs();
  const auto num_inputs = subgraph_->inputs().size();
  for (size_t i = 0; i < num_inputs; ++i) {
      auto input_symbol = subgraph_->inputs()[i];
      auto input_value = inputs[i];
      input_symbol->setType(input_value.type());
  }

  pybind11::gil_scoped_acquire guard{};
  pybind11::function to_onnx =
      pybind11::reinterpret_borrow<pybind11::function>(   // cast from 'object' to 'function - use `borrow` (copy) or `steal` (move)
          pybind11::module::import("torch.onnx.utils").attr("_optimize_graph_1")  // import method "min_rosen" from python "module"
      );
  auto result = to_onnx(subgraph_, ::torch::onnx::OperatorExportTypes::ONNX_ATEN_FALLBACK);
  std::string  model_path = result.cast<std::string>();

  // This function wraps the function pointer we bound our assembly to
  // Adheres to the CompiledCode interface defined in compiler.h
  auto compiled_func = [this, model_path](at::ArrayRef<c10::IValue>& inputs) {
    onnxruntime::Environment& pybind_default_env = GetLtcEnv();
    onnxruntime::SessionOptions sess_opts;
    onnxruntime::InferenceSession sess(sess_opts, pybind_default_env);
    ORT_THROW_IF_ERROR(sess.Load(model_path));
    ORT_THROW_IF_ERROR(sess.Initialize());

    onnxruntime::RunOptions run_options;
    std::vector<std::string> feed_names;
    std::vector<OrtValue> feeds;
    std::vector<std::string> output_names;
    std::vector<OrtValue> fetches;

    const auto num_inputs = subgraph_->inputs().size();
    for (size_t i = 0; i < num_inputs; ++i) {
        feed_names.push_back(subgraph_->inputs().at(i)->debugName());
        if (subgraph_->inputs().at(i)->type()->kind() == c10::TypeKind::TensorType) {
          feeds.push_back(create_ort_tensor_value(inputs.at(i).toTensor()));
        } else {
          // TODO: handle other type correctly.
          feeds.push_back(create_ort_scalar_value(inputs.at(i).toScalar()));
        }
    }
    const auto num_outputs = subgraph_->outputs().size();
    for (size_t i = 0; i < num_outputs; ++i) {
        output_names.push_back(subgraph_->outputs().at(i)->debugName());
    }

    std::cout << "[accelerator.cpp] Run" << std::endl;
    ORT_THROW_IF_ERROR(sess.Run(run_options, feed_names, feeds, output_names, &fetches));
    std::cout << "[accelerator.cpp] Run done" << std::endl;

    std::vector<c10::IValue> outputs;
    for (auto value : fetches) {
        onnxruntime::Tensor* tensor = value.GetMutable<onnxruntime::Tensor>();
        at::Tensor new_tensor = create_at_tensor(*tensor);
        outputs.push_back(new_tensor);
    }

    return outputs;

    //// __value__ is the symbol of arena[__value__] tensor.
    //std::map<torch::jit::Value*, c10::IValue> arena;
    //for (auto value : subgraph_->inputs()) {
    //  arena[value] = inputs[value->offset()];
    //}

    //for (auto node : subgraph_->nodes()) {
    //  switch (node->kind()) {
    //    case aten::relu: {
    //      std::cout << "[compiler.cc] see aten::relu" << std::endl;
    //      auto x = arena[node->inputs()[0]].toTensor().contiguous();
    //      auto y = at::relu(x);
    //      arena[node->outputs()[0]] = y;
    //      break;
    //    }
    //    case aten::mul: {
    //      std::cout << "[compiler.cc] see aten::mul" << std::endl;
    //      auto x = arena[node->inputs()[0]].toTensor().contiguous();
    //      auto y = arena[node->inputs()[1]].toTensor().contiguous();
    //      auto z = at::mul(x, y);
    //      arena[node->outputs()[0]] = z;
    //      break;
    //    }
    //    case aten::gt: {
    //      std::cout << "[compiler.cc] see aten::gt" << std::endl;
    //      auto x = arena[node->inputs()[0]].toTensor().contiguous();
    //      int y = arena[node->inputs()[1]].toInt();
    //      arena[node->outputs()[0]] = at::gt(x, y);
    //      break;
    //    }
    //    case aten::eq: {
    //      std::cout << "[compiler.cc] see aten::eq" << std::endl;
    //      auto x = arena[node->inputs()[0]].toTensor().contiguous();
    //      auto y = arena[node->inputs()[0]].toTensor().contiguous();
    //      auto z = at::eq(x, y);
    //      arena[node->outputs()[0]] = z;
    //      break;
    //    }
    //    case aten::type_as: {
    //      std::cout << "[compiler.cc] see aten::type_as" << std::endl;
    //      auto x = arena[node->inputs()[0]].toTensor().contiguous();
    //      auto y = arena[node->inputs()[1]].toTensor();
    //      auto z = x.to(y.options());
    //      arena[node->outputs()[0]] = z;
    //      break;
    //    }
    //    case prim::Constant: {
    //      std::cout << "[compiler.cc] see prim::Constant" << std::endl;
    //      arena[node->outputs()[0]] = torch::jit::toIValue(node->outputs()[0]).value();
    //      break;
    //    }
    //    case aten::threshold_backward: {
    //      std::cout << "[compiler.cc] see aten::threshold_backward" << std::endl;
    //      auto x = arena[node->inputs()[0]].toTensor().contiguous();
    //      auto y = arena[node->inputs()[1]].toTensor();
    //      //auto z = arena[node->inputs()[2]].to<at::Scalar>();
    //      auto z = arena[node->inputs()[2]].toTensor().contiguous();
    //      auto z_data = z.data_ptr<int64_t>();
    //      std::cout << "To call\n";
    //      auto w = at::_ops::threshold_backward::call(x, y, at::Scalar(*z_data));
    //      std::cout << "Done call\n";
    //      arena[node->outputs()[0]] = w;
    //      break;
    //    }
    //  }
    //}

    //std::vector<c10::IValue> outputs;
    //for (auto value : subgraph_->outputs()) {
    //  outputs.push_back(arena[value]);
    //}

    //return outputs;
  };

  return compiled_func;
}
