// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <sstream>

#include "gmock/gmock.h"
#include "test/providers/provider_test_utils.h"
#include <csignal>
#include <exception>
#include <memory>
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/inference_session.h"
#include "test/util/include/default_providers.h"

#ifdef MICROSOFT_AUTOML
#include "automl_ops/automl_featurizers.h"
namespace dtf = Microsoft::Featurizer::DateTimeFeaturizer;
#endif

using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {

// Check functions for tensor types
template <typename T>
void sort_expected_and_actual_buffers(const T* expected, const T* actual, int64_t size) {
  std::sort(const_cast<T*>(expected), const_cast<T*>(expected + size));
  std::sort(const_cast<T*>(actual), const_cast<T*>(actual + size));
}

// Check functions for tensor types
template <typename T>
void sort_expected_and_actual_buffers(std::vector<T> expected, std::vector<T> actual) {
  ORT_ENFORCE(expected.size() == actual.size(), "The 2 containers contain different number of elements");
  sort_expected_and_actual_buffers(expected.data(), actual.data(), expected.size());
}

// The default implementation compares for equality, specialized versions for other types are below
template <typename T>
void Check(const OpTester::Data& expected_data, const Tensor& output_tensor, const std::string& provider_type) {
  auto& expected_tensor = expected_data.data_.Get<Tensor>();
  auto* expected = expected_tensor.template Data<T>();
  auto* output = output_tensor.template Data<T>();
  auto size = output_tensor.Shape().Size();

  if (expected_data.sort_output_) {
    // if order can be jumbled in the output of an operator, sort both the expected and output buffers prior to
    // comparison this is a "best-effort" algo and should satisfy the requirement for the few ops that do require this
    // support without investing in a more sophisticated infrastructure for the same
    sort_expected_and_actual_buffers<T>(expected, output, size);
  }

  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(expected[i], output[i]) << "i:" << i << ", provider_type: " << provider_type;
  }
}

template <>
void Check<double>(const OpTester::Data& expected_data, const Tensor& output_tensor,
                   const std::string& provider_type) {
  auto& expected_tensor = expected_data.data_.Get<Tensor>();
  auto* expected = expected_tensor.template Data<double>();
  auto* output = output_tensor.template Data<double>();
  auto size = output_tensor.Shape().Size();

  bool has_abs_err = expected_data.absolute_error_.has_value();
  bool has_rel_err = expected_data.relative_error_.has_value();

  // deal with rare cases in which order of output data from a kernel MAY be undefined
  if (expected_data.sort_output_) {
    sort_expected_and_actual_buffers<double>(expected, output, size);
  }

  double threshold = 0.001;
#ifdef USE_CUDA
  threshold = 0.005;
#endif

  for (int i = 0; i < size; ++i) {
    if (std::isinf(expected[i])) {  // Test infinity for equality
      EXPECT_EQ(expected[i], output[i]) << "i:" << i;
    } else if (std::isnan(expected[i])) {
      EXPECT_TRUE(std::isnan(output[i])) << "Expected output " << i << " to be NaN";
    } else {
      if (!has_abs_err && !has_rel_err) {
        // the default for existing tests
        EXPECT_NEAR(expected[i], output[i], threshold) << "i:" << i << ", provider_type: " << provider_type;
      } else {
        if (has_abs_err) {
          EXPECT_NEAR(expected[i], output[i], expected_data.absolute_error_.value())
              << "i:" << i << ", provider_type: " << provider_type;
        }
        if (has_rel_err) {
          EXPECT_NEAR(expected[i], output[i], expected_data.relative_error_.value() * std::abs(expected[i]))
              << "i:" << i << ", provider_type: " << provider_type;
        }
      }
    }
  }
}

template <>
void Check<float>(const OpTester::Data& expected_data, const Tensor& output_tensor, const std::string& provider_type) {
  auto& expected_tensor = expected_data.data_.Get<Tensor>();
  auto* expected = expected_tensor.template Data<float>();
  auto* output = output_tensor.template Data<float>();
  auto size = output_tensor.Shape().Size();

  bool has_abs_err = expected_data.absolute_error_.has_value();
  bool has_rel_err = expected_data.relative_error_.has_value();

  // deal with rare cases in which order of output data from a kernel MAY be undefined
  if (expected_data.sort_output_) {
    sort_expected_and_actual_buffers<float>(expected, output, size);
  }

  float threshold = 0.001f;
#ifdef USE_CUDA
  threshold = 0.005f;
#endif

  for (int i = 0; i < size; ++i) {
    if (std::isinf(expected[i])) {  // Test infinity for equality
      EXPECT_EQ(expected[i], output[i]) << "i:" << i;
    } else if (std::isnan(expected[i])) {
      EXPECT_TRUE(std::isnan(output[i])) << "Expected output " << i << " to be NaN";
    } else {
      if (!has_abs_err && !has_rel_err) {
        // the default for existing tests
        EXPECT_NEAR(expected[i], output[i], threshold) << "i:" << i << ", provider_type: " << provider_type;
      } else {
        if (has_abs_err) {
          EXPECT_NEAR(expected[i], output[i], expected_data.absolute_error_.value())
              << "i:" << i << ", provider_type: " << provider_type;
        }
        if (has_rel_err) {
          EXPECT_NEAR(expected[i], output[i], expected_data.relative_error_.value() * std::abs(expected[i]))
              << "i:" << i << ", provider_type: " << provider_type;
        }
      }
    }
  }
}

template <>
void Check<MLFloat16>(const OpTester::Data& expected_data, const Tensor& output_tensor,
                      const std::string& provider_type) {
  auto& expected_tensor = expected_data.data_.Get<Tensor>();
  auto* expected = expected_tensor.template Data<MLFloat16>();
  auto* output = output_tensor.template Data<MLFloat16>();
  auto size = output_tensor.Shape().Size();

  std::vector<float> f_expected(size);
  std::vector<float> f_output(size);
  ConvertMLFloat16ToFloat(expected, f_expected.data(), static_cast<int>(size));
  ConvertMLFloat16ToFloat(output, f_output.data(), static_cast<int>(size));

  // deal with rare cases in which order of output data from a kernel MAY be undefined
  if (expected_data.sort_output_) {
    sort_expected_and_actual_buffers<float>(f_expected, f_output);
  }

  float threshold = 0.001f;
  for (int i = 0; i < size; ++i) {
    if (std::isinf(f_expected[i]))  // Test infinity for equality
      EXPECT_EQ(f_expected[i], f_output[i]) << "i:" << i;
    else {
      // the default for existing tests
      EXPECT_NEAR(f_expected[i], f_output[i], threshold) << "i:" << i << ", provider_type: " << provider_type;
    }
  }
}

template <>
void Check<BFloat16>(const OpTester::Data& expected_data, const Tensor& output_tensor,
                     const std::string& provider_type) {
  auto& expected_tensor = expected_data.data_.Get<Tensor>();
  auto* expected = expected_tensor.template Data<BFloat16>();
  auto* output = output_tensor.template Data<BFloat16>();
  auto size = output_tensor.Shape().Size();

  std::vector<float> f_expected(size);
  std::vector<float> f_output(size);
  BFloat16ToFloat(expected, f_expected.data(), static_cast<size_t>(size));
  BFloat16ToFloat(output, f_output.data(), static_cast<size_t>(size));

  // deal with rare cases in which order of output data from a kernel MAY be undefined
  if (expected_data.sort_output_) {
    sort_expected_and_actual_buffers<float>(f_expected, f_output);
  }

  /// XXX: May need to adjust threshold as BFloat is coarse
  float threshold = 0.001f;
  for (int i = 0; i < size; ++i) {
    if (std::isinf(f_expected[i]))  // Test infinity for equality
      EXPECT_EQ(f_expected[i], f_output[i]);
    else {
      // the default for existing tests
      EXPECT_NEAR(f_expected[i], f_output[i], threshold) << "provider_type: " << provider_type;
    }
  }
}

template <typename Type>
void CheckDispatch(MLDataType type, const OpTester::Data& expected_data, const Tensor& output_tensor,
                   const std::string& provider_type) {
  if (type == DataTypeImpl::GetType<Type>())
    Check<Type>(expected_data, output_tensor, provider_type);
  else
    ORT_THROW("OpTester:Check() not implemented for output tensor type of ", type);
}

template <typename Type, typename Next, typename... Types>
void CheckDispatch(MLDataType type, const OpTester::Data& expected_data, const Tensor& output_tensor,
                   const std::string& provider_type) {
  if (type == DataTypeImpl::GetType<Type>())
    Check<Type>(expected_data, output_tensor, provider_type);
  else
    CheckDispatch<Next, Types...>(type, expected_data, output_tensor, provider_type);
}

void Check(const OpTester::Data& expected_data, const Tensor& output_tensor, const std::string& provider_type) {
  ORT_ENFORCE(expected_data.data_.Get<Tensor>().Shape() == output_tensor.Shape(),
              "Expected output shape [" + expected_data.data_.Get<Tensor>().Shape().ToString() +
                  "] did not match run output shape [" + output_tensor.Shape().ToString() + "] for " +
                  expected_data.def_.Name());

  CheckDispatch<bool, float, double, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                std::string, MLFloat16, BFloat16>(output_tensor.DataType(), expected_data, output_tensor,
                                                  provider_type);
}

// Check for non tensor types

template <typename T>
void Check(const OpTester::Data& expected_data, const T& run_output, const std::string& provider_type) {
  EXPECT_EQ(expected_data.data_.Get<T>(), run_output) << "provider_type: " << provider_type;
}

template <>
void Check<TensorSeq>(const OpTester::Data& expected_data, const TensorSeq& output_seq,
                      const std::string& provider_type) {
  const auto& exp_seq = expected_data.data_.Get<TensorSeq>();

  // first ensure data types match
  EXPECT_EQ(exp_seq.dtype, output_seq.dtype) << "Data types don't match: Expected: " << DataTypeImpl::ToString(exp_seq.dtype)
                                             << " Output: " << output_seq.dtype << " provider_type: " << provider_type;

  // check num of contained tensors
  size_t expected_num_tensors = exp_seq.tensors.size();
  size_t output_num_tensors = output_seq.tensors.size();
  EXPECT_EQ(expected_num_tensors, output_num_tensors) << "Mismatch in number of tensors in the sequence"
                                                      << " Expected: " << expected_num_tensors << " Output: "
                                                      << output_num_tensors << " provider_type: " << provider_type;

  // now check the contents of the tensors
  auto null_deleter = [](void*) {};

  for (size_t i = 0; i < output_num_tensors; ++i) {
    OrtValue temp_value;
    // Reason for null_deleter: we don't want the tensor destructor to be called as part of this OrtValue destructor
    // as we're creating this OrtValue only to reuse the Check functionality
    temp_value.Init(const_cast<Tensor*>(&exp_seq.tensors[i]), DataTypeImpl::GetType<Tensor>(), null_deleter);
    OpTester::Data temp_data(NodeArg("dummy", nullptr), std::move(temp_value), optional<float>(), optional<float>());
    Check(temp_data, output_seq.tensors[i], provider_type);
  }
}

template <typename Type>
void CheckDispatch(MLDataType type, const OpTester::Data& expected_data, OrtValue& ort_value,
                   const std::string& provider_type) {
  if (type == DataTypeImpl::GetType<Type>())
    Check<Type>(expected_data, ort_value.Get<Type>(), provider_type);
  else
    ORT_THROW("OpTester:Check() not implemented for output tensor type of ", type);
}

template <typename Type, typename Next, typename... Types>
void CheckDispatch(MLDataType type, const OpTester::Data& expected_data, OrtValue& ort_value,
                   const std::string& provider_type) {
  if (type == DataTypeImpl::GetType<Type>())
    Check<Type>(expected_data, ort_value.Get<Type>(), provider_type);
  else
    CheckDispatch<Next, Types...>(type, expected_data, ort_value, provider_type);
}

void Check(const OpTester::Data& expected_data, OrtValue& ort_value, const std::string& provider_type) {
#ifdef MICROSOFT_AUTOML
  CheckDispatch<dtf::TimePoint, VectorMapStringToFloat, VectorMapInt64ToFloat, TensorSeq>(expected_data.data_.Type(), expected_data, ort_value,
                                                                                          provider_type);
#else
  CheckDispatch<VectorMapStringToFloat, VectorMapInt64ToFloat, TensorSeq>(expected_data.data_.Type(), expected_data, ort_value,
                                                                          provider_type);
#endif
}

void DebugTrap() {
#if _MSC_VER
  __debugbreak();
#else
  raise(SIGTRAP);
#endif
}

OpTester::~OpTester() {
#ifndef NDEBUG
  if (!run_called_) {
    std::cerr << "Someone forgot to call OpTester::Run()" << std::endl;
    DebugTrap();
  }
#endif
}

void OpTester::FillFeedsAndOutputNames(std::unordered_map<std::string, OrtValue>& feeds,
                                       std::vector<std::string>& output_names) {
  for (auto& output : output_data_) {
    if (output.def_.Exists()) output_names.push_back(output.def_.Name());
  }

  for (size_t i = 0; i < input_data_.size(); ++i) {
    if (std::find(initializer_index_.begin(), initializer_index_.end(), i) == initializer_index_.end() &&
        input_data_[i].def_.Exists()) {
      feeds[input_data_[i].def_.Name()] = input_data_[i].data_;
    }
  }
}

void OpTester::SetOutputAbsErr(const char* name, float v) {
  auto it = std::find_if(output_data_.begin(), output_data_.end(),
                         [name](Data& data) { return (data.def_.Name() == name); });
  ORT_ENFORCE(it != output_data_.end());
  it->absolute_error_ = optional<float>(v);
}

void OpTester::SetOutputRelErr(const char* name, float v) {
  auto it = std::find_if(output_data_.begin(), output_data_.end(),
                         [name](Data& data) { return (data.def_.Name() == name); });
  ORT_ENFORCE(it != output_data_.end());
  it->relative_error_ = optional<float>(v);
}

void OpTester::AddNodes(onnxruntime::Graph& graph, std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                        std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                        std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) {
  // default behavior is to create a single Node for the op being tested, with node inputs/outputs
  // being 1:1 with graph inputs/outputs.
  auto& node = graph.AddNode("node1", op_, op_, graph_input_defs, graph_output_defs, nullptr, domain_);

  // Add the attributes if any
  for (auto& add_attribute_fn : add_attribute_funcs) add_attribute_fn(node);
}

void OpTester::AddInitializers(onnxruntime::Graph& graph) {
  for (auto index : initializer_index_) {
    auto& data = input_data_[index];
    auto& tensor = data.data_.Get<Tensor>();
    ONNX_NAMESPACE::TensorProto tensor_proto;
    // 1. set dimension
    auto& shape = tensor.Shape();
    for (auto& dim : shape.GetDims()) {
      tensor_proto.add_dims(dim);
    }
    // 2. set type
    tensor_proto.set_data_type(data.def_.TypeAsProto()->tensor_type().elem_type());
    // 3. data
    if (data.def_.TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
      const std::string* string_data = tensor.Data<std::string>();
      for (auto i = 0; i < shape.Size(); i++) {
        tensor_proto.add_string_data(string_data[i]);
      }
    } else {
      auto buffer_size = tensor.DataType()->Size() * shape.Size();
      tensor_proto.set_raw_data(tensor.DataRaw(), buffer_size);
    }
    // 4. name
    tensor_proto.set_name(data.def_.Name());
    graph.AddInitializedTensor(tensor_proto);
  }
}

std::unique_ptr<onnxruntime::Model> OpTester::BuildGraph() {
  // Generate the input & output def lists
  std::vector<onnxruntime::NodeArg*> node_input_defs;
  std::vector<onnxruntime::NodeArg*> output_defs;

  for (size_t i = 0; i < input_data_.size(); ++i) {
    node_input_defs.push_back(&input_data_[i].def_);
  }

  for (auto& data : output_data_) {
    output_defs.push_back(&data.def_);
  }

  // Create a simple model
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[domain_] = opset_version_;
  auto p_model = onnxruntime::make_unique<onnxruntime::Model>("test", false, ModelMetaData(),
                                                              custom_schema_registries_, domain_to_version);
  onnxruntime::Graph& graph = p_model->MainGraph();
  AddNodes(graph, node_input_defs, output_defs, add_attribute_funcs_);

  // Add Initializer
  AddInitializers(graph);
  return p_model;
}

void OpTester::ExecuteModel(Model& model, InferenceSession& session_object, ExpectResult expect_result,
                            const std::string& expected_failure_string, const RunOptions* run_options,
                            std::unordered_map<std::string, OrtValue> feeds, std::vector<std::string> output_names,
                            const std::string& provider_type) {
  std::string s1;
  const bool rc = model.ToProto().SerializeToString(&s1);
  if (!rc) {
    LOGS_DEFAULT(ERROR) << "Failed to serialize proto to string";
    return;
  }
  std::stringstream sstr(s1);
  auto status = session_object.Load(sstr);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
    return;
  }

  status = session_object.Initialize();
  if (!status.IsOK()) {
    if (expect_result == ExpectResult::kExpectFailure) {
      EXPECT_TRUE(!status.IsOK());
      EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
    } else {
      LOGS_DEFAULT(ERROR) << "Initialize failed with status: " << status.ErrorMessage();
      EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    }
  }

  if (!status.IsOK()) {
    return;
  }

  RunOptions default_run_options{};
  default_run_options.run_tag = op_;
  default_run_options.run_log_verbosity_level = 1;

  std::vector<OrtValue> fetches;
  for (int i = 0; i < num_run_calls_; ++i) {
    fetches.clear();
    status = session_object.Run(run_options ? *run_options : default_run_options, feeds, output_names, &fetches);

    if (status.IsOK()) {
      EXPECT_TRUE(expect_result == ExpectResult::kExpectSuccess) << "Expected failure but Run was successful";
      if (expect_result == ExpectResult::kExpectFailure) {
        return;
      }
    } else {
      if (expect_result == ExpectResult::kExpectFailure) {
        // Disable expected_failure_string checks for MKL-DNN and nGraph EP's
        if (provider_type != kMklDnnExecutionProvider && provider_type != kNGraphExecutionProvider) {
          EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
        }
      } else {
        LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
        EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
      }
      return;
    }
  }

  // Verify the outputs
  // Todo: support check output with map/sequence/....
  size_t idx = 0;
  for (auto& expected_data : output_data_) {
    OrtValue& ort_value = fetches[idx];
    if (ort_value.Fence()) ort_value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, 0);

    if (expected_data.def_.Exists()) {  // optional outputs won't exist
      if (expected_data.data_.IsTensor()) {
        // verify output shape inference when input defs have shape
        if (add_shape_to_tensor_data_) {
          auto out_shape_proto = expected_data.def_.Shape();
          EXPECT_TRUE(out_shape_proto != nullptr);
          const auto& tensor_shape = utils::GetTensorShapeFromTensorShapeProto(*out_shape_proto);
          const auto& inferred_dims = tensor_shape.GetDims();
          const auto& expected_shape = expected_data.data_.Get<Tensor>().Shape();
          EXPECT_TRUE(inferred_dims.size() == expected_shape.NumDimensions());
          for (size_t d = 0; d < inferred_dims.size(); ++d) {
            // check equal unless the input involved a symbolic dimension
            if (inferred_dims[d] != -1)
              EXPECT_EQ(expected_shape[d], inferred_dims[d]) << "Output idx = " << idx << " dim = " << d;
          }
        }
        Check(expected_data, ort_value.Get<Tensor>(), provider_type);
      } else {
        Check(expected_data, ort_value, provider_type);
      }
      ++idx;

      // skip missing trailing optional outputs
      if (idx == fetches.size())
        break;
    }
  }
}

void OpTester::Run(ExpectResult expect_result,
                   const std::string& expected_failure_string,
                   const std::unordered_set<std::string>& excluded_provider_types,
                   const RunOptions* run_options,
                   std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers,
                   ExecutionMode execution_mode) {
  SessionOptions so;
  so.session_logid = op_;
  so.session_log_verbosity_level = 1;
  so.execution_mode = execution_mode;
  so.graph_optimization_level = TransformerLevel::Default;  // 'Default' == off
  Run(so, expect_result, expected_failure_string, excluded_provider_types, run_options, execution_providers);
}

void OpTester::Run(SessionOptions so,  // Take the SessionOptions by value (i.e. make a copy) because we may need to modify it
                   ExpectResult expect_result,
                   const std::string& expected_failure_string,
                   const std::unordered_set<std::string>& excluded_provider_types,
                   const RunOptions* run_options,
                   std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers) {
  std::string cur_provider = "not set";
  try {
#ifndef NDEBUG
    run_called_ = true;
#endif
    auto p_model = BuildGraph();
    auto& graph = p_model->MainGraph();

    Status status = Status::OK();
    if (add_shape_to_tensor_data_ && expect_result == ExpectResult::kExpectFailure) {
      // capture possible exceptions from shape inference for invalid testcase
      try {
        status = graph.Resolve();
      } catch (const std::exception& ex) {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
      }
    } else {
      status = graph.Resolve();
    }

    if (!status.IsOK()) {
      if (expect_result == ExpectResult::kExpectFailure) {
        EXPECT_TRUE(!status.IsOK());
        EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
      } else {
        LOGS_DEFAULT(ERROR) << "Resolve failed with status: " << status.ErrorMessage();
        EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
      }
    }

    if (!status.IsOK()) {
      return;
    }

    // Hookup the inputs and outputs
    std::unordered_map<std::string, OrtValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(feeds, output_names);
    // Run the model
    static const std::string all_provider_types[] = {
        kCpuExecutionProvider,
        kCudaExecutionProvider,
        kMklDnnExecutionProvider,
        kNGraphExecutionProvider,
        kNupharExecutionProvider,
        kBrainSliceExecutionProvider,
        kTensorrtExecutionProvider,
        kOpenVINOExecutionProvider,
        kDmlExecutionProvider,
        kAclExecutionProvider,};

    bool has_run = false;

    if (execution_providers) {
      for (auto& entry : *execution_providers) {
        if (entry->Type() == kDmlExecutionProvider) {
          so.enable_mem_pattern = false;
          so.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
          break;
        }
      }

      InferenceSession session_object{so};

      ASSERT_TRUE(!execution_providers->empty()) << "Empty execution providers vector.";
      std::string provider_types;

      for (auto& entry : *execution_providers) {
        provider_types += entry->Type() + ":";
        EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(entry)).IsOK());
      }

      ExecuteModel(*p_model, session_object, expect_result, expected_failure_string, run_options, feeds, output_names,
                   provider_types);
    } else {
      for (const std::string& provider_type : all_provider_types) {
        if (excluded_provider_types.count(provider_type) > 0)
          continue;

        cur_provider = provider_type;

        if (provider_type == kDmlExecutionProvider) {
          so.enable_mem_pattern = false;
          so.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
        }
        InferenceSession session_object{so};

        for (auto& custom_session_registry : custom_session_registries_)
          session_object.RegisterCustomRegistry(custom_session_registry);

        std::unique_ptr<IExecutionProvider> execution_provider;
        if (provider_type == onnxruntime::kCpuExecutionProvider)
          execution_provider = DefaultCpuExecutionProvider();
        else if (provider_type == onnxruntime::kCudaExecutionProvider)
          execution_provider = DefaultCudaExecutionProvider();
        else if (provider_type == onnxruntime::kMklDnnExecutionProvider)
          execution_provider = DefaultMkldnnExecutionProvider();
        else if (provider_type == onnxruntime::kNGraphExecutionProvider)
          execution_provider = DefaultNGraphExecutionProvider();
        else if (provider_type == onnxruntime::kNupharExecutionProvider)
          execution_provider = DefaultNupharExecutionProvider();
        else if (provider_type == onnxruntime::kBrainSliceExecutionProvider)
          execution_provider = DefaultBrainSliceExecutionProvider();
        else if (provider_type == onnxruntime::kTensorrtExecutionProvider)
          execution_provider = DefaultTensorrtExecutionProvider();
        else if (provider_type == onnxruntime::kOpenVINOExecutionProvider)
          execution_provider = DefaultOpenVINOExecutionProvider();
        else if (provider_type == onnxruntime::kNnapiExecutionProvider)
          execution_provider = DefaultNnapiExecutionProvider();
        else if (provider_type == onnxruntime::kAclExecutionProvider)
          execution_provider = DefaultAclExecutionProvider();
        // skip if execution provider is disabled
        if (execution_provider == nullptr)
          continue;

        bool valid = true;

        // set execution provider for all nodes in the graph
        for (auto& node : graph.Nodes()) {
          if (node.OpType() == kConstant)
            continue;

          // if node is not registered for the provider, skip
          node.SetExecutionProviderType(provider_type);
          if (provider_type == onnxruntime::kNGraphExecutionProvider ||
              provider_type == onnxruntime::kTensorrtExecutionProvider ||
              provider_type == onnxruntime::kOpenVINOExecutionProvider ||
              provider_type == onnxruntime::kNupharExecutionProvider)
            continue;
          auto reg = execution_provider->GetKernelRegistry();
          const KernelCreateInfo* kci = reg->TryFindKernel(node, execution_provider->Type());
          if (!kci) {
            valid = false;
            for (auto& custom_session_registry : custom_session_registries_) {
              if (custom_session_registry->GetKernelRegistry()->TryFindKernel(node, execution_provider->Type())) {
                valid = true;
                break;
              }
            }

            if (!valid)
              break;
          }
        }

        if (!valid)
          continue;

        has_run = true;

        EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

        ExecuteModel(*p_model, session_object, expect_result, expected_failure_string, run_options, feeds,
                     output_names, provider_type);

        cur_provider = "not set";
      }

      EXPECT_TRUE(has_run) << "No registered execution providers were able to run the model.";
    }
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\nProvider:" << cur_provider << "\n";
    // rethrow as some tests for error handling expect this
    throw;
  }
}
}  // namespace test
}  // namespace onnxruntime
