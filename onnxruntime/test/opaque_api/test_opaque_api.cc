// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <core/common/logging/logging.h>
#include "core/framework/data_types.h"
#include "core/framework/execution_providers.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "onnx/defs/schema.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

// Data container used to ferry data through C API
extern "C" struct ExperimentalDataContainer {
  // This is a string Tensor
  // OrtValue will need to be released when reading/writing is complete
  // by the client code
  OrtValue* str_;
};
extern std::unique_ptr<Ort::Env> ort_env;
namespace onnxruntime {
// A new Opaque type representation
extern const char kMsTestDomain[] = "com.microsoft.test";
extern const char kTestOpaqueType[] = "ComplexOpaqueType";

// This is the actual Opaque type CPP representation
class ExperimentalType {
 public:
  std::string str_;  // Pass a string
};

// The example demonstrates an approach for more complex types and, therefore,
// data containers. It is all too easy to have a data container that has all
// the primitive things. To have more complex objects inside we'd have to employ
// more complex objects, such as Tensors, Maps and Sequences that themselves may
// potentially contain complex objects.

// This example demonstrates how we pass a single const char* string within a scalar
// Tensor that would store data within std::string object. Eventually, the data makes it
// to Opaque experimental type that simply contains std::string
template <>
struct NonTensorTypeConverter<ExperimentalType> {
  // This will get OrtValue from the ExperimentalDataContainer
  // that contains tensor(string), we then repackage string into ExperimentalType
  // and put that ExperiementalType into the OrtValue that is being used as input to a graph
  static void FromContainer(MLDataType dtype, const void* data, size_t data_size, OrtValue& output) {
    ORT_ENFORCE(data_size == sizeof(ExperimentalDataContainer), "Expecting an instance of ExperimentalDataContainer");
    const ExperimentalDataContainer* container = reinterpret_cast<const ExperimentalDataContainer*>(data);

    ORT_ENFORCE(container->str_->IsTensor(), "Expecting a string Tensor");
    const Tensor& str_tensor = container->str_->Get<Tensor>();
    std::unique_ptr<ExperimentalType> p(new ExperimentalType);
    p->str_ = *str_tensor.Data<std::string>();
    output.Init(p.release(), dtype, dtype->GetDeleteFunc());
  }

  // Reading string from the experimental type
  // On the way back we create an OrtValue within ExperimentalDataContainer and put
  // Tensor(string) back into it from ExperiementalType.
  static void ToContainer(const OrtValue& input, size_t data_size, void* data) {
    ORT_ENFORCE(data_size == sizeof(ExperimentalDataContainer), "Expecting an instance of ExperimentalDataContainer");
    ExperimentalDataContainer* container = reinterpret_cast<ExperimentalDataContainer*>(data);

    // Create and populate Tensor
    TensorShape shape({1});
    std::shared_ptr<IAllocator> allocator = std::make_shared<CPUAllocator>();
    std::unique_ptr<Tensor> tp(new Tensor(DataTypeImpl::GetType<std::string>(), shape, allocator));
    *tp->MutableData<std::string>() = input.Get<ExperimentalType>().str_;

    std::unique_ptr<OrtValue> ort_val(new OrtValue);
    const auto* dtype = DataTypeImpl::GetType<Tensor>();
    ort_val->Init(tp.release(), dtype, dtype->GetDeleteFunc());
    container->str_ = ort_val.release();
  }
};

// Register ExperimentalType as Opaque. There will be a call to place it into a map during the execution part
ORT_REGISTER_OPAQUE_TYPE(ExperimentalType, kMsTestDomain, kTestOpaqueType);

// Now write the actual kernel that will operate on the custom Opaque type
// This kernel will take the input as a custom type and will output
// custom type with a different string. This kernel will take the
// original string will replace any instances of 'h' with '_'
class OpaqueCApiTestKernel final : public OpKernel {
 public:
  OpaqueCApiTestKernel(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* ctx) const override {
    const ExperimentalType* input = ctx->Input<ExperimentalType>(0);
    std::string result = input->str_;
    std::replace(result.begin(), result.end(), 'h', '_');
    ExperimentalType* output = ctx->Output<ExperimentalType>(0);
    output->str_ = std::move(result);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    OpaqueCApiTestKernel,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetType<ExperimentalType>()),
    OpaqueCApiTestKernel);

#define ONNX_TEST_OPERATOR_SCHEMA(name) \
  ONNX_TEST_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define ONNX_TEST_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) \
  ONNX_TEST_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define ONNX_TEST_OPERATOR_SCHEMA_UNIQ(Counter, name)            \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name##Counter) ONNX_UNUSED =      \
      ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__)

static void RegisterCustomKernel() {
  // Register our custom type
  MLDataType dtype = DataTypeImpl::GetType<ExperimentalType>();
  DataTypeImpl::RegisterDataType(dtype);

  // Registry the schema
  ONNX_TEST_OPERATOR_SCHEMA(OpaqueCApiTestKernel)
      .SetDoc("Replace all of h chars to _ in the original string contained within experimental type")
      .SetDomain(onnxruntime::kMSFeaturizersDomain)
      .SinceVersion(1)
      .Input(
          0,
          "custom_type_with_string",
          "Our custom type that has a string with h characters",
          "T",
          OpSchema::Single)
      .Output(
          0,
          "custom_type_with_string",
          "Custom type that has the original string with h characters substituted for _",
          "T",
          OpSchema::Single)
      .TypeConstraint(
          "T",
          {"opaque(com.microsoft.test,ComplexOpaqueType)"},
          "Custom type");

  // Register kernel directly to KernelRegistry
  // because we can not create custom ops with Opaque types
  // as input
  // TODO: But that registry is process-wide, such modification is super dangerous.
  BuildKernelCreateInfoFn fn = BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, OpaqueCApiTestKernel)>;
  auto kernel_registry = CPUExecutionProvider(CPUExecutionProviderInfo()).GetKernelRegistry();
  ORT_ENFORCE(kernel_registry->Register(fn()).IsOK());
}

namespace test {

std::string CreateModel() {
  RegisterCustomKernel();
  Model model("ModelWithOpaque", false, logging::LoggingManager::DefaultLogger());
  auto& graph = model.MainGraph();

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  {
    TypeProto exp_type_proto(*DataTypeImpl::GetType<ExperimentalType>()->GetTypeProto());
    // Find out the shape
    auto& input_arg = graph.GetOrCreateNodeArg("Input", &exp_type_proto);
    inputs.push_back(&input_arg);

    //Output is our custom data type. This will return an Opaque type proto
    auto& output_arg = graph.GetOrCreateNodeArg("Output", &exp_type_proto);
    outputs.push_back(&output_arg);

    auto& node = graph.AddNode("OpaqueCApiTestKernel", "OpaqueCApiTestKernel", "Replace all h to underscore",
                               inputs, outputs, nullptr, onnxruntime::kMSFeaturizersDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }
  EXPECT_TRUE(graph.Resolve().IsOK());
  // Get a proto and load from it
  std::string serialized_model;
  auto model_proto = model.ToProto();
  EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
  return serialized_model;
}

TEST(OpaqueApiTest, RunModelWithOpaqueInputOutput) {
  std::string model_str = CreateModel();

  try {
    // initialize session options if needed
    Ort::SessionOptions session_options;
    Ort::Session session(*ort_env.get(), model_str.data(), model_str.size(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // Expecting one input
    size_t num_input_nodes = session.GetInputCount();
    EXPECT_EQ(num_input_nodes, 1U);
    const char* input_name = session.GetInputName(0, allocator);

    size_t num_output_nodes = session.GetOutputCount();
    EXPECT_EQ(num_output_nodes, 1U);
    const char* output_name = session.GetOutputName(0, allocator);

    const char* const input_names[] = {input_name};
    const char* const output_names[] = {output_name};

    // Input
    const std::string input_string{"hi, hello, high, highest"};
    // Expected output
    const std::string expected_output{"_i, _ello, _ig_, _ig_est"};

    // Place a string into Tensor OrtValue and assign to the container
    std::vector<int64_t> input_dims{1};
    Ort::Value container_str = Ort::Value::CreateTensor(allocator, input_dims.data(), input_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

    // No C++ Api to either create a string Tensor or to fill one with string, so we use C
    const char* const input_char_string[] = {input_string.c_str()};
    Ort::ThrowOnError(Ort::GetApi().FillStringTensor(static_cast<OrtValue*>(container_str), input_char_string, 1U));

    // We put this into our container now
    // This container life-span is supposed to eclipse the model running time
    ExperimentalDataContainer container{static_cast<OrtValue*>(container_str)};

    // Now we put our container into OrtValue
    Ort::Value container_val = Ort::Value::CreateOpaque(kMsTestDomain, kTestOpaqueType, container);
    Ort::Value output_val(nullptr);  // empty

    Ort::RunOptions run_options;
    session.Run(run_options, input_names, &container_val, num_input_nodes,
                output_names, &output_val, num_output_nodes);

    ExperimentalDataContainer result;
    // Need to verify that the output match the expected one
    output_val.GetOpaqueData(kMsTestDomain, kTestOpaqueType, result);
    // Wrap the resulting OrtValue into Ort::Value for C++ access and automatic cleanup
    Ort::Value str_tensor_value(result.str_);
    // Run some checks here
    ASSERT_TRUE(str_tensor_value.IsTensor());
    Ort::TypeInfo result_type_info = str_tensor_value.GetTypeInfo();
    auto tensor_info = result_type_info.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(tensor_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    ASSERT_EQ(tensor_info.GetDimensionsCount(), 1U);

    // Get the actual value and compare
    auto str_len = str_tensor_value.GetStringTensorDataLength();
    ASSERT_EQ(str_len, expected_output.length());
    std::unique_ptr<char[]> actual_result_string(new char[str_len + 1]);
    size_t offset = 0;
    str_tensor_value.GetStringTensorContent(actual_result_string.get(), str_len, &offset, 1);
    actual_result_string[str_len] = 0;
    ASSERT_EQ(expected_output.compare(actual_result_string.get()), 0);
  } catch (const std::exception& ex) {
    std::cerr << "Exception: " << ex.what() << std::endl;
    ASSERT_TRUE(false);
  }
}
}  // namespace test
}  // namespace onnxruntime
