// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"
#include <wil/result.h>
#include <D3d11_4.h>
#include <dxgi1_6.h>
#include "filehelpers.h"
#include <fstream>
#include <MemoryBuffer.h>
#include "CustomOperatorProvider.h"
#include "CustomOps.h"

// For custom operator and shape inferencing support
#include "core/providers/dml/DmlExecutionProvider/inc/MLOperatorAuthor.h"
#include "core/providers/dml/DmlExecutionProvider/src/ErrorHandling.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "core/providers/dml/OperatorAuthorHelper/OperatorHelper.h"
#include "core/providers/dml/OperatorAuthorHelper/OperatorVersions.h"
#include "core/graph/constants.h"
#include "CustomNullOp.h"
#include <wil/wrl.h>

using namespace winml;
using namespace wfc;
using namespace wm;
using namespace wgi;
using namespace ws;
using namespace wss;

static void CustomOpsScenarioTestsClassSetup() {
  winrt::init_apartment();
#ifdef BUILD_INBOX
  winrt_activation_handler = WINRT_RoGetActivationFactory;
#endif
}

// Tests that the execution provider correctly fuses operators together when custom ops are involved.
static void CustomOperatorFusion() {
  constexpr const wchar_t* c_modelFilename = L"squeezenet_tensor_input.onnx";

  // This particular model has 25 Conv ops and 25 Relu ops, all of which are eligible for fusion so we expect them
  // all to be fused (removing them from the graph) and replaced with the appropriate fused op instead. The same
  // goes for the single Gemm+Sigmoid in the model too.
  constexpr const uint32_t c_expectedConvOps = 0;
  constexpr const uint32_t c_expectedReluOps = 0;
  constexpr const uint32_t c_expectedFusedConvOps = 25;
  constexpr const uint32_t c_expectedGemmOps = 0;
  constexpr const uint32_t c_expectedSigmoidOps = 0;
  constexpr const uint32_t c_expectedFusedGemmOps = 1;

  // These ops are also part of the model but shouldn't be fused
  constexpr const uint32_t c_expectedBatchNormOps = 1;
  constexpr const uint32_t c_expectedMaxPoolOps = 3;
  constexpr const uint32_t c_expectedConcatOps = 8;

  struct CallbackOperatorProvider : winrt::implements<
                                      CallbackOperatorProvider,
                                      winml::ILearningModelOperatorProvider,
                                      ILearningModelOperatorProviderNative> {
    struct CallCounts {
      std::atomic<uint32_t> conv = 0;
      std::atomic<uint32_t> relu = 0;
      std::atomic<uint32_t> fusedConv = 0;
      std::atomic<uint32_t> gemm = 0;
      std::atomic<uint32_t> sigmoid = 0;
      std::atomic<uint32_t> fusedGemm = 0;
      std::atomic<uint32_t> batchNorm = 0;
      std::atomic<uint32_t> maxPool = 0;
      std::atomic<uint32_t> concat = 0;
    };

    const CallCounts& GetCallCounts() { return m_callCounts; }

    CallbackOperatorProvider() {
      using namespace OperatorHelper;

      std::wostringstream dll;
      dll << BINARY_NAME;
      auto winml_dll_name = dll.str();

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
      auto m_library = LoadLibraryExW(winml_dll_name.c_str(), nullptr, 0);
#elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_PC_APP)
      auto m_library = LoadPackagedLibrary(winml_dll_name.c_str(), 0 /*Reserved*/);
#endif
      using create_registry_delegate = HRESULT WINAPI(_COM_Outptr_ IMLOperatorRegistry * *registry);
      auto create_registry =
        reinterpret_cast<create_registry_delegate*>(GetProcAddress(m_library, "MLCreateOperatorRegistry"));
      WINML_EXPECT_HRESULT_SUCCEEDED(create_registry(m_registry.put()));

#pragma push_macro("REGISTER_KERNEL")
#define REGISTER_KERNEL(_name, _domain, _opSet, _shapeInferrer, _callCount) \
  NullOperatorFactory::RegisterKernel(                                      \
    #_name,                                                                 \
    (_domain),                                                              \
    _opSet::sc_sinceVer_##_name,                                            \
    m_registry,                                                             \
    winrt::make<NullShapeInferrer<_shapeInferrer>>(),                       \
    (_callCount)                                                            \
  );

      REGISTER_KERNEL(Conv, onnxruntime::kOnnxDomain, OnnxOperatorSet7, ConvHelper, &m_callCounts.conv);
      REGISTER_KERNEL(
        Relu, onnxruntime::kOnnxDomain, OnnxOperatorSet7, GetOutputShapeAsInputShapeHelper, &m_callCounts.relu
      );
      REGISTER_KERNEL(DmlFusedConv, onnxruntime::kMSDmlDomain, MsftOperatorSet1, ConvHelper, &m_callCounts.fusedConv);

      REGISTER_KERNEL(Gemm, onnxruntime::kOnnxDomain, OnnxOperatorSet7, GemmHelper, &m_callCounts.gemm);
      REGISTER_KERNEL(
        Sigmoid, onnxruntime::kOnnxDomain, OnnxOperatorSet7, GetOutputShapeAsInputShapeHelper, &m_callCounts.sigmoid
      );
      REGISTER_KERNEL(DmlFusedGemm, onnxruntime::kMSDmlDomain, MsftOperatorSet1, GemmHelper, &m_callCounts.fusedGemm);

      REGISTER_KERNEL(
        BatchNormalization,
        onnxruntime::kOnnxDomain,
        OnnxOperatorSet7,
        GetOutputShapeAsInputShapeHelper,
        &m_callCounts.batchNorm
      );
      REGISTER_KERNEL(MaxPool, onnxruntime::kOnnxDomain, OnnxOperatorSet7, PoolingHelper, &m_callCounts.maxPool);
      REGISTER_KERNEL(Concat, onnxruntime::kOnnxDomain, OnnxOperatorSet7, ConcatHelper, &m_callCounts.concat);

#pragma pop_macro("REGISTER_KERNEL")
    }

    STDMETHOD(GetRegistry)
    (IMLOperatorRegistry** ppOperatorRegistry) {
      if (ppOperatorRegistry == nullptr) {
        return E_POINTER;
      }

      m_registry.copy_to(ppOperatorRegistry);
      return S_OK;
    }

   private:
    winrt::com_ptr<IMLOperatorRegistry> m_registry;
    CallCounts m_callCounts;
  };

  auto customOperatorProvider = winrt::make<CallbackOperatorProvider>();
  auto provider = customOperatorProvider.as<ILearningModelOperatorProvider>();

  LearningModelDevice device = nullptr;
  WINML_EXPECT_NO_THROW(device = LearningModelDevice(LearningModelDeviceKind::DirectX));
  std::wstring fullPath = FileHelpers::GetModulePath() + c_modelFilename;
  auto model = LearningModel::LoadFromFilePath(fullPath, provider);

  auto featureValue = FileHelpers::LoadImageFeatureValue(L"227x227.png");

  LearningModelSession session = nullptr;
  WINML_EXPECT_NO_THROW(session = LearningModelSession(model, device));
  LearningModelBinding modelBinding(session);

  modelBinding.Bind(L"data", featureValue);
  auto result = session.Evaluate(modelBinding, L"");

  const auto& callCounts = customOperatorProvider.as<CallbackOperatorProvider>()->GetCallCounts();

  // Verify that the correct number of each operator was seen (i.e. that none were dropped / incorrectly fused)
  WINML_EXPECT_EQUAL(c_expectedConvOps, callCounts.conv);
  WINML_EXPECT_EQUAL(c_expectedReluOps, callCounts.relu);
  WINML_EXPECT_EQUAL(c_expectedFusedConvOps, callCounts.fusedConv);
  WINML_EXPECT_EQUAL(c_expectedGemmOps, callCounts.gemm);
  WINML_EXPECT_EQUAL(c_expectedSigmoidOps, callCounts.sigmoid);
  WINML_EXPECT_EQUAL(c_expectedFusedGemmOps, callCounts.fusedGemm);
  WINML_EXPECT_EQUAL(c_expectedBatchNormOps, callCounts.batchNorm);
  WINML_EXPECT_EQUAL(c_expectedMaxPoolOps, callCounts.maxPool);
  WINML_EXPECT_EQUAL(c_expectedConcatOps, callCounts.concat);
}

struct LocalCustomOperatorProvider : winrt::implements<
                                       LocalCustomOperatorProvider,
                                       winml::ILearningModelOperatorProvider,
                                       ILearningModelOperatorProviderNative> {
  LocalCustomOperatorProvider() {
    std::wostringstream dll;
    dll << BINARY_NAME;
    auto winml_dll_name = dll.str();

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
    auto m_library = LoadLibraryExW(winml_dll_name.c_str(), nullptr, 0);
#elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_PC_APP)
    auto m_library = LoadPackagedLibrary(winml_dll_name.c_str(), 0 /*Reserved*/);
#endif
    using create_registry_delegate = HRESULT WINAPI(_COM_Outptr_ IMLOperatorRegistry * *registry);
    auto create_registry =
      reinterpret_cast<create_registry_delegate*>(GetProcAddress(m_library, "MLCreateOperatorRegistry"));
    WINML_EXPECT_HRESULT_SUCCEEDED(create_registry(m_registry.put()));
  }

  STDMETHOD(GetRegistry)
  (IMLOperatorRegistry** ppOperatorRegistry) {
    if (ppOperatorRegistry == nullptr) {
      return E_POINTER;
    }

    m_registry.copy_to(ppOperatorRegistry);
    return S_OK;
  }

  IMLOperatorRegistry* GetRegistry() { return m_registry.get(); }

 protected:
  winrt::com_ptr<IMLOperatorRegistry> m_registry;
};

// Checks test attributes set on ABI kernels can be queried with correct values
void VerifyTestAttributes(const MLOperatorAttributes& attrs) {
  std::string strAttr = attrs.GetAttribute("DefaultedNonRequiredString");
  WINML_EXPECT_EQUAL(strAttr, "1");

  std::vector<std::string> strArrayAttr = attrs.GetAttributeVector("DefaultedNonRequiredStringArray");
  std::vector<std::string> expected = std::vector<std::string>({"1", "2"});
  for (size_t i = 0; i < expected.size(); ++i) {
    WINML_EXPECT_EQUAL(strArrayAttr[i], expected[i]);
  }

  WINML_EXPECT_EQUAL(1, attrs.GetAttribute<int64_t>("DefaultedNonRequiredInt"));
  WINML_EXPECT_EQUAL(1.0f, attrs.GetAttribute<float>("DefaultedNonRequiredFloat"));

  WINML_EXPECT_EQUAL(std::vector<int64_t>({1, 2}), attrs.GetAttributeVector<int64_t>("DefaultedNonRequiredIntArray"));
  WINML_EXPECT_EQUAL(
    std::vector<float>({1.0f, 2.0f}), attrs.GetAttributeVector<float>("DefaultedNonRequiredFloatArray")
  );
}

// Foo kernel which is doing Add and optionally truncates its output
template <typename T, bool VerifyAttributes = false, bool Truncate = false>
class FooKernel {
 public:
  FooKernel(const MLOperatorKernelCreationContext& info) {
    if (VerifyAttributes) {
      VerifyTestAttributes(info);
    }

    VerifyShapeInfo(info);
  }

  void VerifyShapeInfo(const MLOperatorKernelCreationContext& info) {
    if (!Truncate) {
      winrt::com_ptr<IMLOperatorTensorShapeDescription> shapeInfo;
      WINML_EXPECT_EQUAL(info.GetInterface()->HasTensorShapeDescription(), false);
      WINML_EXPECT_HRESULT_FAILED(info.GetInterface()->GetTensorShapeDescription(shapeInfo.put()));
    } else {
      winrt::com_ptr<IMLOperatorTensorShapeDescription> shapeInfo;
      WINML_EXPECT_EQUAL(info.GetInterface()->HasTensorShapeDescription(), true);
      WINML_EXPECT_EQUAL(info.GetInterface()->GetTensorShapeDescription(shapeInfo.put()), S_OK);
    }
  }

  void Compute(const MLOperatorKernelContext& context) const {
    const auto X = context.GetInputTensor(0);
    const auto W = context.GetInputTensor(1);

    auto xData = X.GetData<T>();
    auto wData = W.GetData<T>();

    auto shape = X.GetShape();

    // This is used to test shape inference
    if (Truncate) {
      shape[0] -= 1;
    }

    if (!Truncate) {
      winrt::com_ptr<IMLOperatorTensor> tensor;
      WINML_EXPECT_HRESULT_FAILED(context.GetInterface()->GetOutputTensor(0, tensor.put()));
    } else {
      MLOperatorTensor tensor = context.GetOutputTensor(0);
    }

    auto Y = context.GetOutputTensor(0, shape);
    auto yData = Y.GetData<T>();

    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }

    for (size_t i = 0; i < size; i++) {
      yData[i] = xData[i] + wData[i];
    }
  }
};

template <bool VerifyTestAttributes = false>
void CALLBACK CreateABIFooKernel(IMLOperatorKernelCreationContext* kernelInfo, IMLOperatorKernel** opKernel) {
  HRESULT hr = MLOperatorKernel<FooKernel<float, VerifyTestAttributes>>::CreateInstance(*kernelInfo, opKernel);
  THROW_IF_FAILED(hr);
}

void CALLBACK CreateTruncatedABIFooKernel(IMLOperatorKernelCreationContext* kernelInfo, IMLOperatorKernel** opKernel) {
  HRESULT hr = MLOperatorKernel<FooKernel<float, true, true>>::CreateInstance(*kernelInfo, opKernel);
  THROW_IF_FAILED(hr);
}

// Test using a foo kernel which is doing Add, but register it as "Mul".
static void CustomKernelWithBuiltInSchema() {
  // Create the registry
  auto operatorProvider = winrt::make<LocalCustomOperatorProvider>();
  IMLOperatorRegistry* registry = operatorProvider.as<LocalCustomOperatorProvider>()->GetRegistry();

  // Register the kernel
  MLOperatorEdgeDescription floatTensorType = {
    MLOperatorEdgeType::Tensor, static_cast<uint64_t>(MLOperatorTensorDataType::Float)
  };

  MLOperatorEdgeTypeConstrant constraint = {"T", &floatTensorType, 1};

  MLOperatorKernelDescription kernelDesc = {
    "",
    "Mul",
    7,
    MLOperatorExecutionType::Cpu,
    &constraint,
    1,
    nullptr,
    0,
    MLOperatorKernelOptions::AllowDynamicInputShapes
  };

  Microsoft::WRL::ComPtr<MLOperatorKernelFactory> factory =
    wil::MakeOrThrow<MLOperatorKernelFactory>(CreateABIFooKernel<false>);
  WINML_EXPECT_HRESULT_SUCCEEDED(registry->RegisterOperatorKernel(&kernelDesc, factory.Get(), nullptr));

  // Prepare inputs
  std::vector<int64_t> dimsX = {3, 2};
  std::vector<float> valuesX = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Prepare expected inputs and outputs
  std::vector<int64_t> expectedDimsY = {3, 2};

  // The expected value should be Add's result.
  std::vector<float> expectedValuesY = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  // Create the model and sessions
  std::wstring fullPath = FileHelpers::GetModulePath() + L"mul.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(fullPath, operatorProvider);

  LearningModelSession session(model);
  LearningModelBinding bindings(session);

  // Bind inputs and outputs
  TensorFloat inputTensor = TensorFloat::CreateFromArray(dimsX, winrt::array_view<const float>(std::move(valuesX)));
  bindings.Bind(winrt::hstring(L"X"), inputTensor);

  auto outputValue = TensorFloat::Create();
  WINML_EXPECT_NO_THROW(bindings.Bind(L"Y", outputValue));

  // Evaluate the model
  winrt::hstring correlationId;
  WINML_EXPECT_NO_THROW(session.Evaluate(bindings, correlationId));

  // Check the result shape
  WINML_EXPECT_EQUAL(expectedDimsY.size(), outputValue.Shape().Size());
  for (uint32_t j = 0; j < outputValue.Shape().Size(); j++) {
    WINML_EXPECT_EQUAL(expectedDimsY.at(j), outputValue.Shape().GetAt(j));
  }

  // Check the results
  auto buffer = outputValue.GetAsVectorView();
  WINML_EXPECT_TRUE(buffer != nullptr);
  WINML_EXPECT_TRUE(std::equal(expectedValuesY.cbegin(), expectedValuesY.cend(), begin(buffer)));

  // Release the model before operatorProvider goes out of scope
  model = nullptr;
}

// Similar to MLOperatorShapeInferrer, but using an std::function
class MLOperatorShapeInferrerFromFunc
  : public Microsoft::WRL::
      RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IMLOperatorShapeInferrer> {
 public:
  MLOperatorShapeInferrerFromFunc(std::function<void(IMLOperatorShapeInferenceContext*)> shapeInferenceFn)
    : m_func(shapeInferenceFn) {}

  HRESULT STDMETHODCALLTYPE InferOutputShapes(IMLOperatorShapeInferenceContext* context) noexcept override try {
    m_func(context);
    return S_OK;
  }
  CATCH_RETURN();

 private:
  std::function<void(IMLOperatorShapeInferenceContext*)> m_func;
};

// Test using a custom kernel and schema, while verifying attribute defaults, type mapping, and inference methods
static void CustomKernelWithCustomSchema() {
  // Test cases
  struct {
    // Whether the Foo kernel should truncate its output
    bool truncateOutput;

    // Whether a type label is used in the schema, versus a type description
    bool useTypeLabel;

    // Whether the schema provides a type inference function, and uses an output type
    // of Int32 instead of Float32
    bool useTypeInference;

    // Whether a shape inference method is provided in the schema
    bool useShapeInferenceInSchema;

    // Whether a shape inference method is provided in the kernel
    bool useShapeInferenceInKernel;

    // Whether attribute defaults are provided in the schema, instead of the kernel
    bool attributeDefaultsInSchema;
  } testCases[] = {
    {false,  true, false, false, false, false},
    {false, false, false, false, false, false},
    {false,  true,  true, false, false,  true},
    { true, false, false, false,  true, false},
    { true,  true,  true,  true,  true,  true},
  };

  for (size_t caseIndex = 0; caseIndex < std::size(testCases); ++caseIndex) {
    // Create the registry
    auto operatorProvider = winrt::make<LocalCustomOperatorProvider>();
    IMLOperatorRegistry* registry = operatorProvider.as<LocalCustomOperatorProvider>()->GetRegistry();

    // Create input and output parameters
    MLOperatorSchemaEdgeDescription inputParam = {};
    inputParam.options = MLOperatorParameterOptions::Single;

    if (!testCases[caseIndex].useTypeLabel) {
      assert(!testCases[caseIndex].useTypeInference);

      MLOperatorEdgeDescription edgeDesc = {};
      edgeDesc.edgeType = MLOperatorEdgeType::Tensor;
      edgeDesc.tensorDataType = MLOperatorTensorDataType::Float;

      inputParam.typeFormat = MLOperatorSchemaEdgeTypeFormat::EdgeDescription;
      inputParam.edgeDescription = edgeDesc;
    } else {
      inputParam.typeFormat = MLOperatorSchemaEdgeTypeFormat::Label;
      inputParam.typeLabel = "T1";
    }

    MLOperatorSchemaEdgeDescription outputParam = inputParam;

    // Type inference should set this to tensor(float) even though T2 is not matched
    // on an input label
    if (testCases[caseIndex].useTypeInference) {
      if (inputParam.typeFormat == MLOperatorSchemaEdgeTypeFormat::Label) {
        outputParam.typeLabel = "T2";
      } else {
        outputParam.edgeDescription.tensorDataType = MLOperatorTensorDataType::Int32;
      }
    }

    MLOperatorSchemaEdgeDescription inputs[] = {inputParam, inputParam};

    MLOperatorEdgeDescription edgeTypes[6] = {
      {MLOperatorEdgeType::Tensor, static_cast<uint64_t>(MLOperatorTensorDataType::UInt32)},
      {MLOperatorEdgeType::Tensor, static_cast<uint64_t>(MLOperatorTensorDataType::UInt64)},
      {MLOperatorEdgeType::Tensor,  static_cast<uint64_t>(MLOperatorTensorDataType::Int32)},
      {MLOperatorEdgeType::Tensor,  static_cast<uint64_t>(MLOperatorTensorDataType::Int64)},
      {MLOperatorEdgeType::Tensor,  static_cast<uint64_t>(MLOperatorTensorDataType::Float)},
      {MLOperatorEdgeType::Tensor, static_cast<uint64_t>(MLOperatorTensorDataType::Double)}
    };

    // Type constraints.  Only the first is used unless type inference is provided and
    // the kernel emits a different output type as "T2"
    MLOperatorEdgeTypeConstrant constraints[] = {
      {"T1", edgeTypes, static_cast<uint32_t>(std::size(edgeTypes))},
      {"T2", edgeTypes, static_cast<uint32_t>(std::size(edgeTypes))}
    };

    // Test attributes
    MLOperatorAttribute attributes[] = {
      {           "DefaultedNonRequiredInt",         MLOperatorAttributeType::Int, false},
      {         "DefaultedNonRequiredFloat",       MLOperatorAttributeType::Float, false},
      {        "DefaultedNonRequiredString",      MLOperatorAttributeType::String, false},
      {      "DefaultedNonRequiredIntArray",    MLOperatorAttributeType::IntArray, false},
      {    "DefaultedNonRequiredFloatArray",  MLOperatorAttributeType::FloatArray, false},
      {   "DefaultedNonRequiredStringArray", MLOperatorAttributeType::StringArray, false},

      {"NonDefaultedNonRequiredStringArray", MLOperatorAttributeType::StringArray, false},
    };

    // Defaults.  These are queried back during kernel creation, type and shape inference
    // and tested against the same values
    MLOperatorAttributeNameValue defaultAttributes[] = {
      {        "DefaultedNonRequiredInt",         MLOperatorAttributeType::Int, 1},
      {      "DefaultedNonRequiredFloat",       MLOperatorAttributeType::Float, 1},
      {     "DefaultedNonRequiredString",      MLOperatorAttributeType::String, 1},
      {   "DefaultedNonRequiredIntArray",    MLOperatorAttributeType::IntArray, 2},
      { "DefaultedNonRequiredFloatArray",  MLOperatorAttributeType::FloatArray, 2},
      {"DefaultedNonRequiredStringArray", MLOperatorAttributeType::StringArray, 2},
    };

    int64_t defaultInts[] = {1, 2};
    float defaultFloats[] = {1.0f, 2.0f};
    const char* defaultStrings[] = {"1", "2"};
    defaultAttributes[0].ints = defaultInts;
    defaultAttributes[1].floats = defaultFloats;
    defaultAttributes[2].strings = defaultStrings;
    defaultAttributes[3].ints = defaultInts;
    defaultAttributes[4].floats = defaultFloats;
    defaultAttributes[5].strings = defaultStrings;

    // Schema definition
    MLOperatorSchemaDescription schemaDesc = {};
    schemaDesc.name = "Foo";
    schemaDesc.operatorSetVersionAtLastChange = 7;
    schemaDesc.inputs = inputs;
    schemaDesc.inputCount = 2;
    schemaDesc.outputs = &outputParam;
    schemaDesc.outputCount = 1;
    schemaDesc.typeConstraints = constraints;
    schemaDesc.typeConstraintCount = testCases[caseIndex].useTypeLabel ? 2 : 0;
    schemaDesc.attributes = attributes;
    schemaDesc.attributeCount = static_cast<uint32_t>(std::size(attributes));

    if (testCases[caseIndex].attributeDefaultsInSchema) {
      schemaDesc.defaultAttributes = defaultAttributes;
      schemaDesc.defaultAttributeCount = static_cast<uint32_t>(std::size(defaultAttributes));
    }

    Microsoft::WRL::ComPtr<MLOperatorTypeInferrer> typeInferrer;
    Microsoft::WRL::ComPtr<MLOperatorShapeInferrerFromFunc> shapeInferrer;

    // Type inference function
    if (testCases[caseIndex].useTypeInference) {
      typeInferrer = wil::MakeOrThrow<MLOperatorTypeInferrer>([](IMLOperatorTypeInferenceContext* ctx) -> void {
        VerifyTestAttributes(MLOperatorTypeInferenceContext(ctx));

        MLOperatorEdgeDescription edgeDesc = {};
        edgeDesc.edgeType = MLOperatorEdgeType::Tensor;
        edgeDesc.tensorDataType = MLOperatorTensorDataType::Float;

        MLOperatorTypeInferenceContext(ctx).SetOutputEdgeDescription(0, &edgeDesc);
      });
    }

    // Store the shape inference context with a reference following the call to InferOutputShapes.
    // This will be called after loading the model as an isolated test for how ABI context objects
    // are "closed."
    Microsoft::WRL::ComPtr<IMLOperatorShapeInferenceContext> shapeInferenceContext;

    // Shape inference is tested by truncating the output size
    bool truncateOutput = testCases[caseIndex].truncateOutput;
    if (truncateOutput) {
      shapeInferrer = wil::MakeOrThrow<MLOperatorShapeInferrerFromFunc>(
        [&shapeInferenceContext](IMLOperatorShapeInferenceContext* ctx) -> void {
          VerifyTestAttributes(MLShapeInferenceContext(ctx));
          MLShapeInferenceContext(ctx).SetOutputTensorShape(0, {2, 2});
          shapeInferenceContext = ctx;
        }
      );
    }

    // Register the schema
    MLOperatorSetId opsetId = {"", 7};
    MLOperatorSchemaDescription* opSchemaDescs = &schemaDesc;
    WINML_EXPECT_EQUAL(
      S_OK,
      registry->RegisterOperatorSetSchema(
        &opsetId,
        1,
        &opSchemaDescs,
        1,
        typeInferrer.Get(),
        testCases[caseIndex].useShapeInferenceInSchema ? shapeInferrer.Get() : nullptr
      )
    );

    {
      // Register a future version of the schema in the same domain, while setting its
      // input count to zero to ensure it is not being used.
      auto futureSchemaDesc = schemaDesc;
      futureSchemaDesc.inputCount = 0;

      MLOperatorSetId id = {"", 9};
      MLOperatorSchemaDescription* schemaDescs = &futureSchemaDesc;
      WINML_EXPECT_EQUAL(
        S_OK,
        registry->RegisterOperatorSetSchema(
          &id,
          7,
          &schemaDescs,
          1,
          typeInferrer.Get(),
          testCases[caseIndex].useShapeInferenceInSchema ? shapeInferrer.Get() : nullptr
        )
      );
    }
    {
      // Register in another (unused) domain to the custom registry
      auto otherSchemaDesc = schemaDesc;
      otherSchemaDesc.inputCount = 0;

      MLOperatorSetId id = {"otherDomain", 7};
      MLOperatorSchemaDescription* schemaDescs = &otherSchemaDesc;
      WINML_EXPECT_EQUAL(
        S_OK,
        registry->RegisterOperatorSetSchema(
          &id,
          1,
          &schemaDescs,
          1,
          typeInferrer.Get(),
          testCases[caseIndex].useShapeInferenceInSchema ? shapeInferrer.Get() : nullptr
        )
      );
    }
    // Register the Foo kernel
    MLOperatorEdgeDescription floatTensorEdgeDesc = {};
    floatTensorEdgeDesc.edgeType = MLOperatorEdgeType::Tensor;
    floatTensorEdgeDesc.tensorDataType = MLOperatorTensorDataType::Float;

    MLOperatorEdgeTypeConstrant kernelConstraint = {"T1", &floatTensorEdgeDesc, 1};

    MLOperatorKernelDescription kernelDesc = {
      "", "Foo", 7, MLOperatorExecutionType::Cpu, &kernelConstraint, testCases[caseIndex].useTypeLabel ? 1u : 0u
    };

    if (!testCases[caseIndex].attributeDefaultsInSchema) {
      kernelDesc.defaultAttributes = defaultAttributes;
      kernelDesc.defaultAttributeCount = static_cast<uint32_t>(std::size(defaultAttributes));
    }

    if (!truncateOutput) {
      kernelDesc.options = MLOperatorKernelOptions::AllowDynamicInputShapes;
      Microsoft::WRL::ComPtr<MLOperatorKernelFactory> factory =
        wil::MakeOrThrow<MLOperatorKernelFactory>(CreateABIFooKernel<true>);

      WINML_EXPECT_EQUAL(S_OK, registry->RegisterOperatorKernel(&kernelDesc, factory.Get(), nullptr));
    } else {
      Microsoft::WRL::ComPtr<MLOperatorKernelFactory> factory =
        wil::MakeOrThrow<MLOperatorKernelFactory>(CreateTruncatedABIFooKernel);
      WINML_EXPECT_EQUAL(
        S_OK,
        registry->RegisterOperatorKernel(
          &kernelDesc, factory.Get(), testCases[caseIndex].useShapeInferenceInKernel ? shapeInferrer.Get() : nullptr
        )
      );
    }

    // Prepare inputs
    std::vector<int64_t> dimsX = {3, 2};
    std::vector<float> valuesX = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // Prepare expected inputs and outputs
    std::vector<int64_t> expectedDimsY = {truncateOutput ? 2 : 3, 2};
    // now the expected value should be Add's result.
    std::vector<float> expectedValuesY = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    if (truncateOutput) {
      // The leading dimension is truncated, and the second dimension has two elements over that dim
      expectedValuesY.resize(expectedValuesY.size() - 2);
    }

    // Load the model and sessions
    std::wstring fullPath = FileHelpers::GetModulePath() + (truncateOutput ? L"foo_truncated.onnx" : L"foo.onnx");
    LearningModel model = LearningModel::LoadFromFilePath(fullPath, operatorProvider);
    LearningModelSession session(model);

    // Bind input and outputs
    LearningModelBinding bindings(session);

    TensorFloat inputTensor = TensorFloat::CreateFromArray(dimsX, winrt::array_view<const float>(std::move(valuesX)));
    bindings.Bind(winrt::hstring(L"X"), inputTensor);

    auto outputValue = TensorFloat::Create();
    WINML_EXPECT_NO_THROW(bindings.Bind(L"Y", outputValue));

    // Evaluate the model
    winrt::hstring correlationId;
    WINML_EXPECT_NO_THROW(session.Evaluate(bindings, correlationId));

    // Verify the result shape
    WINML_EXPECT_EQUAL(expectedDimsY.size(), outputValue.Shape().Size());
    for (uint32_t j = 0; j < outputValue.Shape().Size(); j++) {
      WINML_EXPECT_EQUAL(expectedDimsY.at(j), outputValue.Shape().GetAt(j));
    }

    // Verify the result values
    auto buffer = outputValue.GetAsVectorView();
    WINML_EXPECT_TRUE(buffer != nullptr);
    WINML_EXPECT_TRUE(std::equal(expectedValuesY.cbegin(), expectedValuesY.cend(), begin(buffer)));

    // Release the model before operatorProvider goes out of scope
    model = nullptr;

    if (shapeInferenceContext) {
      // Check that the shape inference context is closed and safely fails
      MLOperatorEdgeDescription edgeDesc;
      WINML_EXPECT_EQUAL(E_INVALIDARG, shapeInferenceContext->GetInputEdgeDescription(0, &edgeDesc));
    }
  }
}

const CustomOpsTestsApi& getapi() {
  static CustomOpsTestsApi api = {
    CustomOpsScenarioTestsClassSetup, CustomOperatorFusion, CustomKernelWithBuiltInSchema, CustomKernelWithCustomSchema
  };

  if (SkipGpuTests()) {
    api.CustomOperatorFusion = SkipTest;
  }
  if (RuntimeParameterExists(L"noVideoFrameTests")) {
    api.CustomOperatorFusion = SkipTest;
  }
  return api;
}
