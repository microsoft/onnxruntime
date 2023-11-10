// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"

#include "APITest.h"
#include "LearningModelBindingAPITest.h"
#include "SqueezeNetValidator.h"

#include <sstream>

using namespace winrt;
using namespace winml;
using namespace wfc;
using namespace wgi;
using namespace wm;
using namespace ws;

static void LearningModelBindingAPITestsClassSetup() {
  init_apartment();
#ifdef BUILD_INBOX
  winrt_activation_handler = WINRT_RoGetActivationFactory;
#endif
}

static void CpuSqueezeNet() {
  std::string cpuInstance("CPU");
  WINML_EXPECT_NO_THROW(WinML::Engine::Test::ModelValidator::SqueezeNet(
    cpuInstance,
    LearningModelDeviceKind::Cpu,
    /*dataTolerance*/ 0.00001f,
    false
  ));
}

static void CpuSqueezeNetEmptyOutputs() {
  std::string cpuInstance("CPU");
  WINML_EXPECT_NO_THROW(WinML::Engine::Test::ModelValidator::SqueezeNet(
                          cpuInstance,
                          LearningModelDeviceKind::Cpu,
                          /*dataTolerance*/ 0.00001f,
                          false,
                          OutputBindingStrategy::Empty
  ););
}

static void CpuSqueezeNetUnboundOutputs() {
  std::string cpuInstance("CPU");
  WINML_EXPECT_NO_THROW(WinML::Engine::Test::ModelValidator::SqueezeNet(
                          cpuInstance,
                          LearningModelDeviceKind::Cpu,
                          /*dataTolerance*/ 0.00001f,
                          false,
                          OutputBindingStrategy::Unbound
  ););
}

static void CpuSqueezeNetBindInputTensorAsInspectable() {
  std::string cpuInstance("CPU");
  WINML_EXPECT_NO_THROW(WinML::Engine::Test::ModelValidator::SqueezeNet(
                          cpuInstance,
                          LearningModelDeviceKind::Cpu,
                          /*dataTolerance*/ 0.00001f,
                          false,
                          OutputBindingStrategy::Bound /* empty outputs */,
                          true /* bind inputs as inspectables */
  ););
}

static void CastMapInt64() {
  WINML_EXPECT_NO_THROW(LearningModel::LoadFromFilePath(FileHelpers::GetModulePath() + L"castmap-int64.onnx"));
    // TODO: Check Descriptor
}

static void DictionaryVectorizerMapInt64() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"dictvectorizer-int64.onnx", learningModel));

  auto inputDescriptor = learningModel.InputFeatures().First().Current();
  WINML_EXPECT_TRUE(inputDescriptor.Kind() == LearningModelFeatureKind::Map);
  auto mapDescriptor = inputDescriptor.as<MapFeatureDescriptor>();
  WINML_EXPECT_TRUE(mapDescriptor.KeyKind() == TensorKind::Int64);
  WINML_EXPECT_TRUE(mapDescriptor.ValueDescriptor().Kind() == LearningModelFeatureKind::Tensor);
  auto tensorDescriptor = mapDescriptor.ValueDescriptor().as<TensorFeatureDescriptor>();
    // empty size means tensor of scalar value
  WINML_EXPECT_TRUE(tensorDescriptor.Shape().Size() == 0);
  WINML_EXPECT_TRUE(tensorDescriptor.TensorKind() == TensorKind::Float);

  LearningModelSession modelSession(learningModel);
  LearningModelBinding binding(modelSession);
  std::unordered_map<int64_t, float> map;
  map[1] = 1.f;
  map[10] = 10.f;
  map[3] = 3.f;

  auto mapInputName = inputDescriptor.Name();

    // Bind as IMap
  auto abiMap = winrt::single_threaded_map(std::move(map));
  binding.Bind(mapInputName, abiMap);
  auto mapInputInspectable = abiMap.as<wf::IInspectable>();
  auto first = binding.First();
  WINML_EXPECT_TRUE(first.Current().Key() == mapInputName);
  WINML_EXPECT_TRUE(first.Current().Value() == mapInputInspectable);
  WINML_EXPECT_TRUE(binding.Lookup(mapInputName) == mapInputInspectable);

    // Bind as IMapView
  auto mapView = abiMap.GetView();
  binding.Bind(mapInputName, mapView);
  mapInputInspectable = mapView.as<wf::IInspectable>();
  first = binding.First();
  WINML_EXPECT_TRUE(first.Current().Key() == mapInputName);
  WINML_EXPECT_TRUE(first.Current().Value() == mapView);
  WINML_EXPECT_TRUE(binding.Lookup(mapInputName) == mapView);
}

static void DictionaryVectorizerMapString() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"dictvectorizer-string.onnx", learningModel));

  auto inputDescriptor = learningModel.InputFeatures().First().Current();
  WINML_EXPECT_TRUE(inputDescriptor.Kind() == LearningModelFeatureKind::Map);

  auto mapDescriptor = inputDescriptor.as<MapFeatureDescriptor>();
  WINML_EXPECT_TRUE(mapDescriptor.KeyKind() == TensorKind::String);
  WINML_EXPECT_TRUE(mapDescriptor.ValueDescriptor().Kind() == LearningModelFeatureKind::Tensor);

  auto tensorDescriptor = mapDescriptor.ValueDescriptor().as<TensorFeatureDescriptor>();
    // empty size means tensor of scalar value
  WINML_EXPECT_TRUE(tensorDescriptor.Shape().Size() == 0);
  WINML_EXPECT_TRUE(tensorDescriptor.TensorKind() == TensorKind::Float);

  LearningModelSession modelSession(learningModel);
  LearningModelBinding binding(modelSession);
  std::unordered_map<winrt::hstring, float> map;
  map[L"1"] = 1.f;
  map[L"10"] = 10.f;
  map[L"2"] = 2.f;

  auto mapInputName = inputDescriptor.Name();
  auto abiMap = winrt::single_threaded_map(std::move(map));
  binding.Bind(mapInputName, abiMap);

  auto mapInputInspectable = abiMap.as<wf::IInspectable>();
  auto first = binding.First();
  WINML_EXPECT_TRUE(first.Current().Key() == mapInputName);
  WINML_EXPECT_TRUE(first.Current().Value() == mapInputInspectable);
  WINML_EXPECT_TRUE(binding.Lookup(mapInputName) == mapInputInspectable);

  modelSession.Evaluate(binding, L"");
}

static void RunZipMapInt64(winml::LearningModel model, OutputBindingStrategy bindingStrategy) {
  auto outputFeatures = model.OutputFeatures();
  auto outputDescriptor = outputFeatures.First().Current();
  WINML_EXPECT_TRUE(outputDescriptor.Kind() == LearningModelFeatureKind::Sequence);

  auto seqDescriptor = outputDescriptor.as<SequenceFeatureDescriptor>();
  auto mapDescriptor = seqDescriptor.ElementDescriptor().as<MapFeatureDescriptor>();
  WINML_EXPECT_TRUE(mapDescriptor.KeyKind() == TensorKind::Int64);

  WINML_EXPECT_TRUE(mapDescriptor.ValueDescriptor().Kind() == LearningModelFeatureKind::Tensor);
  auto tensorDescriptor = mapDescriptor.ValueDescriptor().as<TensorFeatureDescriptor>();
  WINML_EXPECT_TRUE(tensorDescriptor.TensorKind() == TensorKind::Float);

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  std::vector<float> inputs = {0.5f, 0.25f, 0.125f};
  std::vector<int64_t> shape = {1, 3};

    // Bind inputs
  auto inputTensor = TensorFloat::CreateFromArray(shape, winrt::array_view<const float>(std::move(inputs)));
  binding.Bind(winrt::hstring(L"X"), inputTensor);

  typedef IMap<int64_t, float> ABIMap;
  typedef IVector<ABIMap> ABISequeneceOfMap;

  ABISequeneceOfMap abiOutput = nullptr;
    // Bind outputs
  if (bindingStrategy == OutputBindingStrategy::Bound) {
    abiOutput = winrt::single_threaded_vector<ABIMap>();
    binding.Bind(winrt::hstring(L"Y"), abiOutput);
  }

    // Evaluate
  auto result = session.Evaluate(binding, L"0").Outputs();

  if (bindingStrategy == OutputBindingStrategy::Bound) {
        // from output binding
    const auto& out1 = abiOutput.GetAt(0);
    const auto& out2 = result.Lookup(L"Y").as<IVectorView<ABIMap>>().GetAt(0);
    WINML_LOG_COMMENT((std::ostringstream() << "size: " << out1.Size()).str());
        // check outputs
    auto iter1 = out1.First();
    auto iter2 = out2.First();
    for (uint32_t i = 0, size = (uint32_t)inputs.size(); i < size; ++i) {
      WINML_EXPECT_TRUE(iter1.HasCurrent());
      WINML_EXPECT_TRUE(iter2.HasCurrent());
      const auto& pair1 = iter1.Current();
      const auto& pair2 = iter2.Current();
      WINML_LOG_COMMENT((std::ostringstream() << "key: " << pair1.Key() << ", value: " << pair2.Value()).str());
      WINML_EXPECT_TRUE(pair1.Key() == i && pair2.Key() == i);
      WINML_EXPECT_TRUE(pair1.Value() == inputs[i] && pair2.Value() == inputs[i]);
      iter1.MoveNext();
      iter2.MoveNext();
    }
    WINML_EXPECT_TRUE(!iter1.HasCurrent());
    WINML_EXPECT_TRUE(!iter2.HasCurrent());
  } else {
    abiOutput = result.Lookup(L"Y").as<ABISequeneceOfMap>();
    WINML_EXPECT_TRUE(abiOutput.Size() == 1);
    ABIMap map = abiOutput.GetAt(0);
    WINML_EXPECT_TRUE(map.Size() == 3);
    WINML_EXPECT_TRUE(map.Lookup(0) == 0.5);
    WINML_EXPECT_TRUE(map.Lookup(1) == .25);
    WINML_EXPECT_TRUE(map.Lookup(2) == .125);
  }
}

static void ZipMapInt64() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"zipmap-int64.onnx", learningModel));
  RunZipMapInt64(learningModel, OutputBindingStrategy::Bound);
}

static void ZipMapInt64Unbound() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"zipmap-int64.onnx", learningModel));
  RunZipMapInt64(learningModel, OutputBindingStrategy::Unbound);
}

static void ZipMapString() {
    // output constraint: "seq(map(string, float))" or "seq(map(int64, float))"
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"zipmap-string.onnx", learningModel));
  auto outputs = learningModel.OutputFeatures();
  auto outputDescriptor = outputs.First().Current();
  WINML_EXPECT_TRUE(outputDescriptor.Kind() == LearningModelFeatureKind::Sequence);
  auto mapDescriptor = outputDescriptor.as<SequenceFeatureDescriptor>().ElementDescriptor().as<MapFeatureDescriptor>();
  WINML_EXPECT_TRUE(mapDescriptor.KeyKind() == TensorKind::String);
  WINML_EXPECT_TRUE(mapDescriptor.ValueDescriptor().Kind() == LearningModelFeatureKind::Tensor);
  auto tensorDescriptor = mapDescriptor.ValueDescriptor().as<TensorFeatureDescriptor>();
  WINML_EXPECT_TRUE(tensorDescriptor.TensorKind() == TensorKind::Float);

  LearningModelSession session(learningModel);
  LearningModelBinding binding(session);

  std::vector<float> inputs = {0.5f, 0.25f, 0.125f};
  std::vector<int64_t> shape = {1, 3};
  std::vector<winrt::hstring> labels = {L"cat", L"dog", L"lion"};
  std::map<winrt::hstring, float> mapData = {
    { L"cat", 0.0f},
    { L"dog", 0.0f},
    {L"lion", 0.0f}
  };
  typedef IMap<winrt::hstring, float> ABIMap;
  ABIMap abiMap = winrt::single_threaded_map<winrt::hstring, float>(std::move(mapData));
  std::vector<ABIMap> seqOutput = {abiMap};
  IVector<ABIMap> ABIOutput = winrt::single_threaded_vector<ABIMap>(std::move(seqOutput));

  TensorFloat inputTensor = TensorFloat::CreateFromArray(shape, winrt::array_view<const float>(std::move(inputs)));
  binding.Bind(winrt::hstring(L"X"), inputTensor);
  binding.Bind(winrt::hstring(L"Y"), ABIOutput);
  auto result = session.Evaluate(binding, L"0").Outputs();
    // from output binding
  const auto& out1 = ABIOutput.GetAt(0);
  const auto& out2 = result.Lookup(L"Y").as<IVectorView<ABIMap>>().GetAt(0);
  WINML_LOG_COMMENT((std::ostringstream() << "size: " << out1.Size()).str());
    // single key,value pair for each map
  auto iter1 = out1.First();
  auto iter2 = out2.First();
  for (uint32_t i = 0, size = (uint32_t)inputs.size(); i < size; ++i) {
    WINML_EXPECT_TRUE(iter2.HasCurrent());
    const auto& pair1 = iter1.Current();
    const auto& pair2 = iter2.Current();
    WINML_LOG_COMMENT((std::ostringstream() << "key: " << pair1.Key().c_str() << ", value " << pair2.Value()).str());
    WINML_EXPECT_TRUE(std::wstring(pair1.Key().c_str()).compare(labels[i]) == 0);
    WINML_EXPECT_TRUE(std::wstring(pair2.Key().c_str()).compare(labels[i]) == 0);
    WINML_EXPECT_TRUE(pair1.Value() == inputs[i] && pair2.Value() == inputs[i]);
    iter1.MoveNext();
    iter2.MoveNext();
  }
  WINML_EXPECT_TRUE(!iter1.HasCurrent());
  WINML_EXPECT_TRUE(!iter2.HasCurrent());
}

static void GpuSqueezeNet() {
  std::string gpuInstance("GPU");
  WINML_EXPECT_NO_THROW(WinML::Engine::Test::ModelValidator::SqueezeNet(
                          gpuInstance,
                          LearningModelDeviceKind::DirectX,
                          /*dataTolerance*/ 0.00001f
  ););
}

static void GpuSqueezeNetEmptyOutputs() {
  std::string gpuInstance("GPU");
  WINML_EXPECT_NO_THROW(WinML::Engine::Test::ModelValidator::SqueezeNet(
                          gpuInstance,
                          LearningModelDeviceKind::DirectX,
                          /*dataTolerance*/ 0.00001f,
                          false,
                          OutputBindingStrategy::Empty
  ););
}

static void GpuSqueezeNetUnboundOutputs() {
  std::string gpuInstance("GPU");
  WINML_EXPECT_NO_THROW(WinML::Engine::Test::ModelValidator::SqueezeNet(
                          gpuInstance,
                          LearningModelDeviceKind::DirectX,
                          /*dataTolerance*/ 0.00001f,
                          false,
                          OutputBindingStrategy::Unbound
  ););
}

// Validates that when the input image is the same as the model expects, the binding step is executed correctly.
static void ImageBindingDimensions() {
  LearningModelBinding learningModelBinding = nullptr;
  LearningModel learningModel = nullptr;
  LearningModelSession learningModelSession = nullptr;
  LearningModelDevice leraningModelDevice = nullptr;
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
    // load a model with expected input size: 224 x 224
  WINML_EXPECT_NO_THROW(leraningModelDevice = LearningModelDevice(LearningModelDeviceKind::Default));
  WINML_EXPECT_NO_THROW(learningModel = LearningModel::LoadFromFilePath(filePath));
  WINML_EXPECT_TRUE(learningModel != nullptr);
  WINML_EXPECT_NO_THROW(learningModelSession = LearningModelSession(learningModel, leraningModelDevice));
  WINML_EXPECT_NO_THROW(learningModelBinding = LearningModelBinding(learningModelSession));

    // Create input images and execute bind
    // Test Case 1: both width and height are larger than model expects
  VideoFrame inputImage1(BitmapPixelFormat::Rgba8, 1000, 1000);
  ImageFeatureValue inputTensor = ImageFeatureValue::CreateFromVideoFrame(inputImage1);
  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"data_0", inputTensor));

    // Test Case 2: only height is larger, while width is smaller
  VideoFrame inputImage2(BitmapPixelFormat::Rgba8, 20, 1000);
  inputTensor = ImageFeatureValue::CreateFromVideoFrame(inputImage2);
  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"data_0", inputTensor));

    // Test Case 3: only width is larger, while height is smaller
  VideoFrame inputImage3(BitmapPixelFormat::Rgba8, 1000, 20);
  inputTensor = ImageFeatureValue::CreateFromVideoFrame(inputImage3);
  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"data_0", inputTensor));

    // Test Case 4: both width and height are smaller than model expects
  VideoFrame inputImage4(BitmapPixelFormat::Rgba8, 20, 20);
  inputTensor = ImageFeatureValue::CreateFromVideoFrame(inputImage4);
  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"data_0", inputTensor));
}

static void VerifyInvalidBindExceptions() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"zipmap-int64.onnx", learningModel));

  LearningModelSession session(learningModel);
  LearningModelBinding binding(session);

  std::vector<float> inputs = {0.5f, 0.25f, 0.125f};
  std::vector<int64_t> shape = {1, 3};

  auto matchException = [](const winrt::hresult_error& e, HRESULT hr) -> bool { return e.code() == hr; };

  auto ensureWinmlSizeMismatch = std::bind(matchException, std::placeholders::_1, WINML_ERR_SIZE_MISMATCH);
  auto ensureWinmlInvalidBinding = std::bind(matchException, std::placeholders::_1, WINML_ERR_INVALID_BINDING);

    /*
        Verify tensor bindings throw correct bind exceptions
    */

    // Bind invalid image as tensorfloat input
  auto image = FileHelpers::LoadImageFeatureValue(L"227x227.png");
  WINML_EXPECT_THROW_SPECIFIC(binding.Bind(L"X", image), winrt::hresult_error, ensureWinmlSizeMismatch);

    // Bind invalid map as tensorfloat input
  std::unordered_map<float, float> map;
  auto abiMap = winrt::single_threaded_map(std::move(map));
  WINML_EXPECT_THROW_SPECIFIC(binding.Bind(L"X", abiMap), winrt::hresult_error, ensureWinmlInvalidBinding);

    // Bind invalid sequence as tensorfloat input
  std::vector<uint32_t> sequence;
  auto abiSequence = winrt::single_threaded_vector(std::move(sequence));
  WINML_EXPECT_THROW_SPECIFIC(binding.Bind(L"X", abiSequence), winrt::hresult_error, ensureWinmlInvalidBinding);

    // Bind invalid tensor size as tensorfloat input
  auto tensorBoolean = TensorBoolean::Create();
  WINML_EXPECT_THROW_SPECIFIC(binding.Bind(L"X", tensorBoolean), winrt::hresult_error, ensureWinmlInvalidBinding);

    // Bind invalid tensor shape as tensorfloat input
  auto tensorInvalidShape = TensorFloat::Create(std::vector<int64_t>{2, 3, 4});
  WINML_EXPECT_THROW_SPECIFIC(binding.Bind(L"X", tensorInvalidShape), winrt::hresult_error, ensureWinmlInvalidBinding);

    /*
        Verify sequence bindings throw correct bind exceptions
    */

    // Bind invalid image as sequence<map<int, float> output
  WINML_EXPECT_THROW_SPECIFIC(binding.Bind(L"Y", image), winrt::hresult_error, ensureWinmlInvalidBinding);

    // Bind invalid map as sequence<map<int, float> output
  WINML_EXPECT_THROW_SPECIFIC(binding.Bind(L"Y", abiMap), winrt::hresult_error, ensureWinmlInvalidBinding);

    // Bind invalid sequence<int> as sequence<map<int, float> output
  WINML_EXPECT_THROW_SPECIFIC(binding.Bind(L"Y", abiSequence), winrt::hresult_error, ensureWinmlInvalidBinding);

    // Bind invalid tensor as sequence<map<int, float> output
  WINML_EXPECT_THROW_SPECIFIC(binding.Bind(L"Y", tensorBoolean), winrt::hresult_error, ensureWinmlInvalidBinding);

    /*
        Verify image bindings throw correct bind exceptions
    */

    // WINML_EXPECT_NO_THROW(LoadModel(L"fns-candy.onnx"));

    // LearningModelSession imageSession(m_model);
    // LearningModelBinding imageBinding(imageSession);

    // auto inputName = m_model.InputFeatures().First().Current().Name();

    // // Bind invalid map as image input
  // WINML_EXPECT_THROW_SPECIFIC(imageBinding.Bind(inputName, abiMap), winrt::hresult_error, ensureWinmlInvalidBinding);

  // // Bind invalid sequence as image input
  // WINML_EXPECT_THROW_SPECIFIC(imageBinding.Bind(inputName, abiSequence), winrt::hresult_error, ensureWinmlInvalidBinding);

  // // Bind invalid tensor type as image input
  // WINML_EXPECT_THROW_SPECIFIC(imageBinding.Bind(inputName, tensorBoolean), winrt::hresult_error, ensureWinmlInvalidBinding);

  // // Bind invalid tensor size as image input
  // auto tensorFloat = TensorFloat::Create(std::vector<int64_t> { 1, 1, 100, 100 });
  // WINML_EXPECT_THROW_SPECIFIC(imageBinding.Bind(inputName, tensorFloat), winrt::hresult_error, ensureWinmlInvalidBinding);

  // // Bind invalid tensor shape as image input
  // WINML_EXPECT_THROW_SPECIFIC(imageBinding.Bind(inputName, tensorInvalidShape), winrt::hresult_error, ensureWinmlInvalidBinding);

  /*
        Verify map bindings throw correct bind exceptions
    */
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"dictvectorizer-int64.onnx", learningModel));

  LearningModelSession mapSession(learningModel);
  LearningModelBinding mapBinding(mapSession);

  auto inputName = learningModel.InputFeatures().First().Current().Name();

  // Bind invalid image as image input
  auto smallImage = FileHelpers::LoadImageFeatureValue(L"100x100.png");
  WINML_EXPECT_THROW_SPECIFIC(mapBinding.Bind(inputName, smallImage), winrt::hresult_error, ensureWinmlInvalidBinding);

  // Bind invalid map as image input
  WINML_EXPECT_THROW_SPECIFIC(mapBinding.Bind(inputName, abiMap), winrt::hresult_error, ensureWinmlInvalidBinding);

  // Bind invalid sequence as image input
  WINML_EXPECT_THROW_SPECIFIC(mapBinding.Bind(inputName, abiSequence), winrt::hresult_error, ensureWinmlInvalidBinding);

  // Bind invalid tensor type as image input
  WINML_EXPECT_THROW_SPECIFIC(
    mapBinding.Bind(inputName, tensorBoolean), winrt::hresult_error, ensureWinmlInvalidBinding
  );
}

// Verify that it throws an error when binding an invalid name.
static void BindInvalidInputName() {
  LearningModel learningModel = nullptr;
  LearningModelBinding learningModelBinding = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  LearningModelSession learningModelSession = nullptr;
  std::wstring modelPath = FileHelpers::GetModulePath() + L"Add_ImageNet1920.onnx";
  WINML_EXPECT_NO_THROW(learningModel = LearningModel::LoadFromFilePath(modelPath));
  WINML_EXPECT_TRUE(learningModel != nullptr);
  WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::Default));
  WINML_EXPECT_NO_THROW(learningModelSession = LearningModelSession(learningModel, learningModelDevice));
  WINML_EXPECT_NO_THROW(learningModelBinding = LearningModelBinding(learningModelSession));

  VideoFrame iuputImage(BitmapPixelFormat::Rgba8, 1920, 1080);
  ImageFeatureValue inputTensor = ImageFeatureValue::CreateFromVideoFrame(iuputImage);

  auto first = learningModel.InputFeatures().First();
  std::wstring testInvalidName = L"0";

  // Verify that testInvalidName is not in model's InputFeatures
  while (first.HasCurrent()) {
    WINML_EXPECT_NOT_EQUAL(testInvalidName, first.Current().Name());
    first.MoveNext();
  }

  // Bind inputTensor to a valid input name
  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"input_39:0", inputTensor));

  // Bind inputTensor to an invalid input name
  WINML_EXPECT_THROW_SPECIFIC(
    learningModelBinding.Bind(testInvalidName, inputTensor),
    winrt::hresult_error,
    [](const winrt::hresult_error& e) -> bool { return e.code() == WINML_ERR_INVALID_BINDING; }
  );
}

static void VerifyOutputAfterEvaluateAsyncCalledTwice() {
  LearningModel learningModel = nullptr;
  LearningModelBinding learningModelBinding = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  LearningModelSession learningModelSession = nullptr;
  std::wstring filePath = FileHelpers::GetModulePath() + L"relu.onnx";
  WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::Default));
  WINML_EXPECT_NO_THROW(learningModel = LearningModel::LoadFromFilePath(filePath));
  WINML_EXPECT_TRUE(learningModel != nullptr);
  WINML_EXPECT_NO_THROW(learningModelSession = LearningModelSession(learningModel, learningModelDevice));
  WINML_EXPECT_NO_THROW(learningModelBinding = LearningModelBinding(learningModelSession));

  auto inputShape = std::vector<int64_t>{5};
  auto inputData1 = std::vector<float>{-50.f, -25.f, 0.f, 25.f, 50.f};
  auto inputValue1 =
    TensorFloat::CreateFromIterable(inputShape, single_threaded_vector<float>(std::move(inputData1)).GetView());

  auto inputData2 = std::vector<float>{50.f, 25.f, 0.f, -25.f, -50.f};
  auto inputValue2 =
    TensorFloat::CreateFromIterable(inputShape, single_threaded_vector<float>(std::move(inputData2)).GetView());

  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"X", inputValue1));

  auto outputValue = TensorFloat::Create();
  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"Y", outputValue));

  WINML_EXPECT_NO_THROW(learningModelSession.Evaluate(learningModelBinding, L""));

  auto buffer1 = outputValue.GetAsVectorView();
  WINML_EXPECT_TRUE(buffer1 != nullptr);

  // The second evaluation
  // If we don't bind output again, the output value will not change
  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"X", inputValue2));
  WINML_EXPECT_NO_THROW(learningModelSession.Evaluate(learningModelBinding, L""));
  auto buffer2 = outputValue.GetAsVectorView();
  WINML_EXPECT_EQUAL(buffer1.Size(), buffer2.Size());
  bool isSame = true;
  for (uint32_t i = 0; i < buffer1.Size(); ++i) {
    if (buffer1.GetAt(i) != buffer2.GetAt(i)) {
      isSame = false;
      break;
    }
  }
  WINML_EXPECT_FALSE(isSame);
}

static VideoFrame CreateVideoFrame(const wchar_t* path) {
  auto imagefile = StorageFile::GetFileFromPathAsync(path).get();
  auto stream = imagefile.OpenAsync(FileAccessMode::Read).get();
  auto decoder = BitmapDecoder::CreateAsync(stream).get();
  auto softwareBitmap = decoder.GetSoftwareBitmapAsync().get();
  return VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
}

static void VerifyOutputAfterImageBindCalledTwice() {
  std::wstring fullModelPath = FileHelpers::GetModulePath() + L"model.onnx";
  std::wstring fullImagePath1 = FileHelpers::GetModulePath() + L"kitten_224.png";
  std::wstring fullImagePath2 = FileHelpers::GetModulePath() + L"fish.png";

  // winml model creation
  LearningModel model = nullptr;
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(fullModelPath));
  LearningModelSession modelSession = nullptr;
  WINML_EXPECT_NO_THROW(
    modelSession = LearningModelSession(model, LearningModelDevice(LearningModelDeviceKind::Default))
  );
  LearningModelBinding modelBinding(modelSession);

  // create the tensor for the actual output
  auto output = TensorFloat::Create();
  modelBinding.Bind(L"softmaxout_1", output);

  // Bind image 1 and evaluate
  auto frame = CreateVideoFrame(fullImagePath1.c_str());
  auto imageTensor = ImageFeatureValue::CreateFromVideoFrame(frame);
  WINML_EXPECT_NO_THROW(modelBinding.Bind(L"data_0", imageTensor));
  WINML_EXPECT_NO_THROW(modelSession.Evaluate(modelBinding, L""));

  // Store 1st result
  auto outputVectorView1 = output.GetAsVectorView();

  // Bind image 2 and evaluate
  // In this scenario, the backing videoframe is updated, and the imagefeaturevalue is rebound.
  // The expected result is that the videoframe will be re-tensorized at bind
  auto frame2 = CreateVideoFrame(fullImagePath2.c_str());
  frame2.CopyToAsync(frame).get();
  WINML_EXPECT_NO_THROW(modelBinding.Bind(L"data_0", imageTensor));
  WINML_EXPECT_NO_THROW(modelSession.Evaluate(modelBinding, L""));

  // Store 2nd result
  auto outputVectorView2 = output.GetAsVectorView();

  WINML_EXPECT_EQUAL(outputVectorView1.Size(), outputVectorView2.Size());
  bool isSame = true;
  for (uint32_t i = 0; i < outputVectorView1.Size(); ++i) {
    if (outputVectorView1.GetAt(i) != outputVectorView2.GetAt(i)) {
      isSame = false;
      break;
    }
  }
  WINML_EXPECT_FALSE(isSame);
}

static void SequenceLengthTensorFloat() {
  // Tests sequence of tensor float as an input
  LearningModel learningModel = nullptr;
  LearningModelBinding learningModelBinding = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  LearningModelSession learningModelSession = nullptr;
  std::wstring filePath = FileHelpers::GetModulePath() + L"sequence_length.onnx";
  WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::Default));
  WINML_EXPECT_NO_THROW(learningModel = LearningModel::LoadFromFilePath(filePath));
  WINML_EXPECT_TRUE(learningModel != nullptr);
  WINML_EXPECT_NO_THROW(learningModelSession = LearningModelSession(learningModel, learningModelDevice));
  WINML_EXPECT_NO_THROW(learningModelBinding = LearningModelBinding(learningModelSession));

  auto input = winrt::single_threaded_vector<TensorFloat>();
  for (int i = 0; i < 3; i++) {
    std::vector<int64_t> shape = {5, 3 * i + 1};
    std::vector<float> data(
      static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies()))
    );
    input.Append(TensorFloat::CreateFromShapeArrayAndDataArray(shape, data));
  }

  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"X", input));
  auto results = learningModelSession.Evaluate(learningModelBinding, L"");

  WINML_EXPECT_EQUAL(3, results.Outputs().Lookup(L"Y").as<TensorInt64Bit>().GetAsVectorView().GetAt(0));
}

static void SequenceConstructTensorString() {
  LearningModel learningModel = nullptr;
  LearningModelBinding learningModelBinding = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  LearningModelSession learningModelSession = nullptr;
  std::wstring filePath = FileHelpers::GetModulePath() + L"sequence_construct.onnx";
  WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::Default));
  WINML_EXPECT_NO_THROW(learningModel = LearningModel::LoadFromFilePath(filePath));
  WINML_EXPECT_TRUE(learningModel != nullptr);
  WINML_EXPECT_NO_THROW(learningModelSession = LearningModelSession(learningModel, learningModelDevice));
  WINML_EXPECT_NO_THROW(learningModelBinding = LearningModelBinding(learningModelSession));

  std::vector<int64_t> shape1 = {2, 3};
  std::vector<int64_t> data1(
    static_cast<size_t>(std::accumulate(shape1.begin(), shape1.end(), static_cast<int64_t>(1), std::multiplies()))
  );
  auto input1 = TensorInt64Bit::CreateFromShapeArrayAndDataArray(shape1, data1);
  std::vector<int64_t> shape2 = {2, 3};
  std::vector<int64_t> data2(
    static_cast<size_t>(std::accumulate(shape2.begin(), shape2.end(), static_cast<int64_t>(1), std::multiplies()))
  );
  auto input2 = TensorInt64Bit::CreateFromShapeArrayAndDataArray(shape2, data2);

  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"tensor1", input1));
  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"tensor2", input2));
  auto results = learningModelSession.Evaluate(learningModelBinding, L"");

  auto output_sequence = results.Outputs().Lookup(L"output_sequence").as<wfc::IVectorView<TensorInt64Bit>>();
  WINML_EXPECT_EQUAL(static_cast<uint32_t>(2), output_sequence.Size());
  WINML_EXPECT_EQUAL(2, output_sequence.GetAt(0).Shape().GetAt(0));
  WINML_EXPECT_EQUAL(3, output_sequence.GetAt(0).Shape().GetAt(1));
  WINML_EXPECT_EQUAL(2, output_sequence.GetAt(1).Shape().GetAt(0));
  WINML_EXPECT_EQUAL(3, output_sequence.GetAt(1).Shape().GetAt(1));

  auto bound_output_sequence = winrt::single_threaded_vector<TensorInt64Bit>();
  WINML_EXPECT_NO_THROW(learningModelBinding.Bind(L"output_sequence", bound_output_sequence));
  WINML_EXPECT_NO_THROW(learningModelSession.Evaluate(learningModelBinding, L""));
  WINML_EXPECT_EQUAL(static_cast<uint32_t>(2), bound_output_sequence.Size());
  WINML_EXPECT_EQUAL(2, bound_output_sequence.GetAt(0).Shape().GetAt(0));
  WINML_EXPECT_EQUAL(3, bound_output_sequence.GetAt(0).Shape().GetAt(1));
  WINML_EXPECT_EQUAL(2, bound_output_sequence.GetAt(1).Shape().GetAt(0));
  WINML_EXPECT_EQUAL(3, bound_output_sequence.GetAt(1).Shape().GetAt(1));
}

const LearningModelBindingAPITestsApi& getapi() {
  static LearningModelBindingAPITestsApi api = {
    LearningModelBindingAPITestsClassSetup,
    CpuSqueezeNet,
    CpuSqueezeNetEmptyOutputs,
    CpuSqueezeNetUnboundOutputs,
    CpuSqueezeNetBindInputTensorAsInspectable,
    CastMapInt64,
    DictionaryVectorizerMapInt64,
    DictionaryVectorizerMapString,
    ZipMapInt64,
    ZipMapInt64Unbound,
    ZipMapString,
    GpuSqueezeNet,
    GpuSqueezeNetEmptyOutputs,
    GpuSqueezeNetUnboundOutputs,
    ImageBindingDimensions,
    VerifyInvalidBindExceptions,
    BindInvalidInputName,
    VerifyOutputAfterEvaluateAsyncCalledTwice,
    VerifyOutputAfterImageBindCalledTwice,
    SequenceLengthTensorFloat,
    SequenceConstructTensorString};

  if (SkipGpuTests()) {
    api.GpuSqueezeNet = SkipTest;
    api.GpuSqueezeNetEmptyOutputs = SkipTest;
    api.GpuSqueezeNetUnboundOutputs = SkipTest;
  }
  if (RuntimeParameterExists(L"noVideoFrameTests")) {
    api.ImageBindingDimensions = SkipTest;
    api.BindInvalidInputName = SkipTest;
    api.VerifyOutputAfterImageBindCalledTwice = SkipTest;
    api.VerifyInvalidBindExceptions = SkipTest;
  }
  return api;
}
