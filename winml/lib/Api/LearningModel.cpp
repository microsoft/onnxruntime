// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#include "LearningModel.h"

#include "core/providers/dml/OperatorAuthorHelper/SchemaInferenceOverrider.h"
#include "core/providers/dml/DmlExecutionProvider/src/MLOperatorAuthorImpl.h"
#include "ModelInfo.h"
#include "PheonixSingleton.h"
#include "TelemetryEvent.h"

#include "LotusEnvironment.h"

#include "MapFeatureDescriptor.h"
#include "SequenceFeatureDescriptor.h"
#include "TensorFeatureDescriptor.h"

namespace winrt::Windows::AI::MachineLearning::implementation {
LearningModel::LearningModel(
    const hstring& path,
    const winml::ILearningModelOperatorProvider op_provider) try : LearningModel(WinML::Strings::UTF8FromHString(path),
                                                                                 op_provider) {}
WINML_CATCH_ALL

LearningModel::LearningModel(
    const std::string& path,
    const winml::ILearningModelOperatorProvider operator_provider) try : lotus_environment_(PheonixSingleton<WinML::LotusEnvironment>()),
                                                                         operator_provider_(operator_provider) {
  _winmlt::PerformanceTelemetryEvent kLoadModel_event(
      WinMLRuntimePerf::kLoadModel);

  OverrideShapeInferenceMethods();

  com_ptr<_winmla::IWinMLAdapter> adapter;
  WINML_THROW_IF_FAILED(OrtGetWinMLAdapter(adapter.put()));
  WINML_THROW_IF_FAILED(adapter->CreateModelProto(path.c_str(), model_proto_.put()));

  Initialize();

  LogCreationEvent(true);
}
WINML_CATCH_ALL

LearningModel::LearningModel(
    const wss::IRandomAccessStreamReference stream,
    const winml::ILearningModelOperatorProvider operator_provider) try : lotus_environment_(PheonixSingleton<WinML::LotusEnvironment>()),
                                                                         operator_provider_(operator_provider) {
  _winmlt::PerformanceTelemetryEvent kLoadModel_event(
      WinMLRuntimePerf::kLoadModel);

  OverrideShapeInferenceMethods();

  com_ptr<_winmla::IWinMLAdapter> adapter;
  WINML_THROW_IF_FAILED(OrtGetWinMLAdapter(adapter.put()));
  WINML_THROW_IF_FAILED(adapter->CreateModelProto(
      static_cast<ABI::Windows::Storage::Streams::IRandomAccessStreamReference*>(winrt::get_abi(stream)), 
      model_proto_.put()));
  
  Initialize();

  LogCreationEvent(true);
}
WINML_CATCH_ALL

void LearningModel::Initialize() {
  model_info_ = std::make_unique<WinML::ModelInfo>(
      model_proto_.get()->get());
}

void LearningModel::LogCreationEvent(bool fromStream) {
  auto input_descriptors = InputFeatures();
  bool use_fp16 = false;
  for (auto descriptor : input_descriptors) {
    ModelUseFP16(descriptor, use_fp16);
    if (use_fp16) {
      break;
    }
  }
  telemetry_helper.LogModelCreation(
      fromStream,
      model_info_->author_,
      model_info_->name_,
      model_info_->domain_,
      model_info_->description_,
      model_info_->version_,
      use_fp16,
      model_info_->model_metadata_);
}

void LearningModel::ModelUseFP16(
    winml::ILearningModelFeatureDescriptor descriptor,
    bool& use_fp16) {
  auto kind = descriptor.Kind();
  switch (kind) {
    case LearningModelFeatureKind::Image:
      //images do not support float16 yet
      break;
    case LearningModelFeatureKind::Map: {
      auto map_descriptor = descriptor.as<MapFeatureDescriptor>();
      ModelUseFP16(map_descriptor->ValueDescriptor(), use_fp16);
    } break;
    case LearningModelFeatureKind::Sequence: {
      auto sequence_descriptor = descriptor.as<SequenceFeatureDescriptor>();
      ModelUseFP16(sequence_descriptor->ElementDescriptor(), use_fp16);
    } break;
    case LearningModelFeatureKind::Tensor: {
      auto tensor_descriptor = descriptor.as<TensorFeatureDescriptor>();
      if (tensor_descriptor->TensorKind() == TensorKind::Float16) {
        use_fp16 = true;
        return;
      }
    } break;
    default:
      break;
  }
}

hstring
LearningModel::Author() try {
  return WinML::Strings::HStringFromUTF8(model_info_->author_);
}
WINML_CATCH_ALL

hstring
LearningModel::Name() try {
  return WinML::Strings::HStringFromUTF8(
      model_info_->name_);
}
WINML_CATCH_ALL

hstring
LearningModel::Domain() try {
  return WinML::Strings::HStringFromUTF8(
      model_info_->domain_);
}
WINML_CATCH_ALL

hstring
LearningModel::Description() try {
  return WinML::Strings::HStringFromUTF8(
      model_info_->description_);
}
WINML_CATCH_ALL

int64_t
LearningModel::Version() try {
  return model_info_->version_;
}
WINML_CATCH_ALL

wfc::IMapView<hstring, hstring>
LearningModel::Metadata() try {
  std::unordered_map<hstring, hstring> map_copy;
  for (auto& pair : model_info_->model_metadata_) {
    auto key = WinML::Strings::HStringFromUTF8(pair.first);
    auto value = WinML::Strings::HStringFromUTF8(pair.second);
    map_copy.emplace(std::move(key), std::move(value));
  }
  auto metadata = winrt::single_threaded_map<winrt::hstring, winrt::hstring>(
      std::move(map_copy));
  return metadata.GetView();
}
WINML_CATCH_ALL

IMLOperatorRegistry*
LearningModel::GetOperatorRegistry() {
  if (operator_provider_ == nullptr) {
    return nullptr;
  }

  // Get the native winrt provider interface out of winrt operator provider.
  auto operator_provider_native =
      operator_provider_.as<ILearningModelOperatorProviderNative>();

  // Retrieve the "operator abi" registry.
  winrt::com_ptr<IMLOperatorRegistry> operator_registry;
  operator_provider_native->GetRegistry(operator_registry.put());

  return operator_registry.get();
}

wfc::IVectorView<winml::ILearningModelFeatureDescriptor>
LearningModel::InputFeatures() try {
  return model_info_->input_features_.GetView();
}
WINML_CATCH_ALL

wfc::IVectorView<winml::ILearningModelFeatureDescriptor>
LearningModel::OutputFeatures() try {
  return model_info_->output_features_.GetView();
}
WINML_CATCH_ALL

void LearningModel::Close() try {
  // close the model
  model_proto_ = nullptr;
}
WINML_CATCH_ALL

bool LearningModel::IsDisposed() {
  return model_proto_ == nullptr;
}

wf::IAsyncOperation<winml::LearningModel>
LearningModel::LoadFromStorageFileAsync(
    ws::IStorageFile const modelFile) {
  return LoadFromStorageFileAsync(modelFile, nullptr);
}

wf::IAsyncOperation<winml::LearningModel>
LearningModel::LoadFromStorageFileAsync(
    ws::IStorageFile const modelFile,
    winml::ILearningModelOperatorProvider const provider) {
  co_await resume_background();
  return make<LearningModel>(modelFile, provider);
}

wf::IAsyncOperation<winml::LearningModel>
LearningModel::LoadFromStreamAsync(
    wss::IRandomAccessStreamReference const model_stream) {
  return LoadFromStreamAsync(model_stream, nullptr);
}

wf::IAsyncOperation<winml::LearningModel>
LearningModel::LoadFromStreamAsync(
    wss::IRandomAccessStreamReference const model_stream,
    winml::ILearningModelOperatorProvider const provider) {
  co_await resume_background();
  return make<LearningModel>(model_stream, provider);
}

winml::LearningModel
LearningModel::LoadFromFilePath(
    hstring const& path) try {
  return LoadFromFilePath(path, nullptr);
}
WINML_CATCH_ALL

winml::LearningModel
LearningModel::LoadFromFilePath(
    hstring const& path,
    winml::ILearningModelOperatorProvider const provider) try {
  return make<LearningModel>(path, provider);
}
WINML_CATCH_ALL

winml::LearningModel
LearningModel::LoadFromStream(
    wss::IRandomAccessStreamReference const model_stream) try {
  return LoadFromStream(model_stream, nullptr);
}
WINML_CATCH_ALL

winml::LearningModel
LearningModel::LoadFromStream(
    wss::IRandomAccessStreamReference const model_stream,
    winml::ILearningModelOperatorProvider const provider) try {
  return make<LearningModel>(model_stream, provider);
}
WINML_CATCH_ALL

_winmla::IModelProto*
LearningModel::DetachModelProto() {
  com_ptr<_winmla::IModelProto> detached_model_proto;
  if (model_proto_ != nullptr) {
    detached_model_proto.attach(model_proto_.detach());

    // Close the model since we now own the model proto
    Close();
  }
  return detached_model_proto.detach();
}

_winmla::IModelProto*
LearningModel::CopyModelProto() {
  if (model_proto_ == nullptr) {
    return nullptr;
  }

  com_ptr<_winmla::IWinMLAdapter> adapter;
  WINML_THROW_IF_FAILED(OrtGetWinMLAdapter(adapter.put()));
  com_ptr<_winmla::IModelProto> model_proto;
  WINML_THROW_IF_FAILED(adapter->CreateModelProto(model_proto_.get(), model_proto.put()));

  return model_proto.detach();
}

static std::once_flag g_schema_override_once_flag;

// Override select shape inference functions which are incomplete in ONNX with versions that are complete,
// and are also used in DML kernel registrations.  Doing this avoids kernel and shader creation being
// deferred until first evaluation.  It also prevents a situation where inference functions in externally
// registered schema are reachable only after upstream schema have been revised in a later OS release,
// which would be a compatibility risk.
void LearningModel::OverrideShapeInferenceMethods() {
  std::call_once(g_schema_override_once_flag, []() {
    SchemaInferenceOverrider::OverrideSchemaInferenceFunctions();
  });
}
}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {
// copied from cppwinrt magic to create abi wrappers.   Need to do it this way
// since peeps underneath (like the constructor) will throw
HRESULT
__stdcall LearningModel::Load(
    const wchar_t* p_model_path,
    uint32_t model_path_size,
    IUnknown** pp_model_unk) {
  try {
    WINML_THROW_HR_IF_NULL_MSG(E_INVALIDARG, p_model_path, "Failed to create LearningModel. Ivalid argument p_model_path.");
    WINML_THROW_HR_IF_FALSE_MSG(E_INVALIDARG, model_path_size > 0, "Failed to create LearningModel. Ivalid argument model_path_size.");
    WINML_THROW_HR_IF_NULL_MSG(E_INVALIDARG, pp_model_unk, "Failed to create LearningModel. Ivalid argument pp_model_unk.");

    auto path = WinML::Strings::UTF8FromUnicode(p_model_path, model_path_size);
    auto model = make<winmlp::LearningModel>(path, nullptr);
    *pp_model_unk = model.as<IUnknown>().detach();
    return S_OK;
  }
  WINML_CATCH_ALL_COM
}
}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
