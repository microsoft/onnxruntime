// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#include "LearningModel.h"

#include "TelemetryEvent.h"
#include "MapFeatureDescriptor.h"
#include "SequenceFeatureDescriptor.h"
#include "TensorFeatureDescriptor.h"

#include "OrtEngineFactory.h"

#include <robuffer.h>

namespace winrt::Windows::AI::MachineLearning::implementation {
LearningModel::LearningModel(
    const hstring& path,
    const winml::ILearningModelOperatorProvider op_provider) try : LearningModel(WinML::Strings::UTF8FromHString(path),
                                                                                 op_provider) {
}
WINML_CATCH_ALL

LearningModel::LearningModel(
    const std::string& path,
    const winml::ILearningModelOperatorProvider operator_provider) try : operator_provider_(operator_provider) {
  _winmlt::TelemetryEvent loadModel_event(_winmlt::EventCategory::kModelLoad);
  WINML_THROW_IF_FAILED(OrtGetWinMLAdapter(adapter_.put()));
  WINML_THROW_IF_FAILED(adapter_->OverrideSchemaInferenceFunctions());
  WINML_THROW_IF_FAILED(adapter_->CreateModelProto(path.c_str(), model_proto_.put()));

  // New adapter
  WINML_THROW_IF_FAILED(CreateOrtEngine(learning_engine_.put()));
  WINML_THROW_IF_FAILED(learning_engine_->CreateModel(path.c_str(), path.size(), model_.put()));
}
WINML_CATCH_ALL

static HRESULT CreateModelFromStream(
    WinML::IEngine* learning_engine,
    const wss::IRandomAccessStreamReference stream,
    WinML::IModel** model) {
  auto content = stream.OpenReadAsync().get();

  wss::Buffer buffer(static_cast<uint32_t>(content.Size()));
  auto result = content.ReadAsync(
                           buffer,
                           buffer.Capacity(),
                           wss::InputStreamOptions::None)
                    .get();

  auto bytes = buffer.try_as<::Windows::Storage::Streams::IBufferByteAccess>();
  WINML_THROW_HR_IF_NULL_MSG(E_UNEXPECTED, bytes, "Model stream is invalid.");

  void* data;
  WINML_THROW_IF_FAILED_MSG(bytes->Buffer(reinterpret_cast<byte**>(&data)), "Failed to acquire buffer from model stream.");

  size_t len = static_cast<size_t>(content.Size());
  WINML_THROW_IF_FAILED(learning_engine->CreateModel(data, len, model));

  return S_OK;
}

LearningModel::LearningModel(
    const wss::IRandomAccessStreamReference stream,
    const winml::ILearningModelOperatorProvider operator_provider) try : operator_provider_(operator_provider) {
  _winmlt::TelemetryEvent loadModel_event(_winmlt::EventCategory::kModelLoad);
  WINML_THROW_IF_FAILED(OrtGetWinMLAdapter(adapter_.put()));
  WINML_THROW_IF_FAILED(adapter_->OverrideSchemaInferenceFunctions());
  WINML_THROW_IF_FAILED(adapter_->CreateModelProto(
      static_cast<ABI::Windows::Storage::Streams::IRandomAccessStreamReference*>(winrt::get_abi(stream)),
      model_proto_.put()));

  WINML_THROW_IF_FAILED(CreateOrtEngine(learning_engine_.put()));
  WINML_THROW_IF_FAILED(CreateModelFromStream(learning_engine_.get(), stream, model_.put()));
}
WINML_CATCH_ALL

hstring
LearningModel::Author() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_->GetAuthor(&out, &len));
  return WinML::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

hstring
LearningModel::Name() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_->GetName(&out, &len));
  return WinML::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

hstring
LearningModel::Domain() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_->GetDomain(&out, &len));
  return WinML::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

hstring
LearningModel::Description() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_->GetDescription(&out, &len));
  return WinML::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

int64_t
LearningModel::Version() try {
  int64_t version;
  WINML_THROW_IF_FAILED(model_->GetVersion(&version));
  return version;
}
WINML_CATCH_ALL

wfc::IMapView<hstring, hstring>
LearningModel::Metadata() try {
  ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>* metadata;
  wfc::IMapView<hstring, hstring> out;
  WINML_THROW_IF_FAILED(model_->GetModelMetadata(&metadata));
  winrt::attach_abi(out, metadata);
  return out;
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

  IMLOperatorRegistry* registry = nullptr;
  WINML_THROW_IF_FAILED(adapter_->GetOperatorRegistry(operator_provider_native.get(), &registry));
  return registry;
}

wfc::IVectorView<winml::ILearningModelFeatureDescriptor>
LearningModel::InputFeatures() try {
  ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>* features;
  wfc::IVectorView<winml::ILearningModelFeatureDescriptor> out;
  WINML_THROW_IF_FAILED(model_->GetInputFeatures(&features));
  winrt::attach_abi(out, features);
  return out;
}
WINML_CATCH_ALL

wfc::IVectorView<winml::ILearningModelFeatureDescriptor>
LearningModel::OutputFeatures() try {
  ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>* features;
  wfc::IVectorView<winml::ILearningModelFeatureDescriptor> out;
  WINML_THROW_IF_FAILED(model_->GetOutputFeatures(&features));
  winrt::attach_abi(out, features);
  return out;
}
WINML_CATCH_ALL

void LearningModel::Close() try {
  // close the model
  model_proto_ = nullptr;
  model_ = nullptr;
}
WINML_CATCH_ALL

bool LearningModel::IsDisposed() {
  return model_proto_ == nullptr;
  //return model_ == nullptr;
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

winmla::IModelProto*
LearningModel::DetachModelProto() {
  com_ptr<winmla::IModelProto> detached_model_proto;
  if (model_proto_ != nullptr) {
    detached_model_proto.attach(model_proto_.detach());

    // Close the model since we now own the model proto
    Close();
  }
  return detached_model_proto.detach();
}

winmla::IModelProto*
LearningModel::CopyModelProto() {
  if (model_proto_ == nullptr) {
    return nullptr;
  }

  com_ptr<winmla::IWinMLAdapter> adapter;
  WINML_THROW_IF_FAILED(OrtGetWinMLAdapter(adapter.put()));
  com_ptr<winmla::IModelProto> model_proto;
  WINML_THROW_IF_FAILED(adapter->CreateModelProto(model_proto_.get(), model_proto.put()));

  return model_proto.detach();
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
