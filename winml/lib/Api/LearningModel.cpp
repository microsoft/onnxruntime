// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#include "LearningModel.h"

#include "TelemetryEvent.h"
#include "MapFeatureDescriptor.h"
#include "SequenceFeatureDescriptor.h"
#include "TensorFeatureDescriptor.h"

#include "OnnxruntimeProvider.h"

#include <robuffer.h>

namespace WINMLP {
LearningModel::LearningModel(
    const hstring& path,
    const winml::ILearningModelOperatorProvider op_provider) try : LearningModel(_winml::Strings::UTF8FromHString(path),
                                                                                 op_provider) {
}
WINML_CATCH_ALL

LearningModel::LearningModel(
    _winml::IEngineFactory* engine_factory,
    _winml::IModel* model,
    const winml::ILearningModelOperatorProvider operator_provider) try :
      operator_provider_(operator_provider) {
  engine_factory_.copy_from(engine_factory);
  model_.copy_from(model);
  WINML_THROW_IF_FAILED(model_->GetModelInfo(model_info_.put()));
}
WINML_CATCH_ALL

LearningModel::LearningModel(
    const std::string& path,
    const winml::ILearningModelOperatorProvider operator_provider) try : operator_provider_(operator_provider) {
  _winmlt::TelemetryEvent loadModel_event(_winmlt::EventCategory::kModelLoad);

  WINML_THROW_IF_FAILED(CreateOnnxruntimeEngineFactory(engine_factory_.put()));
  WINML_THROW_IF_FAILED(engine_factory_->CreateModel(path.c_str(), path.size(), model_.put()));
  WINML_THROW_IF_FAILED(model_->GetModelInfo(model_info_.put()));
}
WINML_CATCH_ALL

static HRESULT CreateModelFromStream(
    _winml::IEngineFactory* engine_factory,
    const wss::IRandomAccessStreamReference stream,
    _winml::IModel** model) {
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
  WINML_THROW_IF_FAILED(engine_factory->CreateModel(data, len, model));

  return S_OK;
}

LearningModel::LearningModel(
    const wss::IRandomAccessStreamReference stream,
    const winml::ILearningModelOperatorProvider operator_provider) try : operator_provider_(operator_provider) {
  _winmlt::TelemetryEvent loadModel_event(_winmlt::EventCategory::kModelLoad);

  WINML_THROW_IF_FAILED(CreateOnnxruntimeEngineFactory(engine_factory_.put()));
  WINML_THROW_IF_FAILED(CreateModelFromStream(engine_factory_.get(), stream, model_.put()));
  WINML_THROW_IF_FAILED(model_->GetModelInfo(model_info_.put()));
}
WINML_CATCH_ALL

hstring
LearningModel::Author() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_info_->GetAuthor(&out, &len));
  return _winml::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

hstring
LearningModel::Name() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_info_->GetName(&out, &len));
  return _winml::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

hstring
LearningModel::Domain() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_info_->GetDomain(&out, &len));
  return _winml::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

hstring
LearningModel::Description() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_info_->GetDescription(&out, &len));
  return _winml::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

int64_t
LearningModel::Version() try {
  int64_t version;
  WINML_THROW_IF_FAILED(model_info_->GetVersion(&version));
  return version;
}
WINML_CATCH_ALL

wfc::IMapView<hstring, hstring>
LearningModel::Metadata() try {
  ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>* metadata = nullptr;
  wfc::IMapView<hstring, hstring> out;
  WINML_THROW_IF_FAILED(model_info_->GetModelMetadata(&metadata));
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
  // Retrieve the "operator abi" registry.
  THROW_IF_FAILED(operator_provider_native->GetRegistry(&registry));
  return registry;
}

wfc::IVectorView<winml::ILearningModelFeatureDescriptor>
LearningModel::InputFeatures() try {
  ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>* features = nullptr;
  wfc::IVectorView<winml::ILearningModelFeatureDescriptor> out;
  WINML_THROW_IF_FAILED(model_info_->GetInputFeatures(&features));
  winrt::attach_abi(out, features);
  return out;
}
WINML_CATCH_ALL

wfc::IVectorView<winml::ILearningModelFeatureDescriptor>
LearningModel::OutputFeatures() try {
  ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>* features = nullptr;
  wfc::IVectorView<winml::ILearningModelFeatureDescriptor> out;
  WINML_THROW_IF_FAILED(model_info_->GetOutputFeatures(&features));
  winrt::attach_abi(out, features);
  return out;
}
WINML_CATCH_ALL

void LearningModel::Close() try {
  // close the model
  model_ = nullptr;
}
WINML_CATCH_ALL

bool LearningModel::IsDisposed() {
  return model_ == nullptr;
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

_winml::IModel*
LearningModel::DetachModel() {
  com_ptr<_winml::IModel> detached_model;
  if (model_ != nullptr) {
    detached_model.attach(model_.detach());

    // Close the model since we now own the model proto
    Close();
  }
  return detached_model.detach();
}

_winml::IModel*
LearningModel::CloneModel() {
  if (model_ == nullptr) {
    return nullptr;
  }

  com_ptr<_winml::IModel> model_copy;
  WINML_THROW_IF_FAILED(model_->CloneModel(model_copy.put()));

  return model_copy.detach();
}

_winml::IEngineFactory*
LearningModel::GetEngineFactory() {
  return engine_factory_.get();
}

}  // namespace WINMLP

namespace WINML::factory_implementation {
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

    auto path = _winml::Strings::UTF8FromUnicode(p_model_path, model_path_size);
    auto model = make<winmlp::LearningModel>(path, nullptr);
    *pp_model_unk = model.as<IUnknown>().detach();
    return S_OK;
  }
  WINML_CATCH_ALL_COM
}
}  // namespace WINML::factory_implementation
