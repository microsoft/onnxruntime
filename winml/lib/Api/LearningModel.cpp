// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api/pch/pch.h"

#include "LearningModel.h"

#include "TelemetryEvent.h"
#include "MapFeatureDescriptor.h"
#include "SequenceFeatureDescriptor.h"
#include "TensorFeatureDescriptor.h"

#include "OnnxruntimeProvider.h"

#include <robuffer.h>

namespace WINMLP {

// IBuffer implementation to avoid calling into WinTypes.dll to create wss::Buffer.
// This will enable model creation on VTL1 without pulling in additional binaries on load.
template <typename T>
class STLVectorBackedBuffer
  : public winrt::implements<STLVectorBackedBuffer<T>, wss::IBuffer, ::Windows::Storage::Streams::IBufferByteAccess> {
 private:
  std::vector<T> data_;
  size_t length_ = 0;

 public:
  STLVectorBackedBuffer(size_t num_elements) : data_(num_elements) {}

  uint32_t Capacity() const try {
    // Return the size of the backing vector in bytes
    return static_cast<uint32_t>(data_.size() * sizeof(T));
  }
  WINML_CATCH_ALL

  uint32_t Length() const try {
    // Return the used buffer in bytes
    return static_cast<uint32_t>(length_);
  }
  WINML_CATCH_ALL

  void Length(uint32_t value) try {
    // Set the use buffer length in bytes
    WINML_THROW_HR_IF_TRUE_MSG(
      E_INVALIDARG, value > Capacity(), "Parameter 'value' cannot be greater than the buffer's capacity."
    );
    length_ = value;
  }
  WINML_CATCH_ALL

  STDMETHOD(Buffer)
  (_Outptr_ BYTE** value) {
    // Return the buffer
    RETURN_HR_IF_NULL(E_POINTER, value);
    *value = reinterpret_cast<BYTE*>(data_.data());
    return S_OK;
  }
};

LearningModel::LearningModel(const hstring& path, const winml::ILearningModelOperatorProvider op_provider) try
  : operator_provider_(op_provider) {
  _winmlt::TelemetryEvent loadModel_event(_winmlt::EventCategory::kModelLoad);

  WINML_THROW_IF_FAILED(CreateOnnxruntimeEngineFactory(engine_factory_.put()));

  wil::unique_handle file_handle {
#if WINVER >= _WIN32_WINNT_WIN8
    CreateFile2(path.c_str(), GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NULL)
  };
#else
    CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, NULL)
  };
#endif

  WINML_THROW_HR_IF_TRUE_MSG(
    __HRESULT_FROM_WIN32(GetLastError()), file_handle.get() == INVALID_HANDLE_VALUE, "Model load failed!"
  );

  auto file_mapping = wil::unique_handle(CreateFileMappingW(
    file_handle.get(),  // current file handle
    NULL,               // default security
    PAGE_READONLY,      // read/write permission
    0,                  // size of mapping object, high
    0,                  // size of mapping object, low
    NULL
  ));  // name of mapping object

  WINML_THROW_HR_IF_TRUE_MSG(__HRESULT_FROM_WIN32(GetLastError()), file_mapping == nullptr, "Model load failed!");

  auto buffer = MapViewOfFile(
    file_mapping.get(),  // handle to mapping object
    FILE_MAP_READ,       // read/write
    0,                   // high-order 32 bits of file offset
    0,                   // low-order 32 bits of file offset
    0
  );  // number of bytes to map. 0 means read whole file.

  WINML_THROW_HR_IF_TRUE_MSG(__HRESULT_FROM_WIN32(GetLastError()), buffer == nullptr, "Model load failed!");
  LARGE_INTEGER file_size;
  WINML_THROW_HR_IF_FALSE_MSG(
    __HRESULT_FROM_WIN32(GetLastError()), GetFileSizeEx(file_handle.get(), &file_size), "GetFileSizeEx"
  );
  WINML_THROW_IF_FAILED(engine_factory_->CreateModel(buffer, static_cast<size_t>(file_size.QuadPart), model_.put()));
  WINML_THROW_HR_IF_TRUE_MSG(E_UNEXPECTED, UnmapViewOfFile(buffer) == 0, "Could not unmap model file.");
  WINML_THROW_IF_FAILED(model_->GetModelInfo(model_info_.put()));
}
WINML_CATCH_ALL

LearningModel::LearningModel(
  _winml::IEngineFactory* engine_factory,
  _winml::IModel* model,
  const winml::ILearningModelOperatorProvider operator_provider
) try
  : operator_provider_(operator_provider) {
  engine_factory_.copy_from(engine_factory);
  model_.copy_from(model);
  WINML_THROW_IF_FAILED(model_->GetModelInfo(model_info_.put()));
}
WINML_CATCH_ALL

static HRESULT CreateModelFromStream(
  _winml::IEngineFactory* engine_factory, const wss::IRandomAccessStreamReference stream, _winml::IModel** model
) {
  auto content = stream.OpenReadAsync().get();

  auto buffer = winrt::make<STLVectorBackedBuffer<BYTE>>(static_cast<size_t>(content.Size()));
  auto result = content.ReadAsync(buffer, buffer.Capacity(), wss::InputStreamOptions::None).get();

  auto bytes = buffer.try_as<::Windows::Storage::Streams::IBufferByteAccess>();
  WINML_THROW_HR_IF_NULL_MSG(E_UNEXPECTED, bytes, "Model stream is invalid.");

  void* data;
  WINML_THROW_IF_FAILED_MSG(
    bytes->Buffer(reinterpret_cast<byte**>(&data)), "Failed to acquire buffer from model stream."
  );

  size_t len = static_cast<size_t>(content.Size());
  if (FAILED(engine_factory->CreateModel(data, len, model))) {
    WINML_THROW_HR(E_INVALIDARG);
  }

  return S_OK;
}

LearningModel::LearningModel(
  const wss::IRandomAccessStreamReference stream, const winml::ILearningModelOperatorProvider operator_provider
) try
  : operator_provider_(operator_provider) {
  _winmlt::TelemetryEvent loadModel_event(_winmlt::EventCategory::kModelLoad);

  WINML_THROW_IF_FAILED(CreateOnnxruntimeEngineFactory(engine_factory_.put()));
  WINML_THROW_IF_FAILED(CreateModelFromStream(engine_factory_.get(), stream, model_.put()));
  WINML_THROW_IF_FAILED(model_->GetModelInfo(model_info_.put()));
}
WINML_CATCH_ALL

hstring LearningModel::Author() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_info_->GetAuthor(&out, &len));
  return _winml::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

hstring LearningModel::Name() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_info_->GetName(&out, &len));
  return _winml::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

hstring LearningModel::Domain() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_info_->GetDomain(&out, &len));
  return _winml::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

hstring LearningModel::Description() try {
  const char* out;
  size_t len;
  WINML_THROW_IF_FAILED(model_info_->GetDescription(&out, &len));
  return _winml::Strings::HStringFromUTF8(out);
}
WINML_CATCH_ALL

int64_t LearningModel::Version() try {
  int64_t version;
  WINML_THROW_IF_FAILED(model_info_->GetVersion(&version));
  return version;
}
WINML_CATCH_ALL

wfc::IMapView<hstring, hstring> LearningModel::Metadata() try {
  ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING>* metadata = nullptr;
  wfc::IMapView<hstring, hstring> out;
  WINML_THROW_IF_FAILED(model_info_->GetModelMetadata(&metadata));
  winrt::attach_abi(out, metadata);
  return out;
}
WINML_CATCH_ALL

IMLOperatorRegistry* LearningModel::GetOperatorRegistry() {
  if (operator_provider_ == nullptr) {
    return nullptr;
  }

  // Get the native winrt provider interface out of winrt operator provider.
  auto operator_provider_native = operator_provider_.as<ILearningModelOperatorProviderNative>();

  IMLOperatorRegistry* registry = nullptr;
  // Retrieve the "operator abi" registry.
  THROW_IF_FAILED(operator_provider_native->GetRegistry(&registry));
  return registry;
}

wfc::IVectorView<winml::ILearningModelFeatureDescriptor> LearningModel::InputFeatures() try {
  ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>* features = nullptr;
  wfc::IVectorView<winml::ILearningModelFeatureDescriptor> out;
  WINML_THROW_IF_FAILED(model_info_->GetInputFeatures(&features));
  winrt::attach_abi(out, features);
  return out;
}
WINML_CATCH_ALL

wfc::IVectorView<winml::ILearningModelFeatureDescriptor> LearningModel::OutputFeatures() try {
  ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>* features = nullptr;
  wfc::IVectorView<winml::ILearningModelFeatureDescriptor> out;
  WINML_THROW_IF_FAILED(model_info_->GetOutputFeatures(&features));
  winrt::attach_abi(out, features);
  return out;
}
WINML_CATCH_ALL

void LearningModel::SetName(const hstring& name) try {
  auto name_std_str = _winml::Strings::UTF8FromHString(name);
  auto name_c_str = name_std_str.c_str();
  WINML_THROW_IF_FAILED(model_->SetName(name_c_str));
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

wf::IAsyncOperation<winml::LearningModel> LearningModel::LoadFromStorageFileAsync(ws::IStorageFile const modelFile) {
  return LoadFromStorageFileAsync(modelFile, nullptr);
}

wf::IAsyncOperation<winml::LearningModel> LearningModel::LoadFromStorageFileAsync(
  ws::IStorageFile const modelFile, winml::ILearningModelOperatorProvider const provider
) {
  co_await resume_background();
  return make<LearningModel>(modelFile, provider);
}

wf::IAsyncOperation<winml::LearningModel> LearningModel::LoadFromStreamAsync(
  wss::IRandomAccessStreamReference const model_stream
) {
  return LoadFromStreamAsync(model_stream, nullptr);
}

wf::IAsyncOperation<winml::LearningModel> LearningModel::LoadFromStreamAsync(
  wss::IRandomAccessStreamReference const model_stream, winml::ILearningModelOperatorProvider const provider
) {
  co_await resume_background();
  return make<LearningModel>(model_stream, provider);
}

winml::LearningModel LearningModel::LoadFromFilePath(hstring const& path) try {
  return LoadFromFilePath(path, nullptr);
}
WINML_CATCH_ALL

winml::LearningModel LearningModel::LoadFromFilePath(
  hstring const& path, winml::ILearningModelOperatorProvider const provider
) try {
  return make<LearningModel>(path, provider);
}
WINML_CATCH_ALL

winml::LearningModel LearningModel::LoadFromStream(wss::IRandomAccessStreamReference const model_stream) try {
  return LoadFromStream(model_stream, nullptr);
}
WINML_CATCH_ALL

winml::LearningModel LearningModel::LoadFromStream(
  wss::IRandomAccessStreamReference const model_stream, winml::ILearningModelOperatorProvider const provider
) try {
  return make<LearningModel>(model_stream, provider);
}
WINML_CATCH_ALL

_winml::IModel* LearningModel::DetachModel() {
  com_ptr<_winml::IModel> detached_model;
  if (model_ != nullptr) {
    detached_model.attach(model_.detach());

    // Close the model since we now own the model proto
    Close();
  }
  return detached_model.detach();
}

_winml::IModel* LearningModel::CloneModel() {
  if (model_ == nullptr) {
    return nullptr;
  }

  com_ptr<_winml::IModel> model_copy;
  WINML_THROW_IF_FAILED(model_->CloneModel(model_copy.put()));

  return model_copy.detach();
}

_winml::IEngineFactory* LearningModel::GetEngineFactory() {
  return engine_factory_.get();
}

void LearningModel::SaveToFile(const hstring& file_name) {
  model_->SaveModel(file_name.c_str(), file_name.size());
}

void LearningModel::JoinModel(
  winml::LearningModel other,
  const std::unordered_map<std::string, std::string>& linkages,
  bool promote_unlinked_outputs,
  bool close_model_on_join,
  const winrt::hstring& join_node_prefix
) {
  auto otherp = other.as<winmlp::LearningModel>();
  winrt::com_ptr<_winml::IModel> other_model;
  if (close_model_on_join) {
    other_model.attach(otherp->DetachModel());
  } else {
    other_model.attach(otherp->CloneModel());
  }

  std::vector<const char*> raw_outputs(linkages.size());
  std::vector<const char*> raw_inputs(linkages.size());
  std::transform(std::begin(linkages), std::end(linkages), std::begin(raw_outputs), [](auto& pair) {
    return pair.first.c_str();
  });
  std::transform(std::begin(linkages), std::end(linkages), std::begin(raw_inputs), [](auto& pair) {
    return pair.second.c_str();
  });

  auto prefix = winrt::to_string(join_node_prefix);
  WINML_THROW_IF_FAILED(model_->JoinModel(
    other_model.get(), raw_outputs.data(), raw_inputs.data(), linkages.size(), promote_unlinked_outputs, prefix.c_str()
  ));

  model_info_ = nullptr;
  WINML_THROW_IF_FAILED(model_->GetModelInfo(model_info_.put()));
}

}  // namespace WINMLP

namespace WINML::factory_implementation {
// copied from cppwinrt magic to create abi wrappers.   Need to do it this way
// since peeps underneath (like the constructor) will throw
HRESULT
__stdcall LearningModel::Load(const wchar_t* p_model_path, uint32_t model_path_size, IUnknown** pp_model_unk) {
  try {
    WINML_THROW_HR_IF_NULL_MSG(
      E_INVALIDARG, p_model_path, "Failed to create LearningModel. Ivalid argument p_model_path."
    );
    WINML_THROW_HR_IF_FALSE_MSG(
      E_INVALIDARG, model_path_size > 0, "Failed to create LearningModel. Ivalid argument model_path_size."
    );
    WINML_THROW_HR_IF_NULL_MSG(
      E_INVALIDARG, pp_model_unk, "Failed to create LearningModel. Ivalid argument pp_model_unk."
    );

    winrt::hstring path(p_model_path, model_path_size);
    auto model = make<winmlp::LearningModel>(path, nullptr);
    *pp_model_unk = model.as<IUnknown>().detach();
    return S_OK;
  }
  WINML_CATCH_ALL_COM
}
}  // namespace WINML::factory_implementation
