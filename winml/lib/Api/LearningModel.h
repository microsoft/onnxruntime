// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "LearningModel.g.h"

namespace Windows::AI::MachineLearning {
struct IEngineFactory;
struct IModel;
struct IModelInfo;
}  // namespace Windows::AI::MachineLearning

namespace winrt::Windows::AI::MachineLearning::implementation {

struct LearningModel : LearningModelT<LearningModel> {
  /* LearningModel constructors (MachineLearningContract 1). */
  LearningModel() = default;

  LearningModel(
      const hstring& path,
      const winml::ILearningModelOperatorProvider operator_provider);

  LearningModel(
      const wss::IRandomAccessStreamReference stream,
      const winml::ILearningModelOperatorProvider operator_provider);

  LearningModel(
      const std::string& path,
      const winml::ILearningModelOperatorProvider operator_provider);

  /* LearningModel properties (MachineLearningContract 1). */
  hstring
  Author();

  hstring
  Name();

  hstring
  Domain();

  hstring
  Description();

  int64_t
  Version();

  wfc::IMapView<hstring, hstring>
  Metadata();

  wfc::IVectorView<winml::ILearningModelFeatureDescriptor>
  InputFeatures();

  wfc::IVectorView<winml::ILearningModelFeatureDescriptor>
  OutputFeatures();

  /* IClosable methods. */
  void Close();

  /* LearningModel static methods (MachineLearningContract 1). */
  static wf::IAsyncOperation<winml::LearningModel>
  LoadFromStorageFileAsync(
      Windows::Storage::IStorageFile const model_file);

  static wf::IAsyncOperation<winml::LearningModel>
  LoadFromStorageFileAsync(
      Windows::Storage::IStorageFile const model_file,
      winml::ILearningModelOperatorProvider const operator_provider);

  static wf::IAsyncOperation<winml::LearningModel>
  LoadFromStreamAsync(
      wss::IRandomAccessStreamReference const stream);

  static wf::IAsyncOperation<winml::LearningModel>
  LoadFromStreamAsync(
      wss::IRandomAccessStreamReference const stream,
      winml::ILearningModelOperatorProvider const operator_provider);

  static winml::LearningModel
  LoadFromFilePath(
      hstring const& path);

  static winml::LearningModel
  LoadFromFilePath(
      hstring const& path,
      winml::ILearningModelOperatorProvider const operator_provider);

  static winml::LearningModel
  LoadFromStream(
      wss::IRandomAccessStreamReference const stream);

  static winml::LearningModel
  LoadFromStream(
      wss::IRandomAccessStreamReference const stream,
      winml::ILearningModelOperatorProvider const operator_provider);

 public:
  /* Non-ABI methods */
  bool IsDisposed();
  IMLOperatorRegistry* GetOperatorRegistry();
  WinML::IModel* DetachModel();
  WinML::IModel* CloneModel();
  WinML::IEngineFactory* GetEngineFactory();

 private:
  com_ptr<WinML::IEngineFactory> engine_factory_;
  com_ptr<WinML::IModel> model_;
  com_ptr<WinML::IModelInfo> model_info_;

  ILearningModelOperatorProvider operator_provider_;
};

}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {

struct LearningModel : LearningModelT<LearningModel, implementation::LearningModel, ILearningModelStaticsNative> {
  STDMETHOD(Load)
  (const wchar_t* p_model_path, UINT32 model_path_size, IUnknown** pp_model_unk);
};

}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
