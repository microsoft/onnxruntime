// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "LearningModel.g.h"

namespace _winml {
struct IEngineFactory;
struct IModel;
struct IModelInfo;
}  // namespace _winml

namespace WINMLP {

struct LearningModel : LearningModelT<LearningModel> {
  /* LearningModel constructors (MachineLearningContract 1). */
  LearningModel() = default;

  LearningModel(const hstring& path, const winml::ILearningModelOperatorProvider operator_provider);

  LearningModel(
    const wss::IRandomAccessStreamReference stream, const winml::ILearningModelOperatorProvider operator_provider
  );

  LearningModel(
    _winml::IEngineFactory* engine_factory,
    _winml::IModel* model,
    const winml::ILearningModelOperatorProvider operator_provider
  );

  /* LearningModel properties (MachineLearningContract 1). */
  hstring Author();

  hstring Name();

  hstring Domain();

  hstring Description();

  int64_t Version();

  wfc::IMapView<hstring, hstring> Metadata();

  wfc::IVectorView<winml::ILearningModelFeatureDescriptor> InputFeatures();

  wfc::IVectorView<winml::ILearningModelFeatureDescriptor> OutputFeatures();

  void SetName(const hstring& name);

  /* IClosable methods. */
  void Close();

  /* LearningModel static methods (MachineLearningContract 1). */
  static wf::IAsyncOperation<winml::LearningModel> LoadFromStorageFileAsync(ws::IStorageFile const model_file);

  static wf::IAsyncOperation<winml::LearningModel> LoadFromStorageFileAsync(
    ws::IStorageFile const model_file, winml::ILearningModelOperatorProvider const operator_provider
  );

  static wf::IAsyncOperation<winml::LearningModel> LoadFromStreamAsync(wss::IRandomAccessStreamReference const stream);

  static wf::IAsyncOperation<winml::LearningModel> LoadFromStreamAsync(
    wss::IRandomAccessStreamReference const stream, winml::ILearningModelOperatorProvider const operator_provider
  );

  static winml::LearningModel LoadFromFilePath(hstring const& path);

  static winml::LearningModel LoadFromFilePath(
    hstring const& path, winml::ILearningModelOperatorProvider const operator_provider
  );

  static winml::LearningModel LoadFromStream(wss::IRandomAccessStreamReference const stream);

  static winml::LearningModel LoadFromStream(
    wss::IRandomAccessStreamReference const stream, winml::ILearningModelOperatorProvider const operator_provider
  );

 public:
  /* Non-ABI methods */
  bool IsDisposed();
  IMLOperatorRegistry* GetOperatorRegistry();
  _winml::IModel* DetachModel();
  _winml::IModel* CloneModel();
  _winml::IEngineFactory* GetEngineFactory();
  void SaveToFile(const hstring& file_name);
  void JoinModel(
    winml::LearningModel other,
    const std::unordered_map<std::string, std::string>& linkages,
    bool promote_unlinked_outputs,
    bool close_model_on_join,
    const winrt::hstring& join_node_prefix
  );

 private:
  com_ptr<_winml::IEngineFactory> engine_factory_;
  com_ptr<_winml::IModel> model_;
  com_ptr<_winml::IModelInfo> model_info_;

  ILearningModelOperatorProvider operator_provider_;
};

}  // namespace WINMLP
namespace WINML::factory_implementation {

struct LearningModel : LearningModelT<LearningModel, implementation::LearningModel, ILearningModelStaticsNative> {
  STDMETHOD(Load)
  (const wchar_t* p_model_path, UINT32 model_path_size, IUnknown** pp_model_unk);
};

}  // namespace WINML::factory_implementation
