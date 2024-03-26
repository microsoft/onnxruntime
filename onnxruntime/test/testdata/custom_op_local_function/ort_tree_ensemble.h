#pragma once

#include "common_kernels.h"
#include "c_op_tree_ensemble_common_.hpp"
#include "c_op_tree_ensemble_common_classifier_.hpp"

namespace ortops {

template <typename IFEATURETYPE, typename TTYPE, typename OTYPE> struct TreeEnsembleKernel {
  TreeEnsembleKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

  // Attributes
  int64_t n_targets_or_classes;
  std::unique_ptr<onnx_c_ops::TreeEnsembleCommon<IFEATURETYPE, TTYPE, OTYPE>>
      reg_type_type_type;
  std::unique_ptr<onnx_c_ops::TreeEnsembleCommonClassifier<IFEATURETYPE, TTYPE, OTYPE>>
      cls_type_type_type;
  bool is_classifier;
};

template <typename IFEATURETYPE, typename TTYPE, typename OTYPE>
struct TreeEnsembleRegressor
    : Ort::CustomOpBase<TreeEnsembleRegressor<IFEATURETYPE, TTYPE, OTYPE>,
                        TreeEnsembleKernel<IFEATURETYPE, TTYPE, OTYPE>> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
};

template <typename ITYPE, typename TTYPE, typename OTYPE>
struct TreeEnsembleClassifier : Ort::CustomOpBase<TreeEnsembleClassifier<ITYPE, TTYPE, OTYPE>,
                                                  TreeEnsembleKernel<ITYPE, TTYPE, OTYPE>> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;
  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
};

} // namespace ortops
