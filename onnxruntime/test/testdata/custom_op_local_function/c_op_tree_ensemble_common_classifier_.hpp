#pragma once
// Implements TreeEnsembleCommonClassifier.

// Inspired from
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc.

#include "c_op_tree_ensemble_common_.hpp"

// https://cims.nyu.edu/~stadler/hpc17/material/ompLec.pdf
// http://amestoy.perso.enseeiht.fr/COURS/CoursMulticoreProgrammingButtari.pdf

namespace onnx_c_ops {

template <typename FeatureType, typename ThresholdType, typename OutputType>
class TreeEnsembleCommonClassifier
    : public TreeEnsembleCommon<FeatureType, ThresholdType, OutputType> {
protected:
  bool weights_are_all_positive_;
  bool binary_case_;
  std::vector<int64_t> class_labels_;

public:
  Status Compute(
      int64_t n_rows, int64_t n_features,
      const typename TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::InputType *X,
      OutputType *Y, int64_t *label) const {
    FeatureType features(X, n_rows, n_features);
    switch (this->aggregate_function_) {
    case AGGREGATE_FUNCTION::SUM:
      DEBUG_PRINT("ComputeCl SUM")
      ComputeAggClassifier(features, Y, label,
                           TreeAggregatorSum<FeatureType, ThresholdType, OutputType>(
                               this->roots_.size(), this->n_targets_or_classes_,
                               this->post_transform_, this->base_values_, this->bias_));
      return Status::OK();
    default:
      EXT_THROW("Unknown aggregation function in TreeEnsemble.");
    }
  }

  Status Init(const std::string &aggregate_function,                       // 3
              const std::vector<ThresholdType> &base_values,               // 4
              int64_t n_targets_or_classes,                                // 5
              const std::vector<int64_t> &nodes_falsenodeids,              // 6
              const std::vector<int64_t> &nodes_featureids,                // 7
              const std::vector<ThresholdType> &nodes_hitrates,            // 8
              const std::vector<int64_t> &nodes_missing_value_tracks_true, // 9
              const std::vector<std::string> &nodes_modes,                 // 10
              const std::vector<int64_t> &nodes_nodeids,                   // 11
              const std::vector<int64_t> &nodes_treeids,                   // 12
              const std::vector<int64_t> &nodes_truenodeids,               // 13
              const std::vector<ThresholdType> &nodes_values,              // 14
              const std::string &post_transform,                           // 15
              const std::vector<int64_t> &class_ids,                       // 16
              const std::vector<int64_t> &class_nodeids,                   // 17
              const std::vector<int64_t> &class_treeids,                   // 18
              const std::vector<ThresholdType> &class_weights,             // 19
              bool is_classifier
  ) {
    TreeEnsembleCommon<FeatureType, ThresholdType, OutputType>::Init(
        aggregate_function,              // 3
        base_values,                     // 4
        n_targets_or_classes,            // 5
        nodes_falsenodeids,              // 6
        nodes_featureids,                // 7
        nodes_hitrates,                  // 8
        nodes_missing_value_tracks_true, // 9
        nodes_modes,                     // 10
        nodes_nodeids,                   // 11
        nodes_treeids,                   // 12
        nodes_truenodeids,               // 13
        nodes_values,                    // 14
        post_transform,                  // 15
        class_ids,                       // 16
        class_nodeids,                   // 17
        class_treeids,                   // 18
        class_weights,                   // 19
        is_classifier
    );
    DEBUG_PRINT("Init")

    InlinedHashSet<int64_t> weights_classes;
    weights_classes.reserve(class_ids.size());
    weights_are_all_positive_ = true;
    for (std::size_t i = 0, end = class_ids.size(); i < end; ++i) {
      weights_classes.insert(class_ids[i]);
      if (weights_are_all_positive_ && (class_weights[i] < 0))
        weights_are_all_positive_ = false;
    }
    binary_case_ = this->n_targets_or_classes_ == 2 && weights_classes.size() == 1;
    return Status::OK();
  }

protected:
  template <typename AGG>
  void ComputeAggClassifier(const FeatureType &data, OutputType *Y, int64_t *labels,
                            const AGG & /* agg */) const {
    DEBUG_PRINT("ComputeAggClassifier")
    this->ComputeAgg(data, Y, labels,
                     TreeAggregatorClassifier<FeatureType, ThresholdType, OutputType>(
                         this->roots_.size(), this->n_targets_or_classes_,
                         this->post_transform_, this->base_values_, this->bias_, binary_case_,
                         weights_are_all_positive_));
  }
};

} // namespace onnx_c_ops
