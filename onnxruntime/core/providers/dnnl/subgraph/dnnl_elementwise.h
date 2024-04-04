// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlElementwise {
 public:
  enum InputTensors : int {
    IN_X = 0,
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  DnnlElementwise();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  /*
   * GetAlpha will get the 'alpha' attribute if the attribute is not found
   * the the `default_alph_` will be returned instead. This is set to 1.0
   * by the `DnnlElementwise` constructor but should be updated for any operator
   * that has an 'alpha' property.
   *
   * See how `GetAlpha` is called for the 'Elu' operator in the `CreatePrimitive` code.
   *
   * Note: The number of operators that use the 'alpha' attribute is much smaller than
   * initially expected.
   */
  float GetAlpha(DnnlNode& node, float default_alpha);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime