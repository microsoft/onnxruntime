// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/graph/onnx_protobuf.h"
#include "core/training/training_optimizer.h"
#include "core/graph/training/gradient_builder_base.h"
#include "core/graph/training/graph_augmenter.h"
#include "core/util/math_cpuonly.h"

// TODO: solve the op version issue in the entire training framework
// Here is to reference GRADIENT_OP_VERSION temporarily for global version control.
#include "core/graph/training/gradient_op_schema.h"

using namespace std;
namespace onnxruntime {
namespace training {
namespace out_graph_optimizer {

template <typename T>
auto MakeEigenArrayMap(Tensor& t) { return EigenVectorArrayMap<T>(t.template MutableData<T>(), t.Shape().Size()); }

template <typename T>
auto MakeEigenArrayMap(const Tensor& t) { return ConstEigenVectorArrayMap<T>(t.template Data<T>(), t.Shape().Size()); }

template <typename T>
auto MakeEigenArrayMap(Tensor* t) { return EigenVectorArrayMap<T>(t->template MutableData<T>(), t->Shape().Size()); }

template <typename T>
auto MakeEigenArrayMap(const Tensor* t) { return ConstEigenVectorArrayMap<T>(t->template Data<T>(), t->Shape().Size()); }

NameMLValMap GradientDescent::CalculateNewWeights(const NameMLValMap& original_weights,
                                                  const NameMLValMap& gradients,
                                                  size_t batch_size) const {
  NameMLValMap new_weights;

  // Update weight - formula:
  // W = W - LearningRate * GradOfW[i] / batch_size

  for (auto pair : original_weights) {
    std::string grad_name = pair.first + "_grad";
    auto element_type = DataTypeImpl::GetType<float>();

    auto grad_it = gradients.find(grad_name);
    if (grad_it == gradients.end()) {
      throw new NotImplementedException("bad gradiant name");
    }
    const Tensor& grad_tensor = grad_it->second.Get<Tensor>();
		
    auto shape = grad_tensor.Shape();
    void* buffer = param_.allocator_ptr_->Alloc(element_type->Size() * shape.Size());
    memset(buffer, 0, element_type->Size() * shape.Size());
    auto p_tensor = new Tensor(element_type,
                               shape,
                               buffer,
                               param_.allocator_ptr_->Info());	

    auto outputGradArrayMap = MakeEigenArrayMap<float>(p_tensor);
    outputGradArrayMap += MakeEigenArrayMap<float>(grad_tensor);

    outputGradArrayMap *= param_.learning_rate_ / static_cast<float>(batch_size);

    auto& original_weight = pair.second.Get<Tensor>();
    auto originalWeightArrayMap = MakeEigenArrayMap<float>(original_weight);

    // reuse outputGradArrayMap to store final output - which is the new weight
    outputGradArrayMap = originalWeightArrayMap - outputGradArrayMap;

    MLValue updated_weight;
    updated_weight.Init(p_tensor,
                        DataTypeImpl::GetType<Tensor>(),
                        DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

    new_weights.insert(std::make_pair(pair.first, updated_weight));
  }

  return new_weights;
}
}  // namespace out_graph_optimizer
}  // namespace training
}  // namespace onnxruntime
