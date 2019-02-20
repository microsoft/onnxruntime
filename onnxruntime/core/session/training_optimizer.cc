// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "core/session/training_optimizer.h"
#include "core/graph/onnx_protobuf.h"

using namespace onnxruntime;
using namespace onnxruntime::training;
using namespace std;

template <typename T>
auto MakeEigenArrayMap(Tensor& t) { return EigenVectorArrayMap<T>(t.template MutableData<T>(), t.Shape().Size()); }

template <typename T>
auto MakeEigenArrayMap(const Tensor& t) { return ConstEigenVectorArrayMap<T>(t.template Data<T>(), t.Shape().Size()); }

template <typename T>
auto MakeEigenArrayMap(Tensor* t) { return EigenVectorArrayMap<T>(t->template MutableData<T>(), t->Shape().Size()); }

template <typename T>
auto MakeEigenArrayMap(const Tensor* t) { return ConstEigenVectorArrayMap<T>(t->template Data<T>(), t->Shape().Size()); }

NameMLValMap GradientDescent::CalculateNewWeights(const NameMLValMap& original_weights,
                                                  const std::vector<NameMLValMap>& gradients_multi_batches) const {
  NameMLValMap newWeights;

  // Update weight - formula:
  // W = W - LearningRate * Sum(GradOfW[i])/n

  for (auto pair : original_weights) {
    auto gradName = pair.first + "_grad";
    auto element_type = DataTypeImpl::GetType<float>();
    auto gradEntry = gradients_multi_batches[0].find(gradName);

    if (gradEntry == gradients_multi_batches[0].end()) {
      throw new NotImplementedException("bad gradiant name");
    }

    auto shape = gradEntry->second.Get<Tensor>().Shape();

    void* buffer = param_.allocator_ptr_->Alloc(element_type->Size() * shape.Size());
    memset(buffer, 0, element_type->Size() * shape.Size());

    auto p_tensor = new Tensor(
        element_type,
        shape,
        buffer,
        param_.allocator_ptr_->Info(),
        param_.allocator_ptr_);

    auto outputAccumulatedGradArrayMap = MakeEigenArrayMap<float>(p_tensor);

    for (auto nvmap : gradients_multi_batches) {
      outputAccumulatedGradArrayMap += MakeEigenArrayMap<float>(nvmap[gradName].Get<Tensor>());
    }

    outputAccumulatedGradArrayMap /= (float)gradients_multi_batches.size();
    outputAccumulatedGradArrayMap *= param_.learning_rate_;

    auto originalValue = pair.second;
    auto& original_weight = originalValue.Get<Tensor>();
    auto originalWeightArrayMap = MakeEigenArrayMap<float>(original_weight);

    // reuse outputAccumulatedGradArrayMap to store final output - which is the new weight
    outputAccumulatedGradArrayMap = originalWeightArrayMap - outputAccumulatedGradArrayMap;

    MLValue newWeight;
    newWeight.Init(p_tensor,
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

    newWeights.insert(std::make_pair(pair.first, newWeight));
  }

  return newWeights;
}
