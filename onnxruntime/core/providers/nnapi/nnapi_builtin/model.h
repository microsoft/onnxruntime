// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

struct ANeuralNetworksModel;
struct ANeuralNetworksCompilation;
struct ANeuralNetworksExecution;
struct NnApi;

namespace onnxruntime {
namespace nnapi{

class Model{
private:
    ANeuralNetworksModel *model_{nullptr};
    ANeuralNetworksCompilation *compilation_{nullptr};
    ANeuralNetworksExecution *execution_{nullptr};
    const NnApi *nnapi_{nullptr};
    Model();

public:
    ~Model() {}
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
};

} } // namespace onnxruntime::nnapi