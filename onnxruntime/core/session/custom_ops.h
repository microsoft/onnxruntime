// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {

common::Status CreateCustomRegistry(gsl::span<OrtCustomOpDomain* const> op_domains, std::shared_ptr<CustomRegistry>& output);

}
