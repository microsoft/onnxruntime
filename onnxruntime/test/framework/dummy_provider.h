// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_provider.h"
#include <memory>
#include "dummy_allocator.h"

namespace onnxruntime {
namespace test {

// Dummy execution provider that does nothing, but will trigger checks for copies to/from devices being needed
// in utils::ExecuteGraph
class DummyExecutionProvider : public IExecutionProvider {
  static constexpr const char* kDummyExecutionProviderType = "DummyExecutionProvider";

 public:
  DummyExecutionProvider() : IExecutionProvider{kDummyExecutionProviderType} {
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;
};

}  // namespace test
}  // namespace onnxruntime
