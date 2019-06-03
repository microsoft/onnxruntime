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
    InsertAllocator(std::make_unique<DummyAllocator>());
  }

  Status CopyTensor(const Tensor& src, Tensor& dst) const override {
    // we can 'copy' from anything we allocated to/from CPU
    ORT_ENFORCE(strcmp(dst.Location().name, DummyAllocator::kDummyAllocator) == 0 ||
                strcmp(dst.Location().name, CPU) == 0);
    ORT_ENFORCE(strcmp(src.Location().name, DummyAllocator::kDummyAllocator) == 0 ||
                strcmp(src.Location().name, CPU) == 0);

    // no really copy needed.
    const void* src_data = src.DataRaw();
    void* dst_data = dst.MutableDataRaw();

    // copying between cpu memory
    memcpy(dst_data, src_data, src.Size());

    return Status::OK();
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace test
}  // namespace onnxruntime
