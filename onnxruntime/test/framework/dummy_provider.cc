#include "dummy_provider.h"

#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
namespace test {

std::shared_ptr<KernelRegistry> DummyExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  return kernel_registry;
}

std::vector<AllocatorPtr> DummyExecutionProvider::CreatePreferredAllocators() {
  return std::vector<AllocatorPtr>{std::make_shared<DummyAllocator>()};
}

}  // namespace test
}  // namespace onnxruntime
