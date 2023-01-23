#include "core/framework/allocator.h"

namespace Dml
{
    class CPUAllocator : public onnxruntime::IAllocator
    {
    public:
        explicit CPUAllocator(OrtMemType memType);

        void* Alloc(size_t size) override;
        void Free(void* p) override;
    };

} // namespace Dml
