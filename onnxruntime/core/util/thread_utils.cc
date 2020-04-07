#include "thread_utils.h"
#include <algorithm>

#include <core/common/make_unique.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <thread>

namespace onnxruntime {
namespace concurrency {
static inline std::vector<size_t> GenerateVectorOfN(size_t n) {
  std::vector<size_t> ret(n);
  std::iota(ret.begin(), ret.end(), 0);
  return ret;
                }
#ifdef _WIN32
		// This function doesn't support systems with more than 64 logical processors
		static std::vector<size_t> GetNumCpuCores() {
			// Indeed 64 should be enough. However, it's harmless to have a little more.
			SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
			DWORD returnLength = sizeof(buffer);
			if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
                          return GenerateVectorOfN(std::thread::hardware_concurrency());
                        }
                        std::vector<size_t> ret;
			int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
			for (int i = 0; i != count; ++i) {
				if (buffer[i].Relationship == RelationProcessorCore) {
					ret.push_back(buffer[i].ProcessorMask);
				}
			}
			if (ret.empty())
                          return GenerateVectorOfN(std::thread::hardware_concurrency());
                        return ret;
                }
#else
		static std::vector<size_t> GetNumCpuCores() {
			return GenerateVectorOfN(std::thread::hardware_concurrency() / 2);
		}
#endif
		std::unique_ptr<ThreadPool> CreateThreadPool(Env* env, OrtThreadPoolParams options, Eigen::Allocator* allocator) {
			if (options.thread_pool_size == 1)
				return nullptr;
			std::vector<size_t> cpu_list;
			ThreadOptions to;
			if (options.affinity_vec_len != 0) {
				to.affinity.assign(options.affinity_vec, options.affinity_vec + options.affinity_vec_len);
			}
			if (options.thread_pool_size <= 0) {  // default
				cpu_list = GetNumCpuCores();
				if (cpu_list.empty() || cpu_list.size() == 1)
					return nullptr;
				options.thread_pool_size = static_cast<int>(cpu_list.size());
				if (options.auto_set_affinity)
					to.affinity = cpu_list;
			}

			return onnxruntime::make_unique<ThreadPool>(env, to, options.name, options.thread_pool_size,
				options.allow_spinning, allocator);
		}
                }  // namespace concurrency
}  // namespace onnxruntime
namespace OrtApis{
ORT_API_STATUS_IMPL(CreateThreadingOptions, _Outptr_ OrtThreadingOptions** out) {
  *out = new OrtThreadingOptions();
  return nullptr;
}

ORT_API(void, ReleaseThreadingOptions, _Frees_ptr_opt_ OrtThreadingOptions* p) {
  delete p;
}
}