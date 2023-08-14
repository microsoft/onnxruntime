#include <time.h>
#include <cstring>
#include <chrono>

#include "roctracer_manager.h"

namespace onnxruntime {
namespace profiling {

// allocate a 16K buffer for recording async activities
static constexpr size_t kActivityBufferSize = 0x4000;

const std::vector<std::string> RoctracerManager::hip_api_calls_to_trace = {
    "hipMemcpy",
    "hipMemcpy2D",
    "hipMemcpyAsync",
    "hipMemcpy2DAsync",
    "hipMemcpyWithStream",
    "hipLaunchKernel",
    "hipMemset",
    "hipMemsetAsync",
    "hipExtModuleLaunchKernel",
    "hipExtLaunchKernel",
};

// Implementation of RoctracerManager
RoctracerManager& RoctracerManager::GetInstance() {
  static RoctracerManager instance;
  return instance;
}

RoctracerManager::~RoctracerManager() {}

#define ROCTRACER_STATUS_RETURN_FALSE_ON_FAIL(expr_) \
  do {                                               \
    if (expr_ != ROCTRACER_STATUS_SUCCESS) {         \
      OnStopLogging();                               \
      return false;                                  \
    }                                                \
  } while (false)

bool RoctracerManager::OnStartLogging() {
  // The following line shows up in all the samples, I do not know
  // what the point is, but without it, the roctracer APIs don't work.
  roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);

  roctracer_properties_t hcc_cb_properties;
  memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
  hcc_cb_properties.buffer_size = kActivityBufferSize;
  hcc_cb_properties.buffer_callback_fun = ActivityCallback;
  ROCTRACER_STATUS_RETURN_FALSE_ON_FAIL(roctracer_open_pool(&hcc_cb_properties));

  // Enable selective activity and API callbacks for the HIP APIs
  ROCTRACER_STATUS_RETURN_FALSE_ON_FAIL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
  ROCTRACER_STATUS_RETURN_FALSE_ON_FAIL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));

  for (auto const& logged_api : hip_api_calls_to_trace) {
    uint32_t cid = 0;
    ROCTRACER_STATUS_RETURN_FALSE_ON_FAIL(
        roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, logged_api.c_str(), &cid, nullptr));
    ROCTRACER_STATUS_RETURN_FALSE_ON_FAIL(
        roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, cid, ApiCallback, nullptr));
    ROCTRACER_STATUS_RETURN_FALSE_ON_FAIL(
        roctracer_enable_op_activity(ACTIVITY_DOMAIN_HIP_API, cid));
  }

  // Enable activity logging in the HIP_OPS/HCC_OPS domain.
  ROCTRACER_STATUS_RETURN_FALSE_ON_FAIL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));

  roctracer_start();
  return true;
}

void RoctracerManager::OnStopLogging() {
  roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API);
  roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
  roctracer_stop();
  roctracer_flush_activity();
  roctracer_close_pool();
  api_call_args_.clear();
}

void RoctracerManager::ActivityCallback(const char* begin, const char* end, void* arg) {
  size_t size = end - begin;
  ProfilerActivityBuffer activity_buffer{reinterpret_cast<const char*>(begin), size};
  auto& instance = GetInstance();
  instance.EnqueueActivityBuffer(std::move(activity_buffer));
}

void RoctracerManager::ApiCallback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
  if (domain != ACTIVITY_DOMAIN_HIP_API) {
    return;
  }
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  if (data->phase == ACTIVITY_API_PHASE_EXIT) {
    // We only save args for async launches on the ACTIVITY_API_PHASE_ENTER phase
    return;
  }

  auto& instance = GetInstance();
  {
    std::lock_guard<std::mutex> lock(instance.api_call_args_mutex_);
    auto& record = instance.api_call_args_[data->correlation_id];
    record.domain_ = domain;
    record.cid_ = cid;
    record.api_data_ = *data;
  }
}

bool RoctracerManager::PushUniqueCorrelation(uint64_t unique_cid) {
  return roctracer_activity_push_external_correlation_id(unique_cid) == ROCTRACER_STATUS_SUCCESS;
}

void RoctracerManager::PopUniqueCorrelation(uint64_t& popped_unique_cid) {
  if (roctracer_activity_pop_external_correlation_id(&popped_unique_cid) != ROCTRACER_STATUS_SUCCESS) {
    popped_unique_cid = 0;
  }
}

void RoctracerManager::FlushActivities() {
  roctracer_flush_activity();
}

uint64_t RoctracerManager::GetGPUTimestampInNanoseconds() {
  uint64_t result;
  if (roctracer_get_timestamp(&result) != ROCTRACER_STATUS_SUCCESS) {
    ORT_THROW("Could not retrieve timestamp from GPU!");
  }
  return result;
}

static inline std::string MemcpyKindToString(hipMemcpyKind kind) {
  switch (kind) {
    case hipMemcpyHostToHost:
      return "H2H";
    case hipMemcpyHostToDevice:
      return "H2D";
    case hipMemcpyDeviceToHost:
      return "D2H";
    case hipMemcpyDeviceToDevice:
      return "D2D";
    default:
      return "Default";
  }
}

bool RoctracerManager::CreateEventForActivityRecord(const roctracer_record_t* record,
                                                    uint64_t start_time_ns,
                                                    const ApiCallRecord& call_record,
                                                    EventRecord& event) {
  std::string name;
  std::unordered_map<std::string, std::string> args;

  switch (call_record.cid_) {
    case HIP_API_ID_hipLaunchKernel: {
      auto const& launch_args = call_record.api_data_.args.hipLaunchKernel;
      name = demangle(hipKernelNameRefByPtr(launch_args.function_address,
                                            launch_args.stream));

      args = {
          {"stream", PointerToHexString((void*)(launch_args.stream))},
          {"grid_x", std::to_string(launch_args.numBlocks.x)},
          {"grid_y", std::to_string(launch_args.numBlocks.y)},
          {"grid_z", std::to_string(launch_args.numBlocks.z)},
          {"block_x", std::to_string(launch_args.dimBlocks.x)},
          {"block_y", std::to_string(launch_args.dimBlocks.y)},
          {"block_z", std::to_string(launch_args.dimBlocks.z)}};
      break;
    }

    case HIP_API_ID_hipMemset:
    case HIP_API_ID_hipMemsetAsync: {
      auto const& launch_args = call_record.api_data_.args;
      name = roctracer_op_string(call_record.domain_, call_record.cid_, 0);

      args = {
          {"stream", call_record.cid_ == HIP_API_ID_hipMemset
                         ? "0"
                         : PointerToHexString((void*)launch_args.hipMemsetAsync.stream)},
          {"dst", PointerToHexString(launch_args.hipMemset.dst)},
          {"size", std::to_string(launch_args.hipMemset.sizeBytes)},
          {"value", std::to_string(launch_args.hipMemset.value)}};
      break;
    }

    case HIP_API_ID_hipMemcpy:
    case HIP_API_ID_hipMemcpyAsync:
    case HIP_API_ID_hipMemcpyWithStream: {
      auto const& launch_args = call_record.api_data_.args;
      name = roctracer_op_string(call_record.domain_, call_record.cid_, 0);

      args = {
          {"stream", call_record.cid_ == HIP_API_ID_hipMemcpy
                         ? "0"
                         : PointerToHexString((void*)launch_args.hipMemcpyAsync.stream)},
          {"src", PointerToHexString(launch_args.hipMemcpy.src)},
          {"dst", PointerToHexString(launch_args.hipMemcpy.dst)},
          {"kind", MemcpyKindToString(launch_args.hipMemcpy.kind)}};
      break;
    }

    case HIP_API_ID_hipMemcpy2D:
    case HIP_API_ID_hipMemcpy2DAsync: {
      auto const& launch_args = call_record.api_data_.args;
      name = roctracer_op_string(call_record.domain_, call_record.cid_, 0);

      args = {
          {"stream", call_record.cid_ == HIP_API_ID_hipMemcpy2D
                         ? "0"
                         : PointerToHexString((void*)launch_args.hipMemcpy2DAsync.stream)},
          {"src", PointerToHexString(launch_args.hipMemcpy2D.src)},
          {"dst", PointerToHexString(launch_args.hipMemcpy2D.dst)},
          {"spitch", std::to_string(launch_args.hipMemcpy2D.spitch)},
          {"dpitch", std::to_string(launch_args.hipMemcpy2D.dpitch)},
          {"width", std::to_string(launch_args.hipMemcpy2D.width)},
          {"height", std::to_string(launch_args.hipMemcpy2D.height)},
          {"kind", MemcpyKindToString(launch_args.hipMemcpy2D.kind)}};
      break;
    }

    case HIP_API_ID_hipExtModuleLaunchKernel: {
      auto const& launch_args = call_record.api_data_.args.hipExtModuleLaunchKernel;
      name = demangle(hipKernelNameRef(launch_args.f));

      args = {
          {"stream", PointerToHexString((void*)launch_args.hStream)},
          {"grid_x", std::to_string(launch_args.globalWorkSizeX)},
          {"grid_y", std::to_string(launch_args.globalWorkSizeY)},
          {"grid_z", std::to_string(launch_args.globalWorkSizeZ)},
          {"block_x", std::to_string(launch_args.localWorkSizeX)},
          {"block_y", std::to_string(launch_args.localWorkSizeY)},
          {"block_z", std::to_string(launch_args.localWorkSizeZ)},
      };
      break;
    }

    case HIP_API_ID_hipExtLaunchKernel: {
      auto const& launch_args = call_record.api_data_.args.hipExtLaunchKernel;
      name = demangle(hipKernelNameRefByPtr(launch_args.function_address,
                                            launch_args.stream));

      args = {
          {"stream", PointerToHexString((void*)(launch_args.stream))},
          {"grid_x", std::to_string(launch_args.numBlocks.x)},
          {"grid_y", std::to_string(launch_args.numBlocks.y)},
          {"grid_z", std::to_string(launch_args.numBlocks.z)},
          {"block_x", std::to_string(launch_args.dimBlocks.x)},
          {"block_y", std::to_string(launch_args.dimBlocks.y)},
          {"block_z", std::to_string(launch_args.dimBlocks.z)}};
      break;
    }

    default:
      return false;
  }

  new (&event) EventRecord{
      /* cat = */ EventCategory::KERNEL_EVENT,
      /* pid = */ -1,
      /* tid = */ -1,
      /* name = */ std::move(name),
      /* ts = */ (int64_t)(this->NormalizeGPUTimestampToCPUEpoch(record->begin_ns) - start_time_ns) / 1000,
      /* dur = */ (int64_t)(record->end_ns - record->begin_ns) / 1000,
      /* args = */ std::move(args)};
  return true;
}

void RoctracerManager::ProcessActivityBuffers(const std::vector<ProfilerActivityBuffer>& buffers,
                                              const TimePoint& start_time) {
  auto start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count();

  for (auto const& buffer : buffers) {
    auto current_record = reinterpret_cast<const roctracer_record_t*>(buffer.GetData());
    auto data_end = reinterpret_cast<const roctracer_record_t*>(buffer.GetData() + buffer.GetSize());
    for (; current_record < data_end; roctracer_next_record(current_record, &current_record)) {
      EventRecord event;
      if (current_record->domain == ACTIVITY_DOMAIN_EXT_API) {
        NotifyNewCorrelation(current_record->correlation_id, current_record->external_id);
        continue;
      } else if (current_record->domain == ACTIVITY_DOMAIN_HIP_OPS) {
        if (current_record->op == 1 && current_record->kind == HipOpMarker) {
          // this is just a marker, ignore it.
          continue;
        }

        auto api_it = api_call_args_.find(current_record->correlation_id);
        if (api_it == api_call_args_.end()) {
          // we're not tracking this activity, ignore it
          continue;
        }

        auto const& call_record = api_it->second;
        if (!CreateEventForActivityRecord(current_record, start_time_ns, call_record, event)) {
          // No event created, skip to the next record to avoid associating an empty
          // event with a client
          continue;
        }
      } else {
        // ignore the superfluous event: this is probably a HIP API callback, which
        // we've had to enable to receive external correlation ids
        continue;
      }
      // map the event to the right client
      MapEventToClient(current_record->correlation_id, std::move(event));
    }
  }
}

} /* end namespace profiling */
} /* end namespace onnxruntime */
