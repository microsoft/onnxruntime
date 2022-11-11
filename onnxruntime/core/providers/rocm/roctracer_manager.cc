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
};

// Implementation of RoctracerActivityBuffer
RoctracerActivityBuffer& RoctracerActivityBuffer::operator=(const RoctracerActivityBuffer& other) {
  if (&other == this) {
    return *this;
  }

  size_ = other.size_;
  data_ = std::make_unique<char[]>(other.size_);
  memcpy(data_.get(), other.data_.get(), size_);
  return *this;
}

RoctracerActivityBuffer& RoctracerActivityBuffer::operator=(RoctracerActivityBuffer&& other) {
  if (&other == this) {
    return *this;
  }
  std::swap(data_, other.data_);
  std::swap(size_, other.size_);
  return *this;
}

// Implementation of RoctracerManager
RoctracerManager& RoctracerManager::GetInstance() {
  static RoctracerManager instance;
  return instance;
}

RoctracerManager::~RoctracerManager() {
  StopLogging();
}

uint64_t RoctracerManager::RegisterClient() {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);
  auto res = next_client_id_++;
  per_client_events_by_ext_correlation_.insert({res, {}});
  ++num_active_clients_;
  return res;
}

void RoctracerManager::DeregisterClient(uint64_t client_handle) {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);
  per_client_events_by_ext_correlation_.erase(client_handle);
  --num_active_clients_;
  if (num_active_clients_ == 0) {
    StopLogging();
  }
}

void RoctracerManager::StartLogging() {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);
  if (logging_enabled_) {
    return;
  }

  // The following line shows up in all the samples, I do not know
  // what the point is, but without it, the roctracer APIs don't work.
  roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);

  roctracer_properties_t hcc_cb_properties;
  memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
  hcc_cb_properties.buffer_size = kActivityBufferSize;
  hcc_cb_properties.buffer_callback_fun = ActivityCallback;
  roctracer_open_pool(&hcc_cb_properties);

  // Enable selective activity and API callbacks for the HIP APIs
  roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
  roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API);

  for (auto const& logged_api : hip_api_calls_to_trace) {
    uint32_t cid = 0;
    roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, logged_api.c_str(), &cid, nullptr);
    roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, cid, ApiCallback, nullptr);
    roctracer_enable_op_activity(ACTIVITY_DOMAIN_HIP_API, cid);
  }

  // Enable activity logging in the HIP_OPS/HCC_OPS domain.
  roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);

  roctracer_start();
  logging_enabled_ = true;
}

// Requires: roctracer_manager_mutex_ must be held
void RoctracerManager::Clear() {
  unprocessed_activity_buffers_.clear();
  api_call_args_.clear();
  unique_correlation_id_to_client_offset_.clear();
  roctracer_correlation_to_unique_correlation_.clear();
  per_client_events_by_ext_correlation_.clear();
}

// Requires: roctracer_manager_mutex_ must be held
void RoctracerManager::StopLogging() {
  if (!logging_enabled_) {
    return;
  }

  roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API);
  roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
  roctracer_stop();
  roctracer_flush_activity();
  roctracer_close_pool();

  logging_enabled_ = false;
  Clear();
}

void RoctracerManager::Consume(uint64_t client_handle, const TimePoint& start_time,
                               std::map<uint64_t, Events>& events) {
  events.clear();
  {
    // Flush any pending activity records before starting
    // to process the accumulated activity records.
    std::lock_guard<std::mutex> lock_manager(roctracer_manager_mutex_);
    roctracer_flush_activity();
  }

  std::vector<RoctracerActivityBuffer> activity_buffers;
  {
    std::lock_guard<std::mutex> lock(unprocessed_activity_buffers_lock_);
    std::swap(unprocessed_activity_buffers_, activity_buffers);
    unprocessed_activity_buffers_.clear();
  }

  {
    // Ensure that at most one thread is working through the activity buffers at any time.
    std::lock_guard<std::mutex> lock_two(activity_buffer_processor_mutex_);
    ProcessActivityBuffers(activity_buffers, start_time);
    auto it = per_client_events_by_ext_correlation_.find(client_handle);
    if (it == per_client_events_by_ext_correlation_.end()) {
      return;
    }
    std::swap(events, it->second);
  }
}

bool RoctracerManager::PushCorrelation(uint64_t client_handle,
                                       uint64_t external_correlation_id,
                                       TimePoint profiling_start_time) {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);

  auto it = per_client_events_by_ext_correlation_.find(client_handle);
  if (it == per_client_events_by_ext_correlation_.end()) {
    // not a registered client, do nothing
    return false;
  }

  // external_correlation_id is simply the timestamp of this event,
  // relative to profiling_start_time. i.e., it was computed as:
  // external_correlation_id =
  //      std::chrono::duration_cast<std::chrono::microseconds>(event_start_time - profiling_start_time).count()
  //
  // Because of the relative nature of the external_correlation_id, the same
  // external_correlation_id can be reused across different clients, which then makes it
  // impossible to recover the client from the external_correlation_id, which in turn
  // makes it impossible to map events (which are tagged with external_correlation_id) to clients.
  //
  // To address these difficulties, we construct a new correlation_id (let's call it unique_cid)
  // as follows:
  // unique_cid =
  //    external_correlation_id +
  //    std::chrono::duration_cast<std::chrono::microseconds>(profiling_start_time.time_since_epoch()).count()
  // now, unique_cid is monotonically increasing with time, so it can be used to reliably map events to clients.
  //
  // Of course, clients expect lists of events to be returned (on a call to Consume()), that are
  // still keyed on the external_correlation_id that they've specified here, so we need to remember the
  // offset to be subtracted

  uint64_t offset =
      std::chrono::duration_cast<std::chrono::microseconds>(profiling_start_time.time_since_epoch()).count();
  auto unique_cid = external_correlation_id + offset;
  roctracer_activity_push_external_correlation_id(unique_cid);

  unique_correlation_id_to_client_offset_[unique_cid] = std::make_pair(client_handle, offset);
  return true;
}

void RoctracerManager::PopCorrelation(uint64_t& popped_external_correlation_id) {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);
  uint64_t unique_cid;
  roctracer_activity_pop_external_correlation_id(&unique_cid);
  // lookup the offset and subtract it before returning popped_external_correlation_id to the client
  auto client_it = unique_correlation_id_to_client_offset_.find(unique_cid);
  if (client_it == unique_correlation_id_to_client_offset_.end()) {
    popped_external_correlation_id = 0;
    return;
  }
  popped_external_correlation_id = unique_cid - client_it->second.second;
}

void RoctracerManager::ActivityCallback(const char* begin, const char* end, void* arg) {
  size_t size = end - begin;
  RoctracerActivityBuffer activity_buffer{reinterpret_cast<const char*>(begin), size};
  auto& instance = GetInstance();
  {
    std::lock_guard<std::mutex> lock(instance.unprocessed_activity_buffers_lock_);
    instance.unprocessed_activity_buffers_.emplace_back(std::move(activity_buffer));
  }
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
    std::lock_guard<std::mutex> lock(instance.api_call_args_lock_);
    auto& record = instance.api_call_args_[data->correlation_id];
    record.domain_ = domain;
    record.cid_ = cid;
    record.api_data_ = *data;
  }
}

static inline std::string PointerToHexString(const void* ptr) {
  std::ostringstream sstr;
  sstr << std::hex << ptr;
  return sstr.str();
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

    default:
      return false;
  }

  new (&event) EventRecord{
      /* cat = */ EventCategory::KERNEL_EVENT,
      /* pid = */ -1,
      /* tid = */ -1,
      /* name = */ std::move(name),
      /* ts = */ (int64_t)(record->begin_ns - start_time_ns) / 1000,
      /* dur = */ (int64_t)(record->end_ns - record->begin_ns) / 1000,
      /* args = */ std::move(args)};
  return true;
}

Events* RoctracerManager::GetEventListForUniqueCorrelationId(uint64_t unique_correlation_id) {
  auto client_it = unique_correlation_id_to_client_offset_.find(unique_correlation_id);
  if (client_it == unique_correlation_id_to_client_offset_.end()) {
    // :-( well, we tried really, really hard to map this event to a client.
    return nullptr;
  }

  // See the comments on the PushCorrelation method for an explanation of
  // of this offset computation and why it's required.
  auto const& client_handle_offset = client_it->second;
  auto external_correlation = unique_correlation_id - client_handle_offset.second;

  auto& event_list = per_client_events_by_ext_correlation_[client_handle_offset.first][external_correlation];
  return &event_list;
}

void RoctracerManager::MapEventsToClient(uint64_t unique_correlation_id, std::vector<EventRecord>&& events) {
  auto p_event_list = GetEventListForUniqueCorrelationId(unique_correlation_id);
  if (p_event_list != nullptr) {
    p_event_list->insert(p_event_list->end(),
                         std::make_move_iterator(events.begin()),
                         std::make_move_iterator(events.end()));
  }
}

void RoctracerManager::MapEventToClient(uint64_t unique_correlation_id, EventRecord&& event) {
  auto p_event_list = GetEventListForUniqueCorrelationId(unique_correlation_id);
  if (p_event_list != nullptr) {
    p_event_list->emplace_back(std::move(event));
  }
}

void RoctracerManager::ProcessActivityBuffers(const std::vector<RoctracerActivityBuffer>& buffers,
                                              const TimePoint& start_time) {
  auto start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count();

  for (auto const& buffer : buffers) {
    auto current_record = reinterpret_cast<const roctracer_record_t*>(buffer.GetData());
    auto data_end = reinterpret_cast<const roctracer_record_t*>(buffer.GetData() + buffer.GetSize());
    for (; current_record < data_end; roctracer_next_record(current_record, &current_record)) {
      EventRecord event;
      if (current_record->domain == ACTIVITY_DOMAIN_EXT_API) {
        roctracer_correlation_to_unique_correlation_[current_record->correlation_id] = current_record->external_id;

        // check for any events pending client mapping on this correlation
        auto pending_it = events_pending_client_mapping_.find(current_record->correlation_id);
        if (pending_it == events_pending_client_mapping_.end()) {
          continue;
        }

        // we have one or more pending events, map them to the client
        MapEventsToClient(current_record->external_id, std::move(pending_it->second));
        events_pending_client_mapping_.erase(pending_it);
        // no additional events to be mapped for this record
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
      auto ext_corr_it = roctracer_correlation_to_unique_correlation_.find(current_record->correlation_id);
      if (ext_corr_it == roctracer_correlation_to_unique_correlation_.end()) {
        // defer the processing of this event
        events_pending_client_mapping_[current_record->correlation_id].emplace_back(std::move(event));
        continue;
      }
      MapEventToClient(ext_corr_it->second, std::move(event));
    }
  }
}

} /* end namespace profiling */
} /* end namespace onnxruntime */
