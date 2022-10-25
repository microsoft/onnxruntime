#include "RoctracerLogger.h"

#include <time.h>
#include <cstring>
#include <chrono>

#include "ThreadUtil.h"

typedef uint64_t timestamp_t;

static timestamp_t timespec_to_ns(const timespec& time) {
    return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
  }

//using namespace std::chrono;

RoctracerLogger& RoctracerLogger::singleton() {
  static RoctracerLogger instance;
  return instance;
}

RoctracerLogger::RoctracerLogger() {
  gpuTraceBuffers_ = std::make_unique<std::list<RoctracerActivityBuffer>>();
}

RoctracerLogger::~RoctracerLogger() {
  stopLogging();
  endTracing();
}

namespace {
  thread_local std::deque<uint64_t> t_externalIds[RoctracerLogger::CorrelationDomain::size];
}

void RoctracerLogger::pushCorrelationID(uint64_t id, CorrelationDomain type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  t_externalIds[type].push_back(id);
}

void RoctracerLogger::popCorrelationID(CorrelationDomain type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  t_externalIds[type].pop_back();
}

void RoctracerLogger::clearLogs() {
  rows_.clear();
  kernelRows_.clear();
  copyRows_.clear();
  mallocRows_.clear();
  gpuTraceBuffers_->clear();
  for (int i = 0; i < CorrelationDomain::size; ++i) {
    externalCorrelations_[i].clear();
  }
}

void RoctracerLogger::api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
  RoctracerLogger *dis = &singleton();

  if (domain == ACTIVITY_DOMAIN_HIP_API && dis->loggedIds_.contains(cid)) {
    const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);

    // Pack callbacks into row structures

    thread_local timespec timestamp;

    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      clock_gettime(CLOCK_MONOTONIC, &timestamp);  // record proper clock
    } else {  // (data->phase == ACTIVITY_API_PHASE_EXIT)
      timespec endTime;
      timespec startTime { timestamp };
      clock_gettime(CLOCK_MONOTONIC, &endTime);  // record proper clock

      switch (cid) {
        case HIP_API_ID_hipLaunchKernel:
        case HIP_API_ID_hipExtLaunchKernel:
        case HIP_API_ID_hipLaunchCooperativeKernel:     // Should work here
          {
          auto &args = data->args.hipLaunchKernel;
          dis->kernelRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.function_address,
                              nullptr,
                              args.numBlocks.x,
                              args.numBlocks.y,
                              args.numBlocks.z,
                              args.dimBlocks.x,
                              args.dimBlocks.y,
                              args.dimBlocks.z,
                              args.sharedMemBytes,
                              args.stream
                            );
          }
          break;
        case HIP_API_ID_hipHccModuleLaunchKernel:
        case HIP_API_ID_hipModuleLaunchKernel:
        case HIP_API_ID_hipExtModuleLaunchKernel:
          {
          auto &args = data->args.hipModuleLaunchKernel;
          dis->kernelRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              nullptr,
                              args.f,
                              args.gridDimX,
                              args.gridDimY,
                              args.gridDimZ,
                              args.blockDimX,
                              args.blockDimY,
                              args.blockDimZ,
                              args.sharedMemBytes,
                              args.stream
                            );
          }
          break;
        case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
        case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
#if 0
          {
            auto &args = data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList__val;
            dis->kernelRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.function_address,
                              nullptr,
                              args.numBlocks.x,
                              args.numBlocks.y,
                              args.numBlocks.z,
                              args.dimBlocks.x,
                              args.dimBlocks.y,
                              args.dimBlocks.z,
                              args.sharedMemBytes,
                              args.stream
                            );
          }
#endif
          break;
        case HIP_API_ID_hipMalloc:
            dis->mallocRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              data->args.hipMalloc.ptr__val,
                              data->args.hipMalloc.size
                              );
          break;
        case HIP_API_ID_hipFree:
            dis->mallocRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              data->args.hipFree.ptr,
                              0
                              );
          break;
        case HIP_API_ID_hipMemcpy:
          {
            auto &args = data->args.hipMemcpy;
            dis->copyRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.src,
                              args.dst,
                              args.sizeBytes,
                              args.kind,
                              static_cast<hipStream_t>(0)  // use placeholder?
                              );
          }
          break;
        case HIP_API_ID_hipMemcpyAsync:
        case HIP_API_ID_hipMemcpyWithStream:
          {
            auto &args = data->args.hipMemcpyAsync;
            dis->copyRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.src,
                              args.dst,
                              args.sizeBytes,
                              args.kind,
                              args.stream
                              );
          }
          break;
        default:
          dis->rows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime)
                              );
          break;
      }  // switch
      // External correlation
      for (int it = CorrelationDomain::begin; it < CorrelationDomain::end; ++it) {
        if (t_externalIds[it].size() > 0) {
          dis->externalCorrelations_[it][data->correlation_id] = t_externalIds[it].back();
        }
      }
    }  // phase exit
  }
}

void RoctracerLogger::activity_callback(const char* begin, const char* end, void* arg)
{
  size_t size = end - begin;
  uint8_t *buffer = (uint8_t*) malloc(size);
  auto &gpuTraceBuffers = singleton().gpuTraceBuffers_;
  memcpy(buffer, begin, size);
  gpuTraceBuffers->emplace_back(buffer, size);
}

void RoctracerLogger::startLogging() {
  if (!registered_) {
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);  // Magic encantation

    // Set some api calls to ignore
    loggedIds_.setInvertMode(true);  // Omit the specified api
    loggedIds_.add("hipGetDevice");
    loggedIds_.add("hipSetDevice");
    loggedIds_.add("hipGetLastError");
    loggedIds_.add("__hipPushCallConfiguration");
    loggedIds_.add("__hipPopCallConfiguration");
    loggedIds_.add("hipCtxSetCurrent");
    loggedIds_.add("hipEventRecord");
    loggedIds_.add("hipEventQuery");
    loggedIds_.add("hipGetDeviceProperties");
    loggedIds_.add("hipPeekAtLastError");
    loggedIds_.add("hipModuleGetFunction");
    loggedIds_.add("hipEventCreateWithFlags");

    // Enable API callbacks
    if (loggedIds_.invertMode() == true) {
        // exclusion list - enable entire domain and turn off things in list
        roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, nullptr);
        const std::unordered_map<uint32_t, uint32_t> &filter = loggedIds_.filterList();
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first);
        }
    }
    else {
        // inclusion list - only enable things in the list
        const std::unordered_map<uint32_t, uint32_t> &filter = loggedIds_.filterList();
        roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first, api_callback, nullptr);
        }
    }
    //roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, nullptr);

    // Allocate default tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    roctracer_open_pool(&properties);

    // Enable async op collection
    roctracer_properties_t hcc_cb_properties;
    memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
    hcc_cb_properties.buffer_size = 0x4000;
    hcc_cb_properties.buffer_callback_fun = activity_callback;
    roctracer_open_pool_expl(&hcc_cb_properties, &hccPool_);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, hccPool_);

    registered_ = true;
  }

  externalCorrelationEnabled_ = true;
  roctracer_start();
}

void RoctracerLogger::stopLogging() {
  roctracer_stop();
  roctracer_flush_activity_expl(hccPool_);
}

void RoctracerLogger::endTracing() {
  if (registered_ == true) {
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    //roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);

    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS);
    roctracer_close_pool_expl(hccPool_);
  }
}


ApiIdList::ApiIdList()
: invert_(true)
{
}

void ApiIdList::add(const std::string &apiName)
{
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, nullptr) == ROCTRACER_STATUS_SUCCESS) {
    filter_[cid] = 1;
  }
}
void ApiIdList::remove(const std::string &apiName)
{
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, nullptr) == ROCTRACER_STATUS_SUCCESS) {
    filter_.erase(cid);
  }
}

bool ApiIdList::loadUserPrefs()
{
  // placeholder
  return false;
}
bool ApiIdList::contains(uint32_t apiId)
{
  return (filter_.find(apiId) != filter_.end()) ? !invert_ : invert_;  // XOR
}
