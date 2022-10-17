#pragma once

#include <functional>
#include <list>
#include <memory>
#include <set>
#include <vector>
#include <map>
#include <unordered_map>
#include <deque>
#include <atomic>

#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hcc.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_ext.h>
#include <roctracer/roctracer_roctx.h>


namespace onnxruntime{
namespace profiling {
class RocmProfiler;
}
}

class RoctracerActivityBuffer {
public:
  // data must be allocated using malloc.
  // Ownership is transferred to this object.
  RoctracerActivityBuffer(uint8_t* data, size_t validSize)
      : data_(data), validSize_(validSize) {}

  ~RoctracerActivityBuffer() {
    free(data_);
  }

  // Allocated by malloc
  uint8_t* data_{nullptr};

  // Number of bytes used
  size_t validSize_;
};


class ApiIdList
{
public:
  ApiIdList();
  bool invertMode() { return invert_; }
  void setInvertMode(bool invert) { invert_ = invert; }
  void add(const std::string &apiName);
  void remove(const std::string &apiName);
  bool loadUserPrefs();
  bool contains(uint32_t apiId);
  const std::unordered_map<uint32_t, uint32_t> &filterList() { return filter_; }

private:
  std::unordered_map<uint32_t, uint32_t> filter_;
  bool invert_;
};

struct roctracerRow {
  roctracerRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
             , uint32_t tid, uint64_t begin, uint64_t end)
    : id(id), domain(domain), cid(cid), pid(pid), tid(tid), begin(begin), end(end) {}
  uint64_t id;  // correlation_id
  uint32_t domain;
  uint32_t cid;
  uint32_t pid;
  uint32_t tid;
  uint64_t begin;
  uint64_t end;
};

struct kernelRow : public roctracerRow {
  kernelRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
          , uint32_t tid, uint64_t begin, uint64_t end
          , const void *faddr, hipFunction_t function
          , unsigned int gx, unsigned int gy, unsigned int gz
          , unsigned int wx, unsigned int wy, unsigned int wz
          , size_t gss, hipStream_t stream)
    : roctracerRow(id, domain, cid, pid, tid, begin, end), functionAddr(faddr)
    , function(function), gridX(gx), gridY(gy), gridZ(gz)
    , workgroupX(wx), workgroupY(wy), workgroupZ(wz), groupSegmentSize(gss)
    , stream(stream) {}
  const void* functionAddr;
  hipFunction_t function;
  unsigned int gridX;
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int workgroupX;
  unsigned int workgroupY;
  unsigned int workgroupZ;
  size_t groupSegmentSize;
  hipStream_t stream;
};

struct copyRow : public roctracerRow {
  copyRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
             , uint32_t tid, uint64_t begin, uint64_t end
             , const void* src, const void *dst, size_t size, hipMemcpyKind kind
             , hipStream_t stream)
    : roctracerRow(id, domain, cid, pid, tid, begin, end)
    , src(src), dst(dst), size(size), kind(kind), stream(stream) {}
  const void *src;
  const void *dst;
  size_t size;
  hipMemcpyKind kind;
  hipStream_t stream;
};

struct mallocRow : public roctracerRow {
  mallocRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
             , uint32_t tid, uint64_t begin, uint64_t end
             , const void* ptr, size_t size)
    : roctracerRow(id, domain, cid, pid, tid, begin, end)
    , ptr(ptr), size(size) {}
  const void *ptr;
  size_t size;
};


class RoctracerLogger {
 public:
  enum CorrelationDomain {
    begin,
    Default = begin,
    Domain0 = begin,
    Domain1,
    end,
    size = end
  };

  RoctracerLogger();
  RoctracerLogger(const RoctracerLogger&) = delete;
  RoctracerLogger& operator=(const RoctracerLogger&) = delete;

  virtual ~RoctracerLogger();

  static RoctracerLogger& singleton();

  static void pushCorrelationID(uint64_t id, CorrelationDomain type);
  static void popCorrelationID(CorrelationDomain type);

  void startLogging();
  void stopLogging();
  void clearLogs();

 private:
  bool registered_{false};
  void endTracing();

  roctracer_pool_t *hccPool_{NULL};
  static void api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
  static void activity_callback(const char* begin, const char* end, void* arg);

  ApiIdList loggedIds_;

  // Api callback data
  std::deque<roctracerRow> rows_;
  std::deque<kernelRow> kernelRows_;
  std::deque<copyRow> copyRows_;
  std::deque<mallocRow> mallocRows_;
  std::map<uint64_t,uint64_t> externalCorrelations_[CorrelationDomain::size];	// tracer -> ext

  std::unique_ptr<std::list<RoctracerActivityBuffer>> gpuTraceBuffers_;
  bool externalCorrelationEnabled_{true};

  friend class onnxruntime::profiling::RocmProfiler;
};
