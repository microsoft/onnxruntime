#pragma once
#include "core/graph/basic_types.h"
#include "core/framework/alloc_kind.h"
#include "core/framework/data_types.h"
#include "core/framework/execution_plan_base.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/tensor.h"
#include "core/framework/ort_value_name_idx_map.h"

#include <iomanip>

namespace onnxruntime {
using IntervalT = std::pair<size_t, size_t>;
using OrtValueIndex = int;
using OrtValueName = std::string;
//TODO: need to extend this enum to include finner-grained decomposition
enum MLValueTensorType {
  WEIGHT = 0,
  FWD_ACTIVATION,
  GRADIENT,
  Unknown,
};

struct MemoryInfoPerTensor {
  MemoryBlock planned_block;
  MemoryBlock alloced_block;
};

struct MemoryInfoMap {
 public:
  MemoryInfoMap() = default;

  inline void AddPlannedMemory(const OrtValueIndex& idx, const MemoryBlock& mb) {
    map_[idx].planned_block = mb;
  }

  inline void AddAllocMemory(const OrtValueIndex& idx, MemoryBlock& mb) {
    map_[idx].alloced_block = mb;
    if (ptr_offset == 0 || (ptr_offset > mb.offset_ && mb.offset_ != 0)) {
      ptr_offset = mb.offset_;
    }
  }

  inline const MemoryBlock& GetPlannedMemory(const OrtValueIndex& idx) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    return map_.at(idx).planned_block;
  }

  inline size_t GetPlannedAddress(const OrtValueIndex& idx) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    return map_.at(idx).planned_block.offset_;
  }

  inline size_t GetPlannedSize(const OrtValueIndex& idx) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    return map_.at(idx).planned_block.size_;
  }
  size_t GetAllocAddress(const OrtValueIndex& idx, bool raw = false) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    if (raw) {
      return map_.at(idx).alloced_block.offset_;
    } else {
      return map_.at(idx).alloced_block.offset_ - ptr_offset;
    }
  }

  inline size_t GetAllocSize(const OrtValueIndex& idx) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    return map_.at(idx).alloced_block.size_;
  }

  inline bool Contain(const OrtValueIndex& idx) {
    return map_.find(idx) != map_.end();
  }

  inline void clear() {
    map_.clear();
  }

  std::unordered_map<onnxruntime::OrtValueIndex, onnxruntime::MemoryInfoPerTensor>::iterator begin() { return map_.begin(); }
  std::unordered_map<onnxruntime::OrtValueIndex, onnxruntime::MemoryInfoPerTensor>::const_iterator begin() const { return map_.begin(); }
  std::unordered_map<onnxruntime::OrtValueIndex, onnxruntime::MemoryInfoPerTensor>::iterator end() { return map_.end(); }
  std::unordered_map<onnxruntime::OrtValueIndex, onnxruntime::MemoryInfoPerTensor>::const_iterator end() const { return map_.end(); }
  onnxruntime::MemoryInfoPerTensor& operator[](const OrtValueIndex& k) { return map_[k]; }

 private:
  std::unordered_map<OrtValueIndex, MemoryInfoPerTensor> map_;
  size_t ptr_offset{0};  //The start ptr of the raw data
};

class MemoryInfo {
 public:
  enum MapType {
    Initializer = 0,
    StaticActivation,
    DynamicActivation,
  };

  struct AllocInfoPerTensor {
    AllocInfoPerTensor() = default;

    MLValueTensorType tensor_type{Unknown};
    OrtValueIndex mlvalue_index{0};
    OrtValueName mlvalue_name{""};

    IntervalT lifetime_interval{0, 0};
    IntervalT alloctime_interval{0, 0};
    bool inplace_reuse{false};
    OrtValueIndex reused_buffer{0};  //The index of the reused tensor, if no reuse, it is its own tensor.
    AllocKind alloc_kind{AllocKind::kAllocate};
    OrtMemoryInfo location;
  };

  struct AllocationSummary {
    size_t total_size = 0;
    size_t used_size = 0;
    std::vector<OrtValueIndex> life_tensosrs;
  };

  struct MemoryInfoProfile {
   public:
    static size_t GetAndIncreasePid() {
      size_t val = pid_++;
      return val;
    }
    static std::string CreateMetadataEvent(const std::string& process_name, size_t process_id);
    static std::string CreateMemoryEvent(size_t pid, size_t tid, const std::string& name, size_t offset, size_t size, const std::string& color_name);
    static std::string CreateSummaryEvent(size_t pid, size_t tid, const AllocationSummary& summary, size_t size, size_t bytes_for_pattern);
    static void CreateEvents(const std::string& p_name, const size_t pid, const MemoryInfo::MapType& map_type, const std::string& name_pattern, const size_t top_k);
    static const std::vector<std::string>& GetEvents() { return events; }

   private:
    static size_t pid_;
    static const std::vector<std::string> color_names;
    static std::vector<std::string> events;
    //Key: the hash function of device+map_type. Value: (key: The time step. value: The allocation information)
    static std::unordered_map<size_t, std::unordered_map<size_t, AllocationSummary> > summary_;
    MemoryInfoProfile() = default;
  };

  static void GenerateTensorMap(const SequentialExecutionPlan* execution_plan, const OrtValueNameIdxMap& value_name_idx_map);
  static void RecordInitializerPatternInfo(const MemoryPatternGroup& mem_patterns);
  static void RecordActivationPatternInfo(const MemoryPatternGroup& mem_patterns);

  static void RecordInitializerAllocInfo(const std::unordered_map<int, OrtValue>& tensor_map);
  static void RecordActivationAllocInfo(const OrtValueIndex idx, const OrtValue& value);
  static void SetDynamicAllocation(const OrtValueIndex idx);
  static void SetLocalRank(const int rank) { local_rank_ = rank; }

  static inline void SetIteration(size_t iteration) { iteration_ = iteration; }

  static void PrintMemoryInfoForLocation(const logging::Logger& /*logger*/, const OrtDevice::DeviceType location);
  static void GenerateMemoryProfile();
  static inline void ClearMemoryInfoPerExecution() {
    for (auto& location_map : tensors_memory_info_map_) {
      location_map.second[MapType::DynamicActivation].clear();
      location_map.second[MapType::StaticActivation].clear();
    }
  }
  static const AllocInfoPerTensor* AllocPlan(const OrtValueIndex& idx) {
    if (tensor_alloc_info_map_.find(idx) != tensor_alloc_info_map_.end())
      return &tensor_alloc_info_map_.at(idx);
    else
      return nullptr;
  }

  static void AddRecordingTensorGroup(const std::string& group_name, const std::string& tensor_name) {
    customized_recording_group_[group_name][tensor_name] = true;
  }

  static const bool InRecordingTensorGroup(const std::string& group_name, const std::string& tensor_name) {
      if (customized_recording_group_.find(group_name) == customized_recording_group_.end()) return false;
      if (customized_recording_group_.at(group_name).find(tensor_name) == customized_recording_group_.at(group_name).end()) return false;
      return true;
  }
 private:
  //MemoryInfo(int local_rank) : profiler(*this), iteration_(0), local_rank_(local_rank) {}
  MemoryInfo() = default;
  static void RecordMemoryPatternInfo(const MemoryPatternGroup& mem_patterns, MapType type);
  static void RecordTensorDeviceAllocInfo(const OrtValueIndex idx, const OrtValue& value, const MapType& type);

  static bool IsInplaceReuse(const OrtValueIndex& idx) {
    return tensor_alloc_info_map_[idx].inplace_reuse;
  }

  //Key: The map type. E.g., initializer, activation. Value: A map from the tensor index to its memory information
  static std::map<OrtMemoryInfo, std::map<MapType, MemoryInfoMap> > tensors_memory_info_map_;

  static std::map<OrtValueIndex, AllocInfoPerTensor> tensor_alloc_info_map_;

  static std::map<const std::string, std::map<const std::string, bool> >customized_recording_group_;

  //TODO: The dynamic and statically planned alignments may not be the same, need to check
  static const int alignment = 256;
  static size_t iteration_;
  static size_t num_node_size_;
  static int local_rank_;
  //Memory Profile
  static std::map<MapType, std::set<size_t> > time_step_trace_;
};

}  // namespace onnxruntime
