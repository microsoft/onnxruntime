#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
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
using OrtValueIndex = int;
using OrtValueName = std::string;

struct MemoryInfoPerTensor {
  MemoryBlock planned_block;
  MemoryBlock allocated_block;
};

struct MemoryInfoMap {
  using MemoryInfoMapT = std::unordered_map<onnxruntime::OrtValueIndex, onnxruntime::MemoryInfoPerTensor>;

 public:
  MemoryInfoMap() = default;

  void AddPlannedMemory(const OrtValueIndex& idx, const MemoryBlock& mb) {
    map_[idx].planned_block = mb;
  }

  void AddAllocMemory(const OrtValueIndex& idx, MemoryBlock& mb) {
    map_[idx].allocated_block = mb;
    if (ptr_offset == 0 || (ptr_offset > mb.offset_ && mb.offset_ != 0)) {
      ptr_offset = mb.offset_;
    }
  }

  const MemoryBlock& GetPlannedMemory(const OrtValueIndex& idx) const {
    auto itr = map_.find(idx);
    ORT_ENFORCE(itr != map_.end());
    return itr->second.planned_block;
  }

  size_t GetPlannedAddress(const OrtValueIndex& idx) const {
    auto itr = map_.find(idx);
    ORT_ENFORCE(itr != map_.end());
    return itr->second.planned_block.offset_;
  }

  size_t GetPlannedSize(const OrtValueIndex& idx) const {
    auto itr = map_.find(idx);
    ORT_ENFORCE(itr != map_.end());
    return itr->second.planned_block.size_;
  }
  size_t GetAllocAddress(const OrtValueIndex& idx, bool raw = false) const {
    auto itr = map_.find(idx);
    ORT_ENFORCE(itr != map_.end());
    if (raw) {
      return itr->second.allocated_block.offset_;
    } else {
      return itr->second.allocated_block.offset_ - ptr_offset;
    }
  }

  size_t GetAllocSize(const OrtValueIndex& idx) const {
    auto itr = map_.find(idx);
    ORT_ENFORCE(itr != map_.end());
    return itr->second.allocated_block.size_;
  }

  bool Contain(const OrtValueIndex& idx) const {
    return map_.find(idx) != map_.end();
  }

  void clear() {
    map_.clear();
  }

  MemoryInfoMapT::iterator begin() { return map_.begin(); }
  MemoryInfoMapT::const_iterator begin() const { return map_.begin(); }
  MemoryInfoMapT::iterator end() { return map_.end(); }
  MemoryInfoMapT::const_iterator end() const { return map_.end(); }
  onnxruntime::MemoryInfoPerTensor& operator[](const OrtValueIndex& k) { return map_[k]; }

 private:
  MemoryInfoMapT map_;
  size_t ptr_offset{0};  //The start ptr of the raw data
};

class MemoryInfo {
 public:
  //We separate the memory types into 3 categories.
  //(1) Initializers: Which are planned and allocated before session run.
  //(2) StaticActivation: Which are planned statically and allocatedly accordingly
  //(3) DynamicActivation: Which are allocated during runtime.
  //The reason with this separations is they start with different memory offsets, so diffcult to visualize them in the same session.
  enum MapType {
    Initializer = 0,
    StaticActivation,
    DynamicActivation,
  };

  struct AllocInfoPerTensor {
    AllocInfoPerTensor() = default;

    OrtValueIndex mlvalue_index{0};
    OrtValueName mlvalue_name{""};

    IntervalT lifetime_interval{0, 0};
    bool inplace_reuse{false};
    OrtValueIndex reused_buffer{0};  //The index of the reused tensor, if no reuse, it is its own tensor.
    AllocKind alloc_kind{AllocKind::kAllocate};
    OrtMemoryInfo location;
  };

  struct AllocationSummary {
    size_t total_size = 0;
    size_t used_size = 0;
    std::vector<OrtValueIndex> live_tensors;
  };

  struct MemoryInfoProfile {
   public:
    //Create sessions in the profiler.
    //p_name: session name
    //pid: sessionid
    //map_type: Initalizer, static_activation, dynamic_activation. We have this separtion because they are using different memory offsets.
    //group_name: The group_name that is recorded previously using function "AddRecordingTensorGroup". Used for generating customized sessions for a group of tensors
    //Top_k: The steps with the top-k highest memory consumptions are plot. When top_k == 0, we plot all the steps
    //device_t: The type of the device where the tensors are.
    static void CreateEvents(const std::string& p_name, const size_t pid, const MemoryInfo::MapType& map_type, const std::string& group_name,
                             const size_t top_k, const OrtDevice::DeviceType device_t = OrtDevice::GPU);

    static const std::vector<std::string>& GetEvents() { return events; }

    static size_t GetAndIncreasePid() {
      size_t val = pid_++;
      return val;
    }

    static void Clear() {
      summary_.clear();
    }

   private:
    static size_t pid_;
    static const std::vector<std::string> color_names;
    static std::vector<std::string> events;
    //Key: the hash function of device+map_type. Value: (key: The time step. value: The allocation information)
    static std::unordered_map<size_t, std::unordered_map<size_t, AllocationSummary> > summary_;
    MemoryInfoProfile() = default;

    static std::string CreateMetadataEvent(const std::string& process_name, size_t process_id);
    static std::string CreateMemoryEvent(size_t pid, size_t tid, const std::string& name, size_t offset, size_t size, const std::string& color_name);

    static std::string CreateSummaryEvent(size_t pid, size_t tid, const AllocationSummary& summary, size_t size, size_t bytes_for_pattern);
  };

  static void GenerateTensorMap(const SequentialExecutionPlan* execution_plan, const OrtValueNameIdxMap& value_name_idx_map);
  static void RecordPatternInfo(const MemoryPatternGroup& mem_patterns,  const MapType& type);

  static void RecordInitializerAllocInfo(const std::unordered_map<int, OrtValue>& tensor_map);
  static void RecordActivationAllocInfo(const OrtValueIndex idx, const OrtValue& value);
  static void SetDynamicAllocation(const OrtValueIndex idx);
  static void SetLocalRank(const int rank) { local_rank_ = rank; }

  static void SetIteration(size_t iteration) { iteration_ = iteration; }
  static void IncreaseIteration() { ++iteration_; }
  static size_t GetIteration() { return iteration_; }

  static void PrintMemoryInfoForLocation(const OrtDevice::DeviceType location);
  static void GenerateMemoryProfile();
  static void ClearMemoryInfoPerExecution() {
    for (auto& location_map : tensors_memory_info_map_) {
      location_map.second[MapType::DynamicActivation].clear();
      location_map.second[MapType::StaticActivation].clear();
    }
  }

  //Get the Allocation Information for tensor with index = idx;
  static const AllocInfoPerTensor* AllocPlan(const OrtValueIndex& idx) {
    if (tensor_alloc_info_map_.find(idx) != tensor_alloc_info_map_.end())
      return &tensor_alloc_info_map_.at(idx);
    else
      return nullptr;
  }

  //Create or add a tensor to a group. This is used to display the trace of specific group of tensors of interest
  //group_name: The name of the group of tensors
  //tensor_name: The tensor_name which is generated by argdef when building the graph
  static void AddRecordingTensorGroup(const std::string& group_name, const std::string& tensor_name) {
    customized_recording_group_[group_name][tensor_name] = true;
  }

  static bool InRecordingTensorGroup(const std::string& group_name, const std::string& tensor_name) {
    if (customized_recording_group_.find(group_name) == customized_recording_group_.end()) return false;
    if (customized_recording_group_.at(group_name).find(tensor_name) == customized_recording_group_.at(group_name).end()) return false;
    return true;
  }

 private:
  MemoryInfo() = default;
  static void RecordMemoryPatternInfo(const MemoryPatternGroup& mem_patterns, MapType type);
  static void RecordTensorDeviceAllocInfo(const OrtValueIndex idx, const OrtValue& value, const MapType& type);

  static bool IsInplaceReuse(const OrtValueIndex& idx) {
    return tensor_alloc_info_map_[idx].inplace_reuse;
  }

  //Key: The map type. E.g., initializer, activation. Value: A map from the tensor index to its memory information
  static std::map<OrtMemoryInfo, std::map<MapType, MemoryInfoMap> > tensors_memory_info_map_;

  //Key: The tensor index. Value: The Allocation information per tensor
  static std::map<OrtValueIndex, AllocInfoPerTensor> tensor_alloc_info_map_;

  //Key: The group name. Value: (key: The tensor name, value: not used (Use std::map for faster lookup).).
  static std::map<const std::string, std::map<const std::string, bool> > customized_recording_group_;

  //TODO: The dynamic and statically planned alignments may not be the same, need to check
  static const int alignment = 256;
  static size_t iteration_;
  static size_t num_node_size_;
  static int local_rank_;
  //Memory Profile
  static std::map<MapType, std::set<size_t> > time_step_trace_;
};

}  // namespace onnxruntime
#endif
