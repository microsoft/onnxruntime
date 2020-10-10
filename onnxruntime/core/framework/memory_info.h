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

struct AllocInfoPerTensor {
  AllocInfoPerTensor() = default;
  MLValueTensorType tensor_type{Unknown};
  OrtValueIndex mlvalue_index{0};
  OrtValueName mlvalue_name{""};

  IntervalT lifetime_interval{0, 0};
  IntervalT alloctime_interval{0, 0};
  bool inplace_reuse{false};
  OrtValueIndex reused_buffer{0};
  AllocKind alloc_kind;
  OrtMemoryInfo location;
};

struct MemoryInfoPerTensor {
  MemoryBlock planned_block;
  MemoryBlock alloced_block;
  bool dynamic_allocation{false};
};

struct MemoryInfoMap {
 public:
  MemoryInfoMap() = default;

  inline void AddPlannedMemory(const OrtValueIndex& idx, const MemoryBlock& mb) {
    map_[idx].planned_block = mb;
  }

  inline void AddAllocMemory(const OrtValueIndex& idx, MemoryBlock& mb) {
    map_[idx].alloced_block = mb;
    if (ptr_offset == 0 || ptr_offset > mb.offset_) {
      ptr_offset = mb.offset_;
    }
  }

  inline void SetDynamicAllocation(const OrtValueIndex& idx, bool flag) {
    map_[idx].dynamic_allocation = flag;
  }

  inline const bool DynamicAllocation(const OrtValueIndex& idx) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    return map_.at(idx).dynamic_allocation;
  }
  inline const MemoryBlock& GetPlannedMemory(const OrtValueIndex& idx) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    return map_.at(idx).planned_block;
  }

  inline const size_t GetPlannedAddress(const OrtValueIndex& idx) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    return map_.at(idx).planned_block.offset_;
  }

  inline const size_t GetPlannedSize(const OrtValueIndex& idx) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    return map_.at(idx).planned_block.size_;
  }
  const size_t GetAllocAddress(const OrtValueIndex& idx, bool raw = false) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    if (raw) {
      return map_.at(idx).alloced_block.offset_;
    } else {
      return map_.at(idx).alloced_block.offset_ - ptr_offset;
    }
  }

  inline const size_t GetAllocSize(const OrtValueIndex& idx) const {
    ORT_ENFORCE(map_.find(idx) != map_.end());
    return map_.at(idx).alloced_block.size_;
  }

  auto begin() { return map_.begin(); }
  auto begin() const { return map_.begin(); }
  auto end() { return map_.end(); }
  auto end() const { return map_.end(); }

 private:
  std::unordered_map<OrtValueIndex, MemoryInfoPerTensor> map_;
  size_t ptr_offset{0};  //The start ptr of the raw data
};

class MemoryInfo {
 public:
  enum MapType {
    Initializer = 0,
    Activation,
  };
  MemoryInfo() : iteration_(0) {
    time_t now_c = std::time(0);
    struct tm timeinfo;
    localtime_s(&timeinfo, &now_c);
    char buffer[80];
    asctime_s(buffer, &timeinfo);
    memory_info_file = "memory_info_file_" + std::string(buffer);
  }
  void GenerateTensorMap(const SequentialExecutionPlan* execution_plan, const OrtValueNameIdxMap& value_name_idx_map);
  void RecordInitializerPatternInfo(const MemoryPatternGroup& mem_patterns);
  void RecordActivationPatternInfo(const MemoryPatternGroup& mem_patterns);

  void RecordInitializerAllocInfo(const std::unordered_map<int, OrtValue>& tensor_map);
  void RecordActivationAllocInfo(const OrtValueIndex idx, const OrtValue& value);
  void SetDynamicAllocation(const OrtValueIndex idx);

  void PrintMemoryInfoForLocation(const logging::Logger& /*logger*/, const OrtDevice::DeviceType location);
  inline void SetIteration(size_t iteration) { iteration_ = iteration; }
  void GenerateMemoryProfile();
  const AllocInfoPerTensor& AllocPlan(const OrtValueIndex& idx) {
    return tensor_alloc_info_map_[idx];
  }

 private:
  void RecordMemoryPatternInfo(const MemoryPatternGroup& mem_patterns, MapType type);
  void RecordTensorDeviceAllocInfo(const OrtValueIndex idx, const OrtValue& value, const MapType& type);

  bool IsInplaceReuse(const OrtValueIndex& idx) {
    return tensor_alloc_info_map_[idx].inplace_reuse;
  }

  //Key: The map type. E.g., initializer, activation. Value: A map from the tensor index to its memory information
  std::unordered_map<MapType, MemoryInfoMap> tensors_memory_info_map_;

  std::unordered_map<OrtValueIndex, AllocInfoPerTensor> tensor_alloc_info_map_;

  //TODO: The dynamic and statically planned alignments may not be the same, need to check
  static const int alignment = 256;
  size_t iteration_ = 0;
  std::string memory_info_file;
  size_t num_node_size_;
};

}  // namespace onnxruntime