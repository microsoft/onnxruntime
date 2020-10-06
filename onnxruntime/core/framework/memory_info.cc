#include "core/framework/memory_info.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/ml_value.h"

#include <fstream>

namespace onnxruntime {
void MemoryInfo::GenerateMemoryMap(const SequentialExecutionPlan* execution_plan, const OrtValueNameIdxMap& value_name_idx_map) {
  if (!tensor_memoryinfo_map_.empty()) {
    return;
  }
  for (OrtValueIndex value_idx = 0; value_idx < execution_plan->allocation_plan.size(); ++value_idx) {
    //Only store tensor information
    if (!(execution_plan->allocation_plan[value_idx].value_type) || !(execution_plan->allocation_plan[value_idx].value_type->IsTensorType()))
      continue;
    MemoryInfoPerTensor mem_info;
    mem_info.mlvalue_index = value_idx;
    value_name_idx_map.GetName(mem_info.mlvalue_index, mem_info.mlvalue_name);
    mem_info.alloc_plan = execution_plan->allocation_plan[value_idx];
    ORT_ENFORCE(mem_info.alloc_plan.life_interval.first <= mem_info.alloc_plan.life_interval.second);
    ORT_ENFORCE(mem_info.alloc_plan.life_interval.first >= mem_info.alloc_plan.allocate_interval.first &&
                mem_info.alloc_plan.life_interval.second <= mem_info.alloc_plan.allocate_interval.second);
    tensor_memoryinfo_map_[value_idx] = std::move(mem_info);
  }
  return;
}

void MemoryInfo::RecordMemoryPatternInfo(const MemoryPatternGroup& mem_patterns) {
  for (const auto& location : mem_patterns.locations) {
    for (const auto& p : mem_patterns.GetPatterns(location)->GetPatternsMap()) {
      ORT_ENFORCE(tensor_memoryinfo_map_.find(p.first) != tensor_memoryinfo_map_.end());
      tensor_memoryinfo_map_[p.first].planned_block.offset_ = p.second.offset_;
      tensor_memoryinfo_map_[p.first].planned_block.size_ = p.second.size_;
    }
  }
}

void MemoryInfo::RecordDeviceAllocInfo(const std::unordered_map<int, OrtValue>& tensor_map) {
  for (const auto& item : tensor_map) {
    RecordTensorDeviceAllocInfo(item.first, item.second);
  }
}

//Comment: Need to add in the memory information for input
void MemoryInfo::RecordInputMemoryInfo(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds) {
  ORT_ENFORCE(feed_mlvalue_idxs.size() == feeds.size());
  for (size_t i = 0; i < feed_mlvalue_idxs.size(); ++i) {
    OrtValueIndex value_idx = feed_mlvalue_idxs[i];
    ORT_ENFORCE(tensor_memoryinfo_map_.find(value_idx) != tensor_memoryinfo_map_.end());
    const OrtValue& feed = feeds[i];
    RecordTensorDeviceAllocInfo(value_idx, feed);
  }
}

//Record the planned memory information
void MemoryInfo::RecordActivationPatternInfo(const MemoryPatternGroup& mem_patterns) {
  RecordMemoryPatternInfo(mem_patterns);
  for (auto& item : tensor_memoryinfo_map_) {
    if (item.second.alloc_plan.alloc_kind == AllocKind::kReuse) {
      auto reuse_buffer = item.second.alloc_plan.reused_buffer;
      item.second.planned_block = tensor_memoryinfo_map_[reuse_buffer].planned_block;
    }
  }
}

void MemoryInfo::SetDynamicAllocation(const OrtValueIndex idx) {
  ORT_ENFORCE(tensor_memoryinfo_map_.find(idx) != tensor_memoryinfo_map_.end());
  tensor_memoryinfo_map_[idx].dynamic_allocation = true;
}

//Record the actual allocated tensor in the device
void MemoryInfo::RecordTensorDeviceAllocInfo(const OrtValueIndex idx, const OrtValue& value) {
  ORT_ENFORCE(tensor_memoryinfo_map_.find(idx) != tensor_memoryinfo_map_.end());
  auto& tensor = value.Get<Tensor>();
  auto sizeinbytes = tensor.SizeInBytes() % alignment ? tensor.SizeInBytes() + alignment - tensor.SizeInBytes() % alignment : tensor.SizeInBytes();
  tensor_memoryinfo_map_[idx].allocated_block.offset_ = size_t(tensor.DataRaw());
  tensor_memoryinfo_map_[idx].allocated_block.size_ = sizeinbytes;
}

std::ostream& operator<<(std::ostream& out, const MemoryInfoPerTensor& mem_info_per_tensor) {
  out << "Tensor name: " << mem_info_per_tensor.mlvalue_name << ", ";
  out << "Index: " << mem_info_per_tensor.mlvalue_index << ", ";
  out << "Type: " << mem_info_per_tensor.tensor_type << ", ";
  out << "Alloc type " << mem_info_per_tensor.alloc_plan.alloc_kind << ", ";
  out << "Location: " << mem_info_per_tensor.alloc_plan.location.name << ", ";
  out << "lifetime: (" << mem_info_per_tensor.alloc_plan.life_interval.first << ", " << mem_info_per_tensor.alloc_plan.life_interval.second << "), ";
  out << "alloc time: (" << mem_info_per_tensor.alloc_plan.allocate_interval.first << ", " << mem_info_per_tensor.alloc_plan.allocate_interval.second << "), ";
  out << "planned block: (" << mem_info_per_tensor.planned_block.offset_ << ", "
      << (mem_info_per_tensor.planned_block.offset_ + mem_info_per_tensor.planned_block.size_) << "), ";
  out << "planned Size: " << mem_info_per_tensor.planned_block.size_ << ", ";
  out << "allocated block: (" << mem_info_per_tensor.allocated_block.offset_ << ", "
      << (mem_info_per_tensor.allocated_block.offset_ + mem_info_per_tensor.allocated_block.size_) << "), ";
  out << "allocated Size: " << mem_info_per_tensor.allocated_block.size_ << ", ";
  out << "is dynamically allocated? " << mem_info_per_tensor.dynamic_allocation << "\n";
  return out;
}

void MemoryInfo::PrintMemoryInfoForLocation(const logging::Logger& /*logger*/, const OrtDevice::DeviceType location) {
  for (const auto& item : tensor_memoryinfo_map_) {
    if (item.second.alloc_plan.location.device.Type() == location) {
      std::cout << item.second;
    }
  }
}

void MemoryInfo::WriteMemoryInfoToFile() {
  std::ofstream f;
  f.open(memory_info_file, std::ios_base::app);
  if (f.is_open()) {
    f << "iteration, name, index, type, alloc_type, location, lifetime_start, \
        lifetime_end, alloctime_start, alloctime_end, plan_block_start, plan_block_size, alloc_block_start, alloc_block_size, is_dynamic";
    for (const auto& item : tensor_memoryinfo_map_) {
      auto mt = item.second;
      f << iteration_ << ", ";
      f << mt.mlvalue_name << ", ";
      f << mt.mlvalue_index << ", ";
      f << mt.tensor_type << ", ";
      f << mt.alloc_plan.alloc_kind << ", ";
      f << mt.alloc_plan.location.name << ", ";
      f << mt.alloc_plan.life_interval.first << ", ";
      f << mt.alloc_plan.life_interval.second << ", ";
      f << mt.alloc_plan.allocate_interval.first << ", ";
      f << mt.alloc_plan.allocate_interval.second << ", ";
      f << mt.planned_block.offset_ << ", ";
      f << mt.planned_block.size_ << ", ";
      f << mt.allocated_block.offset_ << ", ";
      f << mt.allocated_block.size_ << ", ";
      f << mt.dynamic_allocation;
    }
  }
  f.close();
}

//In the mem pattern, a certain memory is allocated but notused.
//void MemoryInfo::ComputeFragmentation() {
//  std::map<const MemoryBlock*, std::vector<OrtValueIndex> > reuse_memory_map;
//  for (const auto& item : tensor_memoryinfo_map_) {
//    if (item.second.dynamic_allocation)
//      continue;
//    if (item.second.alloc_plan.alloc_kind == AllocKind::kReuse) {
//      if (reuse_memory_map[&item.second.planned_block].empty()) {
//        reuse_memory_map[&item.second.planned_block].push_back(item.second.alloc_plan.reused_buffer);
//      }
//      reuse_memory_map[&item.second.planned_block].push_back(item.first);
//    }
//  }
//  for (auto& item : reuse_memory_map) {
//    std::sort(item.second.begin(), item.second.end(), [this](const OrtValueIndex& first, const OrtValueIndex& second) -> bool {
//      auto a = tensor_memoryinfo_map_[first].alloc_plan.life_interval.first;
//      auto b = tensor_memoryinfo_map_[second].alloc_plan.life_interval.first;
//      return (a < b);
//    });
//  }
//}

//void MemoryInfo::CollectMemoryOccupation() {
//  std::map<const MemoryBlock*, std::vector<OrtValueIndex> > memory_tensorid_map_;
//  for (const auto& item : tensor_memoryinfo_map_) {
//    memory_tensorid_map_[&item.second.allocated_block].push_back(item.first);
//  }
//  PrintMemoryOccupation(memory_tensorid_map_);
//}
//
//void MemoryInfo::PrintMemoryOccupation(const OrtDevice::DeviceType location, const std::map<const MemoryBlock*, std::vector<OrtValueIndex> >& memory_tensorid_map_) {
//  std::ofstream f;
//  f.open("test.txt", std::ios_base::app);
//  for (auto& item : memory_tensorid_map_) {
//    auto id = item.second[0];
//    if (tensor_memoryinfo_map_[id].alloc_plan.location.device.Type() != location)
//      continue;
//  }
//  f.close();
//}

}  // namespace onnxruntime
