#include "core/framework/memory_info.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/ml_value.h"

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
    ORT_ENFORCE(mem_info.alloc_plan.life_interval.first < mem_info.alloc_plan.life_interval.second);
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
    ORT_ENFORCE(tensor_memoryinfo_map_.find(item.first) != tensor_memoryinfo_map_.end());
    auto& tensor = item.second.Get<Tensor>();
    auto sizeinbytes = tensor.SizeInBytes() % alignment ? tensor.SizeInBytes() + alignment - tensor.SizeInBytes() % alignment : tensor.SizeInBytes();
    tensor_memoryinfo_map_[item.first].allocated_block.offset_ = size_t(tensor.DataRaw());
    tensor_memoryinfo_map_[item.first].allocated_block.offset_ = sizeinbytes;
  }
}

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

void MemoryInfo::RecordTensorDeviceAllocInfo(const OrtValueIndex idx, const OrtValue& value) {
  ORT_ENFORCE(tensor_memoryinfo_map_.find(idx) != tensor_memoryinfo_map_.end());
  auto& tensor = value.Get<Tensor>();
  auto sizeinbytes = tensor.SizeInBytes() % alignment ? tensor.SizeInBytes() + alignment - tensor.SizeInBytes() % alignment : tensor.SizeInBytes();
  tensor_memoryinfo_map_[idx].allocated_block.offset_ = size_t(tensor.DataRaw());
  tensor_memoryinfo_map_[idx].allocated_block.offset_ = sizeinbytes;
}

//void MemoryInfo::RecordWeightMemoryInfo(const SessionState& session_state) {
//  auto initialized_tensors = session_state.GetInitializedTensors();
//  for (auto& item : initialized_tensors) {
//    auto value_idx = item.first;
//    auto& tensor = initialized_tensors.at(value_idx).Get<Tensor>();
//    auto sizeinbytes = tensor.SizeInBytes() % 256 ? tensor.SizeInBytes() + 256 - tensor.SizeInBytes() % 256 : tensor.SizeInBytes();
//    tensor_memoryinfo_map_[value_idx].block.offset_ = size_t(tensor.DataRaw());
//    tensor_memoryinfo_map_[value_idx].block.size_ = sizeinbytes;
//  }
//}
//
//void MemoryInfo::RecordInputMemoryInfo(const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds) {
//  for (auto value_idx : feed_mlvalue_idxs) {
//    ORT_ENFORCE(tensor_memoryinfo_map_.find(value_idx) != tensor_memoryinfo_map_.end());
//    const OrtValue& feed = feeds[value_idx];
//    auto& tensor = feed.Get<Tensor>();
//    auto sizeinbytes = tensor.SizeInBytes() % 256 ? tensor.SizeInBytes() + 256 - tensor.SizeInBytes() % 256 : tensor.SizeInBytes();
//    tensor_memoryinfo_map_[value_idx].block.offset_ = size_t(tensor.DataRaw());
//    tensor_memoryinfo_map_[value_idx].block.size_ = sizeinbytes;
//  }
//}
//
///*==============================================*/
//
//void MemoryInfo::RecordMemoryInfo(const SessionState& session_state, const MemoryPatternGroup* mem_patterns,
//                                  const std::vector<int>& feed_mlvalue_idxs, const std::vector<OrtValue>& feeds) {
//  auto execution_plan = session_state.GetExecutionPlan();
//
//  //Collect Inputs and outputs
//  for (auto value_idx : feed_mlvalue_idxs) {
//    MemoryInfoPerTensor mem_info;
//    mem_info.mlvalue_index = value_idx;
//    mem_info.location = execution_plan->allocation_plan[value_idx].location;
//    session_state.GetOrtValueNameIdxMap().GetName(mem_info.mlvalue_index, mem_info.mlvalue_name);
//    mem_info.life_interval = execution_plan->allocation_plan[value_idx].life_interval;
//    mem_info.allocate_interval = execution_plan->allocation_plan[value_idx].allocate_interval;
//    const OrtValue& feed = feeds[value_idx];
//    auto& tensor = feed.Get<Tensor>();
//    auto sizeinbytes = tensor.SizeInBytes() % 256 ? tensor.SizeInBytes() + 256 - tensor.SizeInBytes() % 256 : tensor.SizeInBytes();
//    mem_info.block.offset_ = size_t(tensor.DataRaw());
//    mem_info.block.size_ = sizeinbytes;
//    std::cout << "input mem_info.offset_ " << mem_info.block.offset_ << ", " << mem_info.block.offset_ << "\n";
//    tensor_memoryinfo_map_[value_idx] = std::move(mem_info);
//  }
//
//  auto& initialized_tensors = session_state.GetInitializedTensors();
//
//  for (OrtValueIndex value_idx = 0; value_idx < execution_plan->allocation_plan.size(); ++value_idx) {
//    //Only store tensor information
//    if (!(execution_plan->allocation_plan[value_idx].value_type) || !(execution_plan->allocation_plan[value_idx].value_type->IsTensorType()))
//      continue;
//    if (execution_plan->allocation_plan[value_idx].location.device.Type() != OrtDevice::GPU)
//      continue;
//    if (execution_plan->allocation_plan[value_idx].alloc_kind == AllocKind::kPreExisting)
//      continue;
//
//    MemoryInfoPerTensor mem_info;
//    mem_info.mlvalue_index = value_idx;
//    mem_info.location = execution_plan->allocation_plan[value_idx].location;
//    session_state.GetOrtValueNameIdxMap().GetName(mem_info.mlvalue_index, mem_info.mlvalue_name);
//    mem_info.life_interval = execution_plan->allocation_plan[value_idx].life_interval;
//    mem_info.allocate_interval = execution_plan->allocation_plan[value_idx].allocate_interval;
//
//    //Collect Weights
//    if (execution_plan->allocation_plan[value_idx].alloc_kind == AllocKind::kAllocateStatically) {
//      ORT_ENFORCE(initialized_tensors.find(value_idx) != initialized_tensors.end());
//      auto& tensor = initialized_tensors.at(value_idx).Get<Tensor>();
//      auto sizeinbytes = tensor.SizeInBytes() % 256 ? tensor.SizeInBytes() + 256 - tensor.SizeInBytes() % 256 : tensor.SizeInBytes();
//      mem_info.block.offset_ = size_t(tensor.DataRaw());
//      mem_info.block.size_ = sizeinbytes;
//      //if (mem_patterns->GetPatterns(mem_info.location)->GetBlock(mem_info.mlvalue_index)) {
//      //  auto block_offset = mem_patterns->GetPatterns(mem_info.location)->GetBlock(mem_info.mlvalue_index)->offset_;
//      //  if (ptr_startaddress == 0) {
//      //    ptr_startaddress = mem_info.block.offset_ - block_offset;
//      //  }
//      //std::cout << "weight mem_info.offset_ " << block_offset << ", " << mem_info.block.offset_ - ptr_startaddress << "\n";
//      //}
//      tensor_memoryinfo_map_[value_idx] = std::move(mem_info);
//      continue;
//    }
//
//    //ORT_ENFORCE(mem_info.life_interval.first < mem_info.life_interval.second);
//    ORT_ENFORCE(mem_info.life_interval.first >= mem_info.allocate_interval.first && mem_info.life_interval.second <= mem_info.allocate_interval.second);
//    if (mem_patterns->GetPatterns(mem_info.location)->GetBlock(mem_info.mlvalue_index)) {
//      mem_info.block.offset_ = mem_patterns->GetPatterns(mem_info.location)->GetBlock(mem_info.mlvalue_index)->offset_;
//      mem_info.block.size_ = mem_patterns->GetPatterns(mem_info.location)->GetBlock(mem_info.mlvalue_index)->size_;
//      std::cout << "act mem_info.offset_ " << mem_info.block.offset_ << "\n";
//      tensor_memoryinfo_map_[value_idx] = std::move(mem_info);
//    }
//    //if (execution_plan->allocation_plan[value_idx].alloc_kind == AllocKind::kReuse) {
//    //  auto reuse_buffer = execution_plan->allocation_plan[value_idx].reused_buffer;
//    //  mem_info.block = mem_patterns->GetPatterns(mem_info.location)->GetBlock(reuse_buffer);
//    //}
//    //ORT_ENFORCE(mem_info.block);
//  }
//}
//
std::ostream& operator<<(std::ostream& out, const MemoryInfoPerTensor& mem_info_per_tensor) {
  out << "Tensor name: " << mem_info_per_tensor.mlvalue_name << ", ";
  out << "Index: " << mem_info_per_tensor.mlvalue_index << ", ";
  out << "Type: " << mem_info_per_tensor.tensor_type << ", ";
  out << "Location: " << mem_info_per_tensor.alloc_plan.location.name << ", ";
  out << "lifetime: (" << mem_info_per_tensor.alloc_plan.life_interval.first << ", " << mem_info_per_tensor.alloc_plan.life_interval.second << "), ";
  out << "alloc time: (" << mem_info_per_tensor.alloc_plan.allocate_interval.first << ", " << mem_info_per_tensor.alloc_plan.allocate_interval.second << "), ";
  out << "planned block: (" << mem_info_per_tensor.planned_block.offset_ << ", "
      << (mem_info_per_tensor.planned_block.offset_ + mem_info_per_tensor.planned_block.size_ - 1) << ")\n";
  out << "planned Size: " << mem_info_per_tensor.planned_block.size_ << ", ";
  out << "allocated block: (" << mem_info_per_tensor.allocated_block.offset_ << ", "
      << (mem_info_per_tensor.allocated_block.offset_ + mem_info_per_tensor.allocated_block.size_ - 1) << ")\n";
  out << "allocated Size: " << mem_info_per_tensor.allocated_block.size_ << ", ";
  return out;
}

void MemoryInfo::PrintMemoryInfoForLocation(const logging::Logger& /*logger*/, const OrtDevice::DeviceType location) {
  for (const auto& item : tensor_memoryinfo_map_) {
    if (item.second.alloc_plan.location.device.Type() == location) {
      std::cout << item.second;
    }
  }
}

}  // namespace onnxruntime
