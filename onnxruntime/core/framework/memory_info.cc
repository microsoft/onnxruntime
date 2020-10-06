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

static std::string CreateMetadataEvent(const std::string& process_name, size_t process_id) {
  std::stringstream evt;
  evt << "{";
  evt << "\"ph\":\"M\",";
  evt << "\"name\":\"process_name\",";
  evt << "\"pid\":" << process_id << ",";
  evt << "\"args\":{\"name\":\"" << process_name << "\"}";
  evt << "}";
  evt << "," << std::endl;
  evt << "{";
  evt << "\"ph\":\"M\",";
  evt << "\"name\":\"process_sort_index\",";
  evt << "\"pid\":" << process_id << ",";
  evt << "\"args\":{\"sort_index\":\"" << process_id << "\"}";
  evt << "}";
  return evt.str();
}

static std::string CreateMemoryEvent(size_t pid, size_t tid, const std::string& name, size_t offset, size_t size, const std::string& color_name) {
  std::stringstream evt;
  evt << "{";
  evt << "\"ph\":\"X\",";
  evt << "\"pid\":" << pid << ",";
  evt << "\"tid\":" << tid++ << ",";
  evt << "\"ts\":" << offset << ",";
  evt << "\"dur\":" << size << ",";
  evt << "\"name\":\"" << name << "\",";
  evt << "\"cname\":\"" << color_name << "\",";
  evt << "\"args\":{";
  evt << "\"name\":\"" << name << "\",";
  evt << "\"offset\":" << offset << ",";
  evt << "\"size\":" << size;
  evt << "}";
  evt << "}";
  return evt.str();
}

// Data needed for each tensor:
// - name
// - type (initializer, statically allocated activation, dynamically allocated activation)
// - allocation offset (zero-based within it's type)
// - allocation size (in bytes)
// - for activations: lifetime (start and end execution step)
//   - memory blocks/lifetime should not overlap when tensors are the same type
//
// TODO: the data should be organized in the following way
//
// [1] initializers:  tensors that exist for the lifetime of the graph/session.
//       AllocKind=kAllocateStatically.  Need zero-based offset + size.
//       Initializers can be reused.  We need some indication that it's reusing a static initializer.
//       Currently, I use the alloc lifetime to infer this (start=0, end=max int)
// [2] activations (static):  tensors that exist during execution steps, and their allocations are statically planned.
//       Generally, these are AllocKind=kAllocate,kReuse
//       Need zero-based offset + size, lifetime (first step it's used, last step it's used).
//       Need an indication of tensors that were statically planned (offset+size) but ended up dynamically allocated outside the BFC arena (offset+size).
//       For reused tensors, the lifetime should not overlap with other reuses/original allocation.
// [3] activations (dynamic):  tensors that exist during execution steps, and their allocations are dynamically allocated outside a memory plan.
//       Generally, these are AllocKind=kAllocate,kReuse
//       Need zero-based offset + size (based on the addresses from all other dynamic allocations only), lifetime (first step it's used, last step it's used).
//       Need an indication of tensors that were statically planned (offset+size) but ended up dynamically allocated outside the BFC arena (offset+size).
//       For reused tensors, the lifetime should not overlap with other reuses/original allocation.
void MemoryInfo::GenerateMemoryProfile() {
  std::vector<std::string> color_names = {
    "good", "bad", "terrible", "yellow", "olive", "generic_work",
    "background_memory_dump", "light_memory_dump", "detailed_memory_dump",
    "thread_state_uninterruptible", "thread_state_iowait", "thread_state_running",
    "thread_state_runnable", "thread_state_unknown",
    "cq_build_running", "cq_build_passed", "cq_build_failed", "cq_build_abandoned",
    "cq_build_attempt_runnig", "cq_build_attempt_passed", "cq_build_attempt_failed",
  };

  size_t base_offset = std::numeric_limits<size_t>::max();
  for (const auto& item : tensor_memoryinfo_map_) {
    const auto& info = item.second;
    const AllocKind alloc_kind = info.alloc_plan.alloc_kind;
    if (info.alloc_plan.location.device.Type() != OrtDevice::GPU) continue;
    if (alloc_kind == AllocKind::kAllocate) {
      base_offset = std::min(base_offset, info.allocated_block.offset_);
    }
  }

  std::vector<std::string> events;

  // Metadata.
  const size_t initializers_pid = 0;
  const size_t activations_pid = 1;
  events.push_back(CreateMetadataEvent("GPU (initializers)", initializers_pid));
  events.push_back(CreateMetadataEvent("GPU (activations)", activations_pid));

  // Allocations.
  for (const auto& item : tensor_memoryinfo_map_) {
    const auto& info = item.second;
    if (info.alloc_plan.location.device.Type() != OrtDevice::GPU) continue;

    const std::string& name = info.mlvalue_name;
    const std::string cname = color_names[events.size() % color_names.size()];
    const AllocKind alloc_kind = info.alloc_plan.alloc_kind;

    // Skip initializers that are reused.
    if (alloc_kind == AllocKind::kReuse && info.alloc_plan.allocate_interval.first == 0 && info.alloc_plan.allocate_interval.second == 4294967295) continue;
    // Initializers.
    if (alloc_kind == AllocKind::kAllocateStatically) {
      size_t offset = info.planned_block.offset_;
      size_t size = info.planned_block.size_;
      events.push_back(CreateMemoryEvent(initializers_pid, 0, name, offset, size, cname));
    }
    // Activations.
    if (alloc_kind == AllocKind::kAllocate || alloc_kind == AllocKind::kReuse) {
      size_t offset = info.allocated_block.offset_ - base_offset;
      size_t size = info.allocated_block.size_;
      size_t alloc_step = alloc_kind == AllocKind::kReuse ? info.alloc_plan.life_interval.first + 1 : info.alloc_plan.life_interval.first;
      size_t dealloc_step = info.alloc_plan.life_interval.second;
      for (size_t tid = alloc_step; tid <= dealloc_step; tid++) {
        events.push_back(CreateMemoryEvent(activations_pid, tid, name, offset, size, cname));
      }
    }
  }

  // Write memory profile .json
  std::ofstream memory_profile("memory_profile.json", std::ios::trunc);
  memory_profile << "[" << std::endl;
  for (size_t i = 0; i < events.size(); i++) {
    memory_profile << "  " << events[i];
    if (i < events.size() - 1) memory_profile << ",";
    memory_profile << std::endl;
  }
  memory_profile << "]" << std::endl;
}

}  // namespace onnxruntime
