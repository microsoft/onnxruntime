#include "core/framework/memory_info.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/ml_value.h"

#include <fstream>

namespace onnxruntime {
void MemoryInfo::GenerateTensorMap(const SequentialExecutionPlan* execution_plan, const OrtValueNameIdxMap& value_name_idx_map) {
  if (!tensor_alloc_info_map_.empty()) {
    return;
  }
  num_node_size_ = execution_plan->execution_plan.size();
  for (OrtValueIndex value_idx = 0; value_idx < OrtValueIndex(execution_plan->allocation_plan.size()); ++value_idx) {
    //Only store tensor information
    if (!(execution_plan->allocation_plan[value_idx].value_type) || !(execution_plan->allocation_plan[value_idx].value_type->IsTensorType()))
      continue;
    AllocInfoPerTensor mem_info;
    mem_info.mlvalue_index = value_idx;
    value_name_idx_map.GetName(mem_info.mlvalue_index, mem_info.mlvalue_name);
    mem_info.lifetime_interval = execution_plan->allocation_plan[value_idx].life_interval;
    mem_info.alloctime_interval = execution_plan->allocation_plan[value_idx].allocate_interval;
    mem_info.reused_buffer = (execution_plan->allocation_plan[value_idx].alloc_kind != AllocKind::kReuse) ? value_idx : execution_plan->allocation_plan[value_idx].reused_buffer;
    //If the tensor is using memory outside of the scope, do not store it
    if (execution_plan->allocation_plan[mem_info.reused_buffer].alloc_kind == AllocKind::kPreExisting) continue;
    if (execution_plan->allocation_plan[mem_info.reused_buffer].alloc_kind == AllocKind::kAllocateOutput) continue;
    mem_info.inplace_reuse = (execution_plan->allocation_plan[value_idx].inplace_reuse != -1 && execution_plan->allocation_plan[value_idx].inplace_reuse != value_idx);
    mem_info.alloc_kind = execution_plan->allocation_plan[value_idx].alloc_kind;
    mem_info.location = execution_plan->allocation_plan[value_idx].location;

    ORT_ENFORCE(mem_info.lifetime_interval.first <= mem_info.lifetime_interval.second);
    ORT_ENFORCE(mem_info.lifetime_interval.first >= mem_info.alloctime_interval.first &&
                mem_info.lifetime_interval.second <= mem_info.alloctime_interval.second);
    tensor_alloc_info_map_[value_idx] = std::move(mem_info);
    time_step_trace_[mem_info.alloc_kind].insert(mem_info.lifetime_interval.first);
    time_step_trace_[mem_info.alloc_kind].insert(mem_info.lifetime_interval.second);
    tensors_memory_info_map_[mem_info.location];
  }
  return;
}

//Record the planned memory information
void MemoryInfo::RecordMemoryPatternInfo(const MemoryPatternGroup& mem_patterns, MapType type) {
  for (const auto& location : mem_patterns.locations) {
    for (const auto& p : mem_patterns.GetPatterns(location)->GetPatternsMap()) {
      ORT_ENFORCE(AllocPlan(p.first));
      tensors_memory_info_map_.at(location)[type].AddPlannedMemory(p.first, p.second);
    }
  }
}

void MemoryInfo::RecordActivationPatternInfo(const MemoryPatternGroup& mem_patterns) {
  RecordMemoryPatternInfo(mem_patterns, MapType::StaticActivation);
  for (auto& item : tensor_alloc_info_map_) {
    if (item.second.alloc_kind == AllocKind::kReuse) {
      auto reuse_buffer = AllocPlan(item.first)->reused_buffer;
      auto& map = tensors_memory_info_map_.at(AllocPlan(item.first)->location);
      if (map[MapType::StaticActivation].Contain(reuse_buffer)) {
        auto& reused_memory = map[MapType::StaticActivation].GetPlannedMemory(reuse_buffer);
        map[MapType::StaticActivation].AddPlannedMemory(item.first, reused_memory);
      }
    }
  }
}

void MemoryInfo::RecordInitializerPatternInfo(const MemoryPatternGroup& mem_patterns) {
  RecordMemoryPatternInfo(mem_patterns, MapType::Initializer);
  for (auto& item : tensor_alloc_info_map_) {
    if (item.second.alloc_kind == AllocKind::kReuse) {
      auto reuse_buffer = AllocPlan(item.first)->reused_buffer;
      auto& map = tensors_memory_info_map_.at(AllocPlan(item.first)->location);
      if (map[MapType::Initializer].Contain(reuse_buffer)) {
        auto& reused_memory = map[MapType::Initializer].GetPlannedMemory(reuse_buffer);
        map[MapType::Initializer].AddPlannedMemory(item.first, reused_memory);
      }
    }
  }
}

//Record the actual allocated tensor in the device
void MemoryInfo::RecordTensorDeviceAllocInfo(const OrtValueIndex idx, const OrtValue& value, const MapType& type) {
  if (tensor_alloc_info_map_.find(idx) == tensor_alloc_info_map_.end()) return;
  auto& tensor = value.Get<Tensor>();
  auto sizeinbytes = tensor.SizeInBytes() % alignment ? tensor.SizeInBytes() + alignment - tensor.SizeInBytes() % alignment : tensor.SizeInBytes();
  MemoryBlock mb(size_t(tensor.DataRaw()), sizeinbytes);
  if (type == MapType::StaticActivation) {
    bool valid = tensors_memory_info_map_.at(AllocPlan(idx)->location).at(type).Contain(idx);
    ORT_ENFORCE(valid);
  }
  tensors_memory_info_map_.at(AllocPlan(idx)->location)[type].AddAllocMemory(idx, mb);
}

void MemoryInfo::RecordInitializerAllocInfo(const std::unordered_map<int, OrtValue>& tensor_map) {
  for (const auto& item : tensor_map) {
    ORT_ENFORCE(AllocPlan(item.first));
    RecordTensorDeviceAllocInfo(item.first, item.second, MapType::Initializer);
  }
}

void MemoryInfo::RecordActivationAllocInfo(const OrtValueIndex idx, const OrtValue& value) {
  if (!AllocPlan(idx)) return;
  auto reuse_buffer = AllocPlan(idx)->reused_buffer;
  MapType map_type;
  auto& map = tensors_memory_info_map_.at(AllocPlan(reuse_buffer)->location);
  if (map[MapType::DynamicActivation].Contain(reuse_buffer))
    map_type = MapType::DynamicActivation;
  else if (map[MapType::StaticActivation].Contain(reuse_buffer))
    map_type = MapType::StaticActivation;
  else if (map[MapType::Initializer].Contain(reuse_buffer))
    map_type = MapType::Initializer;
  else
    ORT_ENFORCE(false, "Unsupported tensor type.");

  RecordTensorDeviceAllocInfo(idx, value, map_type);
}

void MemoryInfo::SetDynamicAllocation(const OrtValueIndex idx) {
  if (!AllocPlan(idx)) return;
  if (!tensors_memory_info_map_.at(AllocPlan(idx)->location)[MapType::DynamicActivation].Contain(idx)) {
    tensors_memory_info_map_.at(AllocPlan(idx)->location)[MapType::DynamicActivation][idx];
  }
}

void PrintInforPerTensor(const MemoryInfo::AllocInfoPerTensor& alloc_info, const MemoryInfoPerTensor& mem_info, const size_t& rel_addr) {
  std::cout << "Tensor name: " << alloc_info.mlvalue_name << ", ";
  std::cout << "Index: " << alloc_info.mlvalue_index << ", ";
  std::cout << "Type: " << alloc_info.tensor_type << ", ";
  std::cout << "Reuse inplace: " << alloc_info.inplace_reuse << ", ";
  std::cout << "Alloc type " << alloc_info.alloc_kind << ", ";
  std::cout << "Location: " << alloc_info.location.name << ", ";
  std::cout << "lifetime: (" << alloc_info.lifetime_interval.first << ", " << alloc_info.lifetime_interval.second << "), ";
  std::cout << "alloc time: (" << alloc_info.alloctime_interval.first << ", " << alloc_info.alloctime_interval.second << "), ";
  std::cout << "planned block: (" << mem_info.planned_block.offset_ << ", "
            << (mem_info.planned_block.offset_ + mem_info.planned_block.size_) << "), ";
  std::cout << "planned Size: " << mem_info.planned_block.size_ << ", ";
  std::cout << "allocated block: (" << rel_addr << ", "
            << (rel_addr + mem_info.alloced_block.size_) << "), ";
  std::cout << "allocated Size: " << mem_info.alloced_block.size_ << "\n";
}

void MemoryInfo::PrintMemoryInfoForLocation(const logging::Logger& /*logger*/, const OrtDevice::DeviceType location) {
  for (const auto& map : tensors_memory_info_map_) {
    std::cout << "Initializer in " << map.first.name << "\n";
    const auto& initailizer_map = map.second.at(MapType::Initializer);
    for (const auto& item : initailizer_map) {
      if (AllocPlan(item.first)->location.device.Type() != location) continue;
      PrintInforPerTensor(*AllocPlan(item.first), item.second, initailizer_map.GetAllocAddress(item.first));
    }
    std::cout << "\nStatic Activation in " << map.first.name << "\n";
    const auto& static_activation_map = map.second.at(MapType::StaticActivation);
    for (const auto& item : static_activation_map) {
      if (AllocPlan(item.first)->location.device.Type() != location) continue;
      PrintInforPerTensor(*AllocPlan(item.first), item.second, static_activation_map.GetAllocAddress(item.first));
    }

    std::cout << "\nDynamic Activation in " << map.first.name << "\n";
    const auto& dynamic_activation_map = map.second.at(MapType::DynamicActivation);
    for (const auto& item : dynamic_activation_map) {
      if (AllocPlan(item.first)->location.device.Type() != location) continue;
      PrintInforPerTensor(*AllocPlan(item.first), item.second, dynamic_activation_map.GetAllocAddress(item.first));
    }
  }
}

std::string MemoryInfo::MemoryInfoProfile::CreateMetadataEvent(const std::string& process_name, size_t process_id) {
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

std::string MemoryInfo::MemoryInfoProfile::CreateMemoryEvent(size_t pid, size_t tid, const std::string& name, size_t offset, size_t size, const std::string& color_name) {
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

//Data needed for each tensor:
//- name
//- type (initializer, statically allocated activation, dynamically allocated activation)
//- allocation offset (zero-based within it's type)
//- allocation size (in bytes)
//- for activations: lifetime (start and end execution step)
//  - memory blocks/lifetime should not overlap when tensors are the same type

//TODO: the data should be organized in the following way

//[1] initializers:  tensors that exist for the lifetime of the graph/session.
//      AllocKind=kAllocateStatically.  Need zero-based offset + size.
//      Initializers can be reused.  We need some indication that it's reusing a static initializer.
//      Currently, I use the alloc lifetime to infer this (start=0, end=max int)
//[2] activations (static):  tensors that exist during execution steps, and their allocations are statically planned.
//      Generally, these are AllocKind=kAllocate,kReuse
//      Need zero-based offset + size, lifetime (first step it's used, last step it's used).
//      Need an indication of tensors that were statically planned (offset+size) but ended up dynamically allocated outside the BFC arena (offset+size).
//      For reused tensors, the lifetime should not overlap with other reuses/original allocation.
//[3] activations (dynamic):  tensors that exist during execution steps, and their allocations are dynamically allocated outside a memory plan.
//      Generally, these are AllocKind=kAllocate,kReuse
//      Need zero-based offset + size (based on the addresses from all other dynamic allocations only), lifetime (first step it's used, last step it's used).
//      Need an indication of tensors that were statically planned (offset+size) but ended up dynamically allocated outside the BFC arena (offset+size).
//      For reused tensors, the lifetime should not overlap with other reuses/original allocation.

const std::vector<std::string> MemoryInfo::MemoryInfoProfile::color_names = {
    "good",
    "bad",
    "terrible",
    "yellow",
    "olive",
    "generic_work",
    "background_memory_dump",
    "light_memory_dump",
    "detailed_memory_dump",
    "thread_state_uninterruptible",
    "thread_state_iowait",
    "thread_state_running",
    "thread_state_runnable",
    "thread_state_unknown",
    "cq_build_running",
    "cq_build_passed",
    "cq_build_failed",
    "cq_build_abandoned",
    "cq_build_attempt_runnig",
    "cq_build_attempt_passed",
    "cq_build_attempt_failed",
};

void MemoryInfo::MemoryInfoProfile::CreateEvents(const std::string p_name, const size_t pid, const MemoryInfo::MapType& map_type, const std::string name_pattern) {
  // Metadata.
  std::string pid_name_internal = p_name + name_pattern;
  events.push_back(CreateMetadataEvent(pid_name_internal, pid));

  //Create Event for each tensor
  for (const auto& location_map : mem_info_.tensors_memory_info_map_) {
    if (location_map.first.device.Type() != OrtDevice::GPU) continue;
    const auto& map = location_map.second.at(map_type);
    for (const auto& item : map) {
      const auto info = mem_info_.AllocPlan(item.first);
      if (info->inplace_reuse) continue;

      const std::string& name = info->mlvalue_name;
      //Filter out string with no certain name
      if (!name_pattern.empty() && name.find(name_pattern) == std::string::npos) continue;
      const std::string cname = color_names[events.size() % color_names.size()];
      //Sometimes a tensor can be both statically planned and dynamically planed, so we need to use planed address/size in static_activation type
      size_t offset = map_type == MemoryInfo::MapType::StaticActivation ? map.GetPlannedAddress(item.first) : map.GetAllocAddress(item.first);
      size_t size = map_type == MemoryInfo::MapType::StaticActivation ? map.GetPlannedSize(item.first) : map.GetAllocSize(item.first);
      size_t alloc_step = mem_info_.AllocPlan(item.first)->lifetime_interval.first;  //alloc_kind == AllocKind::kReuse ? info.alloc_plan.life_interval.first + 1 : info.alloc_plan.life_interval.first;
      size_t dealloc_step = mem_info_.AllocPlan(item.first)->lifetime_interval.second;
      const auto& ts_map = mem_info_.time_step_trace_[mem_info_.AllocPlan(item.first)->alloc_kind];
      const auto& start_itr = ts_map.find(alloc_step);
      const auto& end_itr = ts_map.find(dealloc_step);
      ORT_ENFORCE(start_itr != ts_map.end() && end_itr != ts_map.end());
      for (auto itr = start_itr; itr != end_itr; ++itr) {
        events.push_back(CreateMemoryEvent(pid, *itr, name, offset, size, cname));
      }
      events.push_back(CreateMemoryEvent(pid, *end_itr, name, offset, size, cname));
    }
  }
}

void MemoryInfo::GenerateMemoryProfile() {
  //for (const auto& location_map : tensors_memory_info_map_) {
  //  if (location_map.first.device.Type() != OrtDevice::GPU) continue;
  //  //for (const auto& item : location_map.second.at(MapType::StaticActivation)) {
  //  //  const auto info = AllocPlan(item.first);
  //  //  //time_step_info_map_[info->lifetime_interval.first].AddTensor(item.second.planned_block);
  //  //  //time_step_info_map_[info->lifetime_interval.second + 1].RemoveTensor(item.second.planned_block);
  //  //}
  //}
  profiler.CreateEvents("GPU (initializers)", profiler.GetAndIncreasePid(), MapType::Initializer);
  profiler.CreateEvents("GPU (static activations)", profiler.GetAndIncreasePid(), MapType::StaticActivation);
  profiler.CreateEvents("GPU (dynamic activations)", profiler.GetAndIncreasePid(), MapType::DynamicActivation);
  profiler.CreateEvents("GPU (static activations) ", profiler.GetAndIncreasePid(), MapType::StaticActivation, "_grad");
  profiler.CreateEvents("GPU (dynamic activations) ", profiler.GetAndIncreasePid(), MapType::DynamicActivation, "_grad");

  // Write memory profile .json
  std::ofstream memory_profile("memory_profile_" + std::to_string(local_rank_) + ".json", std::ios::trunc);
  memory_profile << "[" << std::endl;
  for (size_t i = 0; i < profiler.events.size(); i++) {
    memory_profile << "  " << profiler.events[i];
    if (i < profiler.events.size() - 1) memory_profile << ",";
    memory_profile << std::endl;
  }
  memory_profile << "]" << std::endl;
  memory_profile.close();
}

}  // namespace onnxruntime
