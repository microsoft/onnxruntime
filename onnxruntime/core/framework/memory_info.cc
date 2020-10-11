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
  for (OrtValueIndex value_idx = 0; value_idx < execution_plan->allocation_plan.size(); ++value_idx) {
    //Only store tensor information
    if (!(execution_plan->allocation_plan[value_idx].value_type) || !(execution_plan->allocation_plan[value_idx].value_type->IsTensorType()))
      continue;
    AllocInfoPerTensor mem_info;
    mem_info.mlvalue_index = value_idx;
    value_name_idx_map.GetName(mem_info.mlvalue_index, mem_info.mlvalue_name);
    mem_info.lifetime_interval = execution_plan->allocation_plan[value_idx].life_interval;
    mem_info.alloctime_interval = execution_plan->allocation_plan[value_idx].allocate_interval;
    mem_info.reused_buffer = (execution_plan->allocation_plan[value_idx].alloc_kind != AllocKind::kReuse) ? value_idx : execution_plan->allocation_plan[value_idx].reused_buffer;
    mem_info.inplace_reuse = execution_plan->allocation_plan[value_idx].inplace_reuse;
    mem_info.alloc_kind = execution_plan->allocation_plan[value_idx].alloc_kind;
    mem_info.location = execution_plan->allocation_plan[value_idx].location;
    mem_info.map_type = execution_plan->allocation_plan[value_idx].alloc_kind == AllocKind::kAllocateStatically ? MapType::Initializer : MapType::StaticActivation;

    ORT_ENFORCE(mem_info.lifetime_interval.first <= mem_info.lifetime_interval.second);
    ORT_ENFORCE(mem_info.lifetime_interval.first >= mem_info.alloctime_interval.first &&
                mem_info.lifetime_interval.second <= mem_info.alloctime_interval.second);
    tensor_alloc_info_map_[value_idx] = std::move(mem_info);
  }
  return;
}

//Record the planned memory information
void MemoryInfo::RecordMemoryPatternInfo(const MemoryPatternGroup& mem_patterns, MapType type) {
  for (const auto& location : mem_patterns.locations) {
    for (const auto& p : mem_patterns.GetPatterns(location)->GetPatternsMap()) {
      tensors_memory_info_map_[type].AddPlannedMemory(p.first, p.second);
    }
  }
}

void MemoryInfo::RecordActivationPatternInfo(const MemoryPatternGroup& mem_patterns) {
  RecordMemoryPatternInfo(mem_patterns, MapType::StaticActivation);
  for (auto& item : tensors_memory_info_map_[MapType::StaticActivation]) {
    if (AllocPlan(item.first).alloc_kind == AllocKind::kReuse) {
      auto reuse_buffer = AllocPlan(item.first).reused_buffer;
      auto reuse_map_type = AllocPlan(reuse_buffer).map_type;
      auto& reused_meomry = tensors_memory_info_map_[reuse_map_type].GetPlannedMemory(reuse_buffer);
      tensors_memory_info_map_[MapType::StaticActivation].AddPlannedMemory(item.first, reused_meomry);
    }
  }
}

void MemoryInfo::RecordInitializerPatternInfo(const MemoryPatternGroup& mem_patterns) {
  RecordMemoryPatternInfo(mem_patterns, MapType::Initializer);
}

//Record the actual allocated tensor in the device
void MemoryInfo::RecordTensorDeviceAllocInfo(const OrtValueIndex idx, const OrtValue& value, const MapType& type) {
  auto& tensor = value.Get<Tensor>();
  auto sizeinbytes = tensor.SizeInBytes() % alignment ? tensor.SizeInBytes() + alignment - tensor.SizeInBytes() % alignment : tensor.SizeInBytes();
  MemoryBlock mb(size_t(tensor.DataRaw()), sizeinbytes);
  tensors_memory_info_map_[type].AddAllocMemory(idx, mb);
}

void MemoryInfo::RecordInitializerAllocInfo(const std::unordered_map<int, OrtValue>& tensor_map) {
  for (const auto& item : tensor_map) {
    RecordTensorDeviceAllocInfo(item.first, item.second, MapType::Initializer);
  }
}

void MemoryInfo::RecordActivationAllocInfo(const OrtValueIndex idx, const OrtValue& value) {
  auto reuse_buffer = AllocPlan(idx).reused_buffer;
  auto map_type = AllocPlan(reuse_buffer).map_type;
  RecordTensorDeviceAllocInfo(idx, value, map_type);
}

void MemoryInfo::SetDynamicAllocation(const OrtValueIndex idx) {
  AllocPlan(idx).map_type = MapType::DynamicActivation;
}

void PrintInforPerTensor(const MemoryInfo::AllocInfoPerTensor& alloc_info, const MemoryInfoPerTensor& mem_info, const size_t& rel_addr) {
  std::cout << "Tensor name: " << alloc_info.mlvalue_name << ", ";
  std::cout << "Index: " << alloc_info.mlvalue_index << ", ";
  std::cout << "Type: " << alloc_info.tensor_type << ", ";
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
  std::cout << "Initializer\n";
  const auto& initailizer_map = tensors_memory_info_map_[MapType::Initializer];
  for (auto& item : initailizer_map) {
    if (AllocPlan(item.first).location.device.Type() != location) continue;
    PrintInforPerTensor(AllocPlan(item.first), item.second, initailizer_map.GetAllocAddress(item.first));
  }
  std::cout << "Static Activation\n";
  const auto& static_activation_map = tensors_memory_info_map_[MapType::StaticActivation];
  for (auto& item : static_activation_map) {
    if (AllocPlan(item.first).location.device.Type() != location) continue;
    PrintInforPerTensor(AllocPlan(item.first), item.second, static_activation_map.GetAllocAddress(item.first));
  }

  std::cout << "Dynamic Activation\n";
  const auto& dynamic_activation_map = tensors_memory_info_map_[MapType::DynamicActivation];
  for (auto& item : dynamic_activation_map) {
    if (AllocPlan(item.first).location.device.Type() != location) continue;
    PrintInforPerTensor(AllocPlan(item.first), item.second, dynamic_activation_map.GetAllocAddress(item.first));
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

void MemoryInfo::GenerateMemoryProfile() {
  std::vector<std::string> color_names = {
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

  std::vector<std::string> events;

  // Metadata.
  const size_t initializers_pid = 0;
  const size_t static_activations_pid = 1;
  const size_t dynamic_activations_pid = 2;
  events.push_back(CreateMetadataEvent("GPU (initializers)", initializers_pid));
  events.push_back(CreateMetadataEvent("GPU (static activations)", static_activations_pid));
  events.push_back(CreateMetadataEvent("GPU (dynamic activations)", dynamic_activations_pid));

  //Static Activation
  const auto& static_activation_map = tensors_memory_info_map_.at(MapType::StaticActivation);
  for (const auto& item : static_activation_map) {
    const auto& info = AllocPlan(item.first);
    if (info.location.device.Type() != OrtDevice::GPU) continue;
    if (info.inplace_reuse) continue;

    const std::string& name = info.mlvalue_name;
    const std::string cname = color_names[events.size() % color_names.size()];
    size_t offset = static_activation_map.GetAllocAddress(item.first);
    size_t size = static_activation_map.GetAllocSize(item.first);
    size_t alloc_step = AllocPlan(item.first).lifetime_interval.first;  //alloc_kind == AllocKind::kReuse ? info.alloc_plan.life_interval.first + 1 : info.alloc_plan.life_interval.first;
    size_t dealloc_step = AllocPlan(item.first).lifetime_interval.second;
    for (size_t tid = alloc_step; tid <= dealloc_step; tid++) {
      events.push_back(CreateMemoryEvent(static_activations_pid, tid, name, offset, size, cname));
    }
  }

  //Dynamic Activation
  const auto& dynamic_activation_map = tensors_memory_info_map_.at(MapType::DynamicActivation);
  for (const auto& item : dynamic_activation_map) {
    const auto& info = AllocPlan(item.first);
    if (info.location.device.Type() != OrtDevice::GPU) continue;
    if (info.inplace_reuse) continue;

    const std::string& name = info.mlvalue_name;
    const std::string cname = color_names[events.size() % color_names.size()];
    size_t offset = dynamic_activation_map.GetAllocAddress(item.first);
    size_t size = dynamic_activation_map.GetAllocSize(item.first);
    size_t alloc_step = AllocPlan(item.first).lifetime_interval.first;  //alloc_kind == AllocKind::kReuse ? info.alloc_plan.life_interval.first + 1 : info.alloc_plan.life_interval.first;
    size_t dealloc_step = AllocPlan(item.first).lifetime_interval.second;
    for (size_t tid = alloc_step; tid <= dealloc_step; tid++) {
      events.push_back(CreateMemoryEvent(dynamic_activations_pid, tid, name, offset, size, cname));
    }
  }

  //Initalizer
  const auto& initializer_map = tensors_memory_info_map_.at(MapType::Initializer);
  for (const auto& item : initializer_map) {
    const auto& info = AllocPlan(item.first);
    if (info.location.device.Type() != OrtDevice::GPU) continue;
    const std::string& name = info.mlvalue_name;
    const std::string cname = color_names[events.size() % color_names.size()];
    size_t offset = initializer_map.GetAllocAddress(item.first);
    size_t size = initializer_map.GetAllocSize(item.first);
    size_t alloc_step = AllocPlan(item.first).lifetime_interval.first;  //alloc_kind == AllocKind::kReuse ? info.alloc_plan.life_interval.first + 1 : info.alloc_plan.life_interval.first;
    size_t dealloc_step = AllocPlan(item.first).lifetime_interval.second;
    for (size_t tid = alloc_step; tid <= dealloc_step; tid++) {
      events.push_back(CreateMemoryEvent(initializers_pid, tid, name, offset, size, cname));
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
