// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)

#include "core/framework/memory_info.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/ort_value.h"

#include <fstream>
#include <map>
#include <numeric>
#include <queue>
#include <set>

namespace onnxruntime {

// Record allocation information for each tensor, based on the execution plan
void MemoryInfo::Init(const SequentialExecutionPlan* execution_plan,
                      const OrtValueNameIdxMap& value_name_idx_map) {
  num_node_size_ = execution_plan->execution_plan.size();
  for (OrtValueIndex value_idx = 0; value_idx < OrtValueIndex(execution_plan->allocation_plan.size()); ++value_idx) {
    // Only store tensor information
    if (!(execution_plan->allocation_plan[value_idx].value_type) ||
        !(execution_plan->allocation_plan[value_idx].value_type->IsTensorType()))
      continue;

    AllocInfoPerTensor mem_info;
    mem_info.mlvalue_index = value_idx;
    ORT_THROW_IF_ERROR(value_name_idx_map.GetName(mem_info.mlvalue_index, mem_info.mlvalue_name));
    mem_info.lifetime_interval = execution_plan->allocation_plan[value_idx].life_interval;
    mem_info.reused_buffer = (execution_plan->allocation_plan[value_idx].alloc_kind != AllocKind::kReuse)
                                 ? value_idx
                                 : execution_plan->allocation_plan[value_idx].reused_buffer;

    // If the tensor is using memory outside of the scope, do not store it
    if (execution_plan->allocation_plan[mem_info.reused_buffer].alloc_kind == AllocKind::kPreExisting) continue;
    if (execution_plan->allocation_plan[mem_info.reused_buffer].alloc_kind == AllocKind::kAllocateOutput) continue;
    if (execution_plan->allocation_plan[mem_info.reused_buffer].alloc_kind == AllocKind::kAllocatedExternally) continue;
    mem_info.inplace_reuse = (execution_plan->allocation_plan[value_idx].inplace_reuse != -1 &&
                              execution_plan->allocation_plan[value_idx].inplace_reuse != value_idx);
    mem_info.alloc_kind = execution_plan->allocation_plan[value_idx].alloc_kind;
    mem_info.location = execution_plan->allocation_plan[value_idx].location;

    ORT_ENFORCE(mem_info.lifetime_interval.first <= mem_info.lifetime_interval.second);
    tensor_alloc_info_map_[value_idx] = std::move(mem_info);
    tensors_memory_info_map_[mem_info.location];
  }
  return;
}

// Record the planned memory information
void MemoryInfo::RecordMemoryPatternInfo(const MemoryPatternGroup& mem_patterns, MapType type) {
  for (const auto& location : mem_patterns.locations) {
    for (const auto& p : mem_patterns.GetPatterns(location)->GetPatternsMap()) {
      ORT_ENFORCE(AllocPlan(p.first));
      tensors_memory_info_map_.at(location)[type].AddPlannedMemory(p.first, p.second);
    }
  }
}

void MemoryInfo::RecordPatternInfo(const MemoryPatternGroup& mem_patterns, const MapType& type) {
  RecordMemoryPatternInfo(mem_patterns, type);
  for (auto& item : tensor_alloc_info_map_) {
    if (item.second.alloc_kind == AllocKind::kReuse) {
      auto reuse_buffer = AllocPlan(item.first)->reused_buffer;
      auto& map = tensors_memory_info_map_.at(AllocPlan(item.first)->location);
      if (map[type].Contain(reuse_buffer)) {
        auto& reused_memory = map[type].GetPlannedMemory(reuse_buffer);
        map[type].AddPlannedMemory(item.first, reused_memory);
      }
    }
  }
}

// Record the actual allocated tensor in the device
void MemoryInfo::RecordTensorDeviceAllocInfo(const OrtValueIndex idx, const OrtValue& value, const MapType& type) {
  if (tensor_alloc_info_map_.find(idx) == tensor_alloc_info_map_.end()) return;
  ORT_ENFORCE(value.IsTensor(), "Memory profiler only supports tensor type.");
  auto& tensor = value.Get<Tensor>();
  auto tensor_size_in_bytes = tensor.SizeInBytes();
  auto remainder = tensor_size_in_bytes % alignment_;
  auto size_in_bytes = tensor_size_in_bytes % alignment_ ? tensor_size_in_bytes + alignment_ - remainder
                                                         : tensor_size_in_bytes;
  MemoryBlock mb((size_t)(tensor.DataRaw()), size_in_bytes);

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
  // Some activations are not tracked in GenerateTensorMap, for example some pre-existing tensors are re-used with
  // Alias by some operators, it will reuse this buffer and the buffer will not be tracked in GenerateTensorMap.
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
    std::cout << "Find no map type for reuse_buffer: " << reuse_buffer << ", so skipping" << std::endl;

  RecordTensorDeviceAllocInfo(idx, value, map_type);
}

void MemoryInfo::SetDynamicAllocation(const OrtValueIndex idx) {
  if (!AllocPlan(idx)) return;
  auto& da_map = tensors_memory_info_map_.at(AllocPlan(idx)->location)[MapType::DynamicActivation];
  if (!da_map.Contain(idx)) da_map[idx];
}

void PrintInforPerTensor(const MemoryInfo::AllocInfoPerTensor& alloc_info, const MemoryInfoPerTensor& mem_info,
                         const size_t& rel_addr) {
  std::cout << "Tensor name: " << alloc_info.mlvalue_name << ", ";
  std::cout << "Index: " << alloc_info.mlvalue_index << ", ";
  std::cout << "Reuse inplace: " << alloc_info.inplace_reuse << ", ";
  std::cout << "Alloc type: " << alloc_info.alloc_kind << ", ";
  std::cout << "Location: " << alloc_info.location.name << ", ";
  std::cout << "lifetime: (" << alloc_info.lifetime_interval.first << ", " << alloc_info.lifetime_interval.second << "), ";
  std::cout << "planned block: (" << mem_info.planned_block.offset_ << ", "
            << (mem_info.planned_block.offset_ + mem_info.planned_block.size_) << "), ";
  std::cout << "planned size: " << mem_info.planned_block.size_ << ", ";
  std::cout << "allocated block: (" << rel_addr << ", "
            << (rel_addr + mem_info.allocated_block.size_) << "), ";
  std::cout << "allocated size: " << mem_info.allocated_block.size_ << "\n";
}

void MemoryInfo::PrintMemoryInfoForLocation(const OrtDevice::DeviceType location) {
  for (const auto& map : tensors_memory_info_map_) {
    if (map.first.device.Type() != location) continue;
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

std::string MemoryProfiler::CreateMetadataEvent(const std::string& process_name, size_t process_id) {
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

std::string MemoryProfiler::CreateMemoryEvent(size_t pid, size_t tid, const std::string& name,
                                              size_t offset, size_t size,
                                              const std::string& color_name) {
  std::stringstream evt;
  evt << "{";
  evt << "\"ph\":\"X\",";
  evt << "\"pid\":" << pid << ",";
  evt << "\"tid\":" << tid << ",";
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

std::string MemoryProfiler::CreateSummaryEvent(size_t pid, size_t tid, const MemoryInfo::AllocationSummary& summary,
                                               size_t size, size_t bytes_for_pattern) {
  const size_t total_bytes = summary.total_size;
  const size_t used_bytes = summary.used_size;
  const size_t free_bytes = total_bytes - used_bytes;
  const float used_percent = (float)used_bytes / (float)total_bytes;
  const float free_percent = (float)free_bytes / (float)total_bytes;

  std::stringstream evt;
  evt << "{";
  evt << "\"ph\":\"X\",";
  evt << "\"pid\":" << pid << ",";
  evt << "\"tid\":" << tid << ",";
  evt << "\"ts\":" << -(int64_t)size << ",";
  evt << "\"dur\":" << size << ",";
  evt << "\"name\":\"Summary\",";
  evt << "\"cname\":\"black\",";
  evt << "\"args\":{";
  evt << "\"total_bytes\":" << total_bytes << ",";
  evt << "\"used_bytes\":" << used_bytes << ",";
  evt << "\"free_bytes\":" << free_bytes << ",";
  evt << "\"used_percent\":" << used_percent << ",";
  evt << "\"free_percent\":" << free_percent << ",";
  evt << "\"bytes for pattern\":" << bytes_for_pattern;
  evt << "}";
  evt << "}";
  return evt.str();
}

static bool IsStaticType(const MemoryInfo::MapType& map_type) {
  return map_type == MemoryInfo::MapType::Initializer || map_type == MemoryInfo::MapType::StaticActivation;
}

static void UpdateSummary(MemoryInfo::AllocationSummary& summary, size_t alloc_offset, size_t alloc_size,
                          const OrtValueIndex idx, const MemoryInfo::MapType& map_type) {
  summary.total_size = std::max(summary.total_size, alloc_offset + alloc_size);
  summary.used_size += alloc_size;
  if (!IsStaticType(map_type)) {
    summary.total_size = summary.used_size;
  } else {
    summary.total_size = std::max(summary.total_size, alloc_offset + alloc_size);
  }
  summary.live_tensors.push_back(idx);
}

// Create sessions in the profiler.
// p_name: session name
// pid: sessionid
// map_type: Initializer, static_activation, dynamic_activation.
//     We have this separtion because they are using different memory offsets.
// group_name: The group_name that is recorded previously using function "AddRecordingTensorGroup".
//     Used for generating customized sessions for a group of tensors
// Top_k: The steps with the top-k highest memory consumptions are plot. When top_k == 0, we plot all the steps
// device_type: The type of the device where the tensors are.
void MemoryProfiler::CreateEvents(const std::string& p_name,
                                  const size_t pid,
                                  const MemoryInfo::MapType& map_type,
                                  const std::string& group_name,
                                  const size_t top_k,
                                  const OrtDevice::DeviceType device_type) {
  // Metadata.
  std::string pid_name_internal = "device_" + std::to_string(device_type) + "_" + p_name + group_name;
  events.push_back(CreateMetadataEvent(pid_name_internal, pid));
  size_t summary_size = 10;
  std::hash<std::string> str_hash;

  // Create Event for each tensor
  auto& time_step_trace = GetMemoryInfo().time_step_trace_;
  for (const auto& location_map : GetMemoryInfo().tensors_memory_info_map_) {
    const OrtMemoryInfo& memory_info = location_map.first;
    const auto& maptype_to_map_mapping = location_map.second;

    if (memory_info.device.Type() != device_type) continue;

    // If there is no tensor of a map_type, we skip creating event for that map_type
    if (maptype_to_map_mapping.find(map_type) == maptype_to_map_mapping.end()) continue;

    auto summary_key = str_hash(memory_info.ToString() + std::to_string(map_type));
    // Preprocessing
    if (time_step_trace[map_type].empty()) {
      for (const auto& item : maptype_to_map_mapping.at(map_type)) {
        time_step_trace[map_type].insert(GetMemoryInfo().AllocPlan(item.first)->lifetime_interval.first);
        time_step_trace[map_type].insert(GetMemoryInfo().AllocPlan(item.first)->lifetime_interval.second);
      }
    }

    const auto& map = maptype_to_map_mapping.at(map_type);
    // Update summary
    if (summary_.find(summary_key) == summary_.end()) {
      for (const auto& item : map) {
        const OrtValueIndex idx = item.first;
        const auto info = GetMemoryInfo().AllocPlan(idx);
        if (info->inplace_reuse) continue;

        size_t offset = IsStaticType(map_type) ? map.GetPlannedAddress(idx) : map.GetAllocAddress(idx);
        size_t size = IsStaticType(map_type) ? map.GetPlannedSize(idx) : map.GetAllocSize(idx);
        size_t alloc_step = GetMemoryInfo().AllocPlan(idx)->lifetime_interval.first;
        size_t dealloc_step = GetMemoryInfo().AllocPlan(idx)->lifetime_interval.second;
        const auto& ts_map = GetMemoryInfo().time_step_trace_[map_type];
        const auto& start_itr = ts_map.find(alloc_step);
        const auto& end_itr = ts_map.find(dealloc_step);
        ORT_ENFORCE(start_itr != ts_map.end(),
                    "The allocation step is not recorded: ", alloc_step, ", ortvalue_index: ", idx);
        ORT_ENFORCE(end_itr != ts_map.end(),
                    "The deallocation step is not recorded: ", dealloc_step, ", ortvalue_index: ", idx);

        for (auto itr = start_itr; itr != end_itr; ++itr) {
          UpdateSummary(summary_[summary_key][*itr], offset, size, idx, map_type);
        }
        UpdateSummary(summary_[summary_key][*end_itr], offset, size, idx, map_type);
      }
    }

    // extract top_k total size
    size_t top_kth_total_size = 0;
    if (top_k != 0) {
      std::set<size_t> top_k_set;
      for (const auto& item : summary_[summary_key]) {
        if (top_k_set.size() < top_k)
          top_k_set.insert(item.second.total_size);
        else if (*top_k_set.cbegin() < item.second.total_size) {
          top_k_set.erase(top_k_set.begin());
          top_k_set.insert(item.second.total_size);
        }
      }
      top_kth_total_size = *top_k_set.cbegin();
    }

    for (const auto& item : summary_[summary_key]) {
      bool has_other_tensors = false;
      if (top_k != 0 && item.second.total_size < top_kth_total_size) continue;
      size_t alloc_size_for_pattern = 0;
      for (const auto& live_tensor : item.second.live_tensors) {
        const auto info = GetMemoryInfo().AllocPlan(live_tensor);
        if (info->inplace_reuse) continue;
        const std::string& name = info->mlvalue_name;
        // Filter out string without a certain name
        if (!group_name.empty()) {
          if (!GetMemoryInfo().InRecordingTensorGroup(group_name, name)) continue;
        }
        const std::string cname = color_names[str_hash(name) % color_names.size()];
        // Sometimes a tensor can be both statically planned and dynamically allocated, so we need to use planned
        // address/size in static_activation type
        size_t offset = IsStaticType(map_type) ? map.GetPlannedAddress(live_tensor) : map.GetAllocAddress(live_tensor);
        size_t size = IsStaticType(map_type) ? map.GetPlannedSize(live_tensor) : map.GetAllocSize(live_tensor);
        alloc_size_for_pattern += size;
        events.push_back(CreateMemoryEvent(pid, item.first, name, offset, size, cname));
        has_other_tensors = true;
      }
      // If for that time steps, we have other tensors to plot other than just summary, add summary events.
      // These will show up visually as 10 % of the max width in a section.
      if (has_other_tensors) {
        summary_size = std::max(summary_size, item.second.total_size / 10);
        events.push_back(CreateSummaryEvent(pid, item.first, item.second, summary_size, alloc_size_for_pattern));
      }
    }
  }
}

void MemoryProfiler::GenerateMemoryProfile() {
  // Write memory profile .json
  std::stringstream ss;
  ss << "memory_profile_" << GetMemoryInfo().GetLocalRank() << "_" << profiler_id_ << ".json";
  std::ofstream memory_profile(ss.str(), std::ios::trunc);
  memory_profile << "[" << std::endl;
  for (size_t i = 0; i < GetEvents().size(); i++) {
    memory_profile << "  " << GetEvents().at(i);
    if (i < GetEvents().size() - 1) memory_profile << ",";
    memory_profile << std::endl;
  }
  memory_profile << "]" << std::endl;
  memory_profile.close();
}

}  // namespace onnxruntime
#endif
