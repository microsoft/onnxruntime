// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/ort_api.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace onnxruntime {

#if BUILD_QNN_EP_STATIC_LIB
static std::unique_ptr<std::vector<std::function<void()>>> s_run_on_unload_;

void RunOnUnload(std::function<void()> function) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard(mutex);
  if (!s_run_on_unload_) {
    s_run_on_unload_ = std::make_unique<std::vector<std::function<void()>>>();
  }
  s_run_on_unload_->push_back(std::move(function));
}

struct OnUnload {
  ~OnUnload() {
    if (!s_run_on_unload_)
      return;

    for (auto& function : *s_run_on_unload_)
      function();

    s_run_on_unload_.reset();
  }

} g_on_unload;
#endif  // BUILD_QNN_EP_STATIC_LIB

std::vector<const Node*> Graph__Nodes(const Graph& graph) {
#if BUILD_QNN_EP_STATIC_LIB
  std::vector<const Node*> nodes;
  nodes.reserve(graph.NumberOfNodes());

  for (const Node& node : graph.Nodes()) {
    nodes.push_back(&node);
  }

  return nodes;
#else
  return graph.Nodes();
#endif
}

#if BUILD_QNN_EP_STATIC_LIB
#define NODE_ATTR_ITER_VAL(iter) (iter)->second
#else
#define NODE_ATTR_ITER_VAL(iter) (iter)->second()
#endif

NodeAttrHelper::NodeAttrHelper(const onnxruntime::Node& node)
    : node_attributes_(node.GetAttributes()) {}

NodeAttrHelper::NodeAttrHelper(const NodeUnit& node_unit)
    : node_attributes_(node_unit.GetNode().GetAttributes()) {}

float NodeAttrHelper::Get(const std::string& key, float def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return NODE_ATTR_ITER_VAL(entry).f();
  }

  return def_val;
}

int32_t NodeAttrHelper::Get(const std::string& key, int32_t def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return narrow<int32_t>(NODE_ATTR_ITER_VAL(entry).i());
  }

  return def_val;
}

uint32_t NodeAttrHelper::Get(const std::string& key, uint32_t def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return narrow<uint32_t>(NODE_ATTR_ITER_VAL(entry).i());
  }

  return def_val;
}

int64_t NodeAttrHelper::Get(const std::string& key, int64_t def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return NODE_ATTR_ITER_VAL(entry).i();
  }

  return def_val;
}

const std::string& NodeAttrHelper::Get(const std::string& key, const std::string& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return NODE_ATTR_ITER_VAL(entry).s();
  }

  return def_val;
}

std::vector<std::string> NodeAttrHelper::Get(const std::string& key, const std::vector<std::string>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    std::vector<std::string> res;
    for (int i = 0; i < NODE_ATTR_ITER_VAL(entry).strings_size(); i++) {
      res.emplace_back(NODE_ATTR_ITER_VAL(entry).strings(i));
    }
    return res;
  }

  return def_val;
}

std::vector<int32_t> NodeAttrHelper::Get(const std::string& key, const std::vector<int32_t>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
    const int64_t* cbegin = values.data();
    const int64_t* cend = values.data() + values.size();
    std::vector<int32_t> v;
    v.reserve(static_cast<size_t>(values.size()));
    std::transform(cbegin, cend, std::back_inserter(v),
                   [](int64_t val) -> int32_t { return narrow<int32_t>(val); });
    return v;
  }

  return def_val;
}

std::vector<uint32_t> NodeAttrHelper::Get(const std::string& key, const std::vector<uint32_t>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
    const int64_t* cbegin = values.data();
    const int64_t* cend = values.data() + values.size();
    std::vector<uint32_t> v;
    v.reserve(static_cast<size_t>(values.size()));
    std::transform(cbegin, cend, std::back_inserter(v),
                   [](int64_t val) -> uint32_t { return narrow<uint32_t>(val); });
    return v;
  }

  return def_val;
}

std::vector<int64_t> NodeAttrHelper::Get(const std::string& key, const std::vector<int64_t>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
    const int64_t* cbegin = values.data();
    const int64_t* cend = values.data() + values.size();
    return std::vector<int64_t>{cbegin, cend};
  }

  return def_val;
}

std::vector<float> NodeAttrHelper::Get(const std::string& key, const std::vector<float>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).floats();
    const float* cbegin = values.data();
    const float* cend = values.data() + values.size();
    return std::vector<float>{cbegin, cend};
  }

  return def_val;
}

std::optional<float> NodeAttrHelper::GetFloat(const std::string& key) const {
  std::optional<float> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    result = NODE_ATTR_ITER_VAL(entry).f();
  }

  return result;
}

std::optional<int64_t> NodeAttrHelper::GetInt64(const std::string& key) const {
  std::optional<int64_t> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    result = NODE_ATTR_ITER_VAL(entry).i();
  }

  return result;
}

std::optional<std::vector<float>> NodeAttrHelper::GetFloats(const std::string& key) const {
  std::optional<std::vector<float>> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).floats();
    const float* cbegin = values.data();
    const float* cend = values.data() + values.size();
    result = std::vector<float>(cbegin, cend);
  }

  return result;
}

std::optional<std::vector<int64_t>> NodeAttrHelper::GetInt64s(const std::string& key) const {
  std::optional<std::vector<int64_t>> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
    const int64_t* cbegin = values.data();
    const int64_t* cend = values.data() + values.size();
    result = std::vector<int64_t>(cbegin, cend);
  }

  return result;
}

std::optional<std::string> NodeAttrHelper::GetString(const std::string& key) const {
  std::optional<std::string> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    result = NODE_ATTR_ITER_VAL(entry).s();
  }

  return result;
}

bool NodeAttrHelper::HasAttr(const std::string& key) const {
  return node_attributes_.find(key) != node_attributes_.end();
}
}  // namespace onnxruntime
