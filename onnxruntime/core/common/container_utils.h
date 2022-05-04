#pragma once

#include <type_traits>

#include "core/common/common.h"

namespace onnxruntime::container_utils {
template <typename Key, typename Value, typename... OtherContainerArgs,
          template <typename...> typename AssociativeContainer>
inline const Value* AtOrNull(const AssociativeContainer<Key, Value, OtherContainerArgs...>& container, const Key& key) {
  auto it = container.find(key);
  return it != container.end() ? &it->second : nullptr;
}

template <typename Key, typename Value, typename... OtherContainerArgs,
          template <typename...> typename AssociativeContainer>
inline const Value& At(const AssociativeContainer<Key, Value, OtherContainerArgs...>& container, const Key& key) {
  const auto* value = AtOrNull(container, key);
  ORT_ENFORCE(value != nullptr, "Key not found in associative container.");
  return *value;
}

template <typename Key, typename Value, typename... OtherContainerArgs,
          template <typename...> typename AssociativeContainer>
inline Value* MutableAtOrNull(AssociativeContainer<Key, Value, OtherContainerArgs...>& container, const Key& key) {
  static_assert(!std::is_const_v<Value>, "Associative container Value type cannot be const.");
  auto it = container.find(key);
  return it != container.end() ? &it->second : nullptr;
}

template <typename Key, typename Value, typename... OtherContainerArgs,
          template <typename...> typename AssociativeContainer>
inline Value& MutableAt(AssociativeContainer<Key, Value, OtherContainerArgs...>& container, const Key& key) {
  const auto* value = MutableAtOrNull(container, key);
  ORT_ENFORCE(value != nullptr, "Key not found in associative container.");
  return *value;
}
}
