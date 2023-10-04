#pragma once

#ifndef MICROSOFT_AI_MACHINELEARNING_H_
#define MICROSOFT_AI_MACHINELEARNING_H_

#define ML_FAIL_FAST_IF(condition) \
  do {                             \
    bool _cond = condition;        \
    if (_cond) {                   \
      __fastfail(0);               \
    }                              \
  } while (0)

namespace Microsoft {
namespace AI {
namespace MachineLearning {
using tensor_shape_type = int64_t;
}
}  // namespace AI
}  // namespace Microsoft

#include "winml_microsoft.h"

namespace Microsoft {
namespace AI {
namespace MachineLearning {
namespace Details {
using learning_model = WinMLLearningModel;
using learning_model_device = WinMLLearningModelDevice;
using learning_model_session = WinMLLearningModelSession;
using learning_model_binding = WinMLLearningModelBinding;
using learning_model_results = WinMLLearningModelResults;
}  // namespace Details
}  // namespace MachineLearning
}  // namespace AI
}  // namespace Microsoft

namespace Microsoft {
namespace AI {
namespace MachineLearning {

struct learning_model {
  friend struct learning_model_session;

  learning_model(const wchar_t* model_path, size_t size) : m_model(model_path, size) {}

  learning_model(const char* bytes, size_t size) : m_model(bytes, size) {}

 private:
  Details::learning_model m_model;
};

struct learning_model_results {
  friend struct learning_model_session;

  int32_t get_output(const wchar_t* feature_name, size_t feature_name_size, void** pp_buffer, size_t* p_capacity) {
    return m_results.get_output(feature_name, feature_name_size, pp_buffer, p_capacity);
  }

 private:
  learning_model_results(Details::learning_model_results results) : m_results(results) {}

 private:
  Details::learning_model_results m_results;
};

struct learning_model_device {
  friend struct learning_model_session;

  learning_model_device() : m_device() {}

  learning_model_device(learning_model_device&& device) : m_device(std::move(device.m_device)) {}

  learning_model_device(learning_model_device& device) : m_device(device.m_device) {}

  void operator=(learning_model_device& device) { m_device = device.m_device; }

 protected:
  learning_model_device(Details::learning_model_device&& learning_model_device)
    : m_device(std::move(learning_model_device)) {}

 private:
  Details::learning_model_device m_device;
};

struct learning_model_session {
  friend struct learning_model_binding;

  learning_model_session(const learning_model& model) : m_session(model.m_model) {}

  learning_model_session(const learning_model& model, const learning_model_device& device)
    : m_session(model.m_model, device.m_device) {}

  inline learning_model_results evaluate(learning_model_binding& binding);

 private:
  Details::learning_model_session m_session;
};

struct learning_model_binding {
  friend struct learning_model_session;

  learning_model_binding(const learning_model_session& session) : m_binding(session.m_session) {}

  template <typename T>
  int32_t bind_as_reference(
    const wchar_t* feature_name,
    size_t feature_name_size,
    tensor_shape_type* p_shape,
    size_t shape_size,
    T* p_data,
    size_t data_size
  ) {
    return m_binding.bind_as_reference<T>(feature_name, feature_name_size, p_shape, shape_size, p_data, data_size);
  }

  template <typename T = float>
  int32_t bind_as_references(
    const wchar_t* feature_name, size_t feature_name_size, T** p_data, size_t* data_sizes, size_t num_buffers
  ) {
    return m_binding.bind_as_references<T>(feature_name, feature_name_size, p_data, data_sizes, num_buffers);
  }

  template <typename T>
  int32_t bind(
    const wchar_t* feature_name,
    size_t feature_name_size,
    tensor_shape_type* p_shape,
    size_t shape_size,
    T* p_data,
    size_t data_size
  ) {
    return m_binding.bind<T>(feature_name, feature_name_size, p_shape, shape_size, p_data, data_size);
  }

  template <typename T = float>
  int32_t bind(const wchar_t* feature_name, size_t feature_name_size, tensor_shape_type* p_shape, size_t shape_size) {
    return m_binding.bind<T>(feature_name, feature_name_size, p_shape, shape_size);
  }

 private:
  Details::learning_model_binding m_binding;
};

learning_model_results learning_model_session::evaluate(learning_model_binding& binding) {
  return Details::learning_model_results(m_session.evaluate(binding.m_binding));
}

}  // namespace MachineLearning
}  // namespace AI
}  // namespace Microsoft

#endif  // MICROSOFT_AI_MACHINELEARNING_H_
