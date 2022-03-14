// Copyright (c) Microsoft Corporation. All rights reserved.

#include <torch/extension.h>

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>

#define MTA_CHUNK_SIZE 2048 * 32

const size_t EMIT_NUM = 4;

// This will avoid the copies when doing implict Python list <==> C++ std::vector<> conversion.
PYBIND11_MAKE_OPAQUE(std::vector<at::Tensor>);

// This function is adapted from microsoft/DeepSpeed fused_adam_frontend.cpp
void multi_tensor_adam_cuda(int chunk_size,
                            at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr,
                            const float beta1,
                            const float beta2,
                            const float epsilon,
                            const int step,
                            const int mode,
                            const int bias_correction,
                            const float weight_decay);

// This function is adapted from NVIDIA/apex 
// https://github.com/NVIDIA/apex/blob/0c7d8e3fa9a095a1641a2290877436d0314b69c6/csrc/amp_C_frontend.cpp#L3
void multi_tensor_scale_cuda(int chunk_size,
                             at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>>& tensor_lists,
                             float scale);


// This function is adapted from NVIDIA/apex 
// https://github.com/NVIDIA/apex/blob/0c7d8e3fa9a095a1641a2290877436d0314b69c6/csrc/amp_C_frontend.cpp#L22
void multi_tensor_axpby_cuda(int chunk_size,
                             at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>>& tensor_lists,
                             float a,
                             float b,
                             int arg_to_check);


class MemoryBuffer {
    public:
        MemoryBuffer(size_t numel, at::Tensor val){
            data_buffer_ = at::empty({numel}, val.options());
        }

        at::Tensor Get(at::Tensor param, size_t start_index) {
            size_t end_index = start_index + param.numel();
            return data_buffer_.slice(0, start_index, end_index).view(param.sizes());
        }

    private:
        at::Tensor data_buffer_;
};

class CachedStates {
    public:
        static CachedStates& GetInstance(){
            static CachedStates states_;
            return states_;
        }

        void ClearStates(){
            idx_to_numel_map.clear();
        }

        // Parameter index to number of element mapping for each parameter. 
        std::vector<std::pair<size_t, size_t>> idx_to_numel_map;

    private:
        CachedStates(){}
};

bool SortByElementSizeDesc(const std::pair<size_t, size_t> &a,
                           const std::pair<size_t, size_t> &b) {
    return (a.second > b.second);
};

// This function is trying to move into C++ implementation from Python logic
// https://github.com/NVIDIA/apex/blob/0c7d8e3fa9a095a1641a2290877436d0314b69c6/apex/amp/_process_optimizer.py#L161.
// This would reduce the overhead of long loops.
void unscale_fp16_grads_into_fp32_grads(std::vector<at::Tensor>& all_fp16_params,
                                        std::vector<at::Tensor>& all_fp32_from_fp16_params,
                                        at::Tensor is_overflow_buffer,
                                        float scale) {
    if (all_fp16_params.size() == 0 || all_fp32_from_fp16_params.size() == 0) {
        return;
    }

    const float inv_scale = 1.0 / scale;
    TORCH_CHECK(all_fp16_params.size() == all_fp32_from_fp16_params.size(), 
                "mismatch param size between fp16_param and fp32_from_fp16_param.");

    // Use cached states only parameter count did not get changed.
    bool need_reset_states = 
        all_fp32_from_fp16_params.size() != CachedStates::GetInstance().idx_to_numel_map.size();
    if (need_reset_states) {
        CachedStates::GetInstance().ClearStates();
    }

    std::vector<at::Tensor> fp16_grads_needing_unscale;
    std::vector<at::Tensor> fp16_grads_needing_unscale_with_stash;
    std::vector<at::Tensor> preexisting_fp32_grads;

    // Parameter index to parameter mapping for each fp32_from_fp16 parameter.
    std::unordered_map<size_t, at::Tensor> idx_to_fp32_from_fp16_params;

    // "buffer index" to "offset in memory buffer" mapping for each fp32_from_fp16 parameter.
    std::vector<size_t> memory_buffer_idx_to_offset_map;
    size_t memory_buffer_size = 0;
    auto& idx_to_numel_map = CachedStates::GetInstance().idx_to_numel_map;

    for (size_t idx = 0; idx < all_fp16_params.size(); ++idx) {
        auto& fp16_param_grad = all_fp16_params[idx].grad();
        bool fp16_param_has_grad = fp16_param_grad.defined();

        auto& fp32_from_fp16_param = all_fp32_from_fp16_params[idx];
        auto& fp32_from_fp16_param_grad = fp32_from_fp16_param.grad();
        bool fp32_from_fp16_param_has_grad = fp32_from_fp16_param_grad.defined();

        size_t num_elem = fp32_from_fp16_param.numel();
        if (need_reset_states) {
            idx_to_numel_map.push_back(std::make_pair(idx, num_elem));
        }

        if (fp16_param_has_grad && !fp32_from_fp16_param_has_grad) {
            idx_to_fp32_from_fp16_params.emplace(std::make_pair(idx, fp32_from_fp16_param));
            fp16_grads_needing_unscale.emplace_back(fp16_param_grad);
            memory_buffer_idx_to_offset_map.emplace_back(memory_buffer_size);
            memory_buffer_size += num_elem;
        } else if (fp16_param_has_grad && fp32_from_fp16_param_has_grad) {
            fp16_grads_needing_unscale_with_stash.emplace_back(fp16_param_grad);
            preexisting_fp32_grads.emplace_back(fp32_from_fp16_param_grad);
        }
    }

    if (need_reset_states) {
        std::sort(idx_to_numel_map.begin(), idx_to_numel_map.end(), SortByElementSizeDesc);
    }

    if (idx_to_fp32_from_fp16_params.size() > 0) {
        auto mem_buffer = MemoryBuffer(memory_buffer_size, idx_to_fp32_from_fp16_params.begin()->second);
        const size_t emit_threshhold = memory_buffer_size / EMIT_NUM;

        size_t acc_size = 0;
        std::vector<at::Tensor> partial_new_fp32_grads;
        std::vector<at::Tensor> partial_fp16_grads_needing_unscale;
        for (size_t idx = 0, fp32_from_fp16_param_idx = 0; idx < idx_to_numel_map.size(); ++idx) {
            if (idx_to_fp32_from_fp16_params.find(idx) == idx_to_fp32_from_fp16_params.end()) {
                continue;
            }

            acc_size += idx_to_numel_map[idx].second;
            idx_to_fp32_from_fp16_params[idx].mutable_grad() = 
                mem_buffer.Get(idx_to_fp32_from_fp16_params[idx],
                memory_buffer_idx_to_offset_map[fp32_from_fp16_param_idx]);
            partial_new_fp32_grads.emplace_back(idx_to_fp32_from_fp16_params[idx].grad());
            partial_fp16_grads_needing_unscale.emplace_back(fp16_grads_needing_unscale[fp32_from_fp16_param_idx]);

            if (acc_size > emit_threshhold || fp32_from_fp16_param_idx == idx_to_fp32_from_fp16_params.size() - 1) {
                if (partial_fp16_grads_needing_unscale.size() > 0) {
                    std::vector<std::vector<at::Tensor>> tensor_lists;
                    tensor_lists.emplace_back(partial_fp16_grads_needing_unscale);
                    tensor_lists.emplace_back(partial_new_fp32_grads);
                    multi_tensor_scale_cuda(MTA_CHUNK_SIZE, is_overflow_buffer, tensor_lists, inv_scale);

                    partial_fp16_grads_needing_unscale.clear();
                    partial_new_fp32_grads.clear();
                    acc_size = 0;
                }
            }
            ++fp32_from_fp16_param_idx;
        }
    }

    if (fp16_grads_needing_unscale_with_stash.size() > 0) {
        std::vector<std::vector<at::Tensor>> tensor_lists;
        tensor_lists.emplace_back(fp16_grads_needing_unscale_with_stash);
        tensor_lists.emplace_back(preexisting_fp32_grads);
        tensor_lists.emplace_back(preexisting_fp32_grads);
        // a * x + b * y
        multi_tensor_axpby_cuda(MTA_CHUNK_SIZE, is_overflow_buffer, tensor_lists, inv_scale, float(1.0), 0);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // Cannot use the shortcut API below because https://github.com/pybind/pybind11/issues/1470
    // py::bind_vector<std::vector<at::Tensor>>(m, "TorchTensorVector");
    py::class_<std::vector<at::Tensor>>(m, "TorchTensorVector")
        .def(py::init<>())
        .def("clear", &std::vector<at::Tensor>::clear)
        .def("pop_back", &std::vector<at::Tensor>::pop_back)
        .def("__len__", [](const std::vector<at::Tensor> &v) { return v.size(); })
        .def("__iter__", [](std::vector<at::Tensor> &v) {
            return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0, 1>())
        .def("extend", [](std::vector<at::Tensor> &v, const std::vector<at::Tensor> &src) {
            v.insert(v.end(), src.begin(), src.end());
        })
        .def(py::init([](const py::iterable &it) {
            auto v = std::unique_ptr<std::vector<at::Tensor>>(new std::vector<at::Tensor>());
            v->reserve(py::len_hint(it));
            for (py::handle h : it) {
                v->push_back(h.cast<at::Tensor>());
            }
            return v.release();
        }));

    m.def("multi_tensor_adam",
          &multi_tensor_adam_cuda,
          "Compute and apply gradient update to parameters for Adam optimizer");
    m.def("unscale_fp16_grads_into_fp32_grads",
          &unscale_fp16_grads_into_fp32_grads,
          "Unscale those fp16 gradients into fp32 gradient buffers.");
}
