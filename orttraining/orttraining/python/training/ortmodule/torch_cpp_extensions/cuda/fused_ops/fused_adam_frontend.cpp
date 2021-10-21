// Copyright (c) Microsoft Corporation. All rights reserved.

#include <torch/extension.h>

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

const int fixed_chunk_size = 2048 * 32;
// This function is trying to move into C++ implementation from Python logic
// https://github.com/NVIDIA/apex/blob/0c7d8e3fa9a095a1641a2290877436d0314b69c6/apex/amp/_process_optimizer.py#L161.
// This would reduce the overhead of long loops.
void unscale_fp16_grads_into_fp32_grads(std::vector<at::Tensor>& all_fp16_params, 
                                        std::vector<at::Tensor>& all_fp32_from_fp16_params,
                                        at::Tensor is_overflow_buffer,
                                        float scale) {
    const float inv_scale = 1.0 / scale;
    TORCH_CHECK(all_fp16_params.size() == all_fp32_from_fp16_params.size(), 
                "mismatch param size between fp16_param and fp32_from_fp16_param.");
    std::vector<at::Tensor> fp16_grads_needing_unscale; 
    std::vector<at::Tensor> new_fp32_grads;
    std::vector<at::Tensor> fp16_grads_needing_unscale_with_stash;
    std::vector<at::Tensor> preexisting_fp32_grads;

    for (size_t i = 0; i < all_fp16_params.size(); ++i) {
        auto& fp16_param_grad = all_fp16_params[i].grad();
        bool fp16_param_has_grad = fp16_param_grad.defined();

        auto& fp32_from_fp16_param = all_fp32_from_fp16_params[i];
        auto& fp32_from_fp16_param_grad = fp32_from_fp16_param.grad();
        bool fp32_from_fp16_param_has_grad = fp32_from_fp16_param_grad.defined();

        if (fp16_param_has_grad && !fp32_from_fp16_param_has_grad) {
            fp32_from_fp16_param.mutable_grad() = at::empty_like(fp32_from_fp16_param);
            fp16_grads_needing_unscale.emplace_back(fp16_param_grad);
            new_fp32_grads.emplace_back(fp32_from_fp16_param.grad());
        } else if (fp16_param_has_grad && fp32_from_fp16_param_has_grad) {
            fp16_grads_needing_unscale_with_stash.emplace_back(fp16_param_grad);
            preexisting_fp32_grads.emplace_back(fp32_from_fp16_param_grad);
        }
    }

    if (fp16_grads_needing_unscale.size() > 0) {
        std::vector<std::vector<at::Tensor>> tensor_lists {fp16_grads_needing_unscale, new_fp32_grads};
        multi_tensor_scale_cuda(fixed_chunk_size, is_overflow_buffer, tensor_lists, inv_scale);
    }

    if (fp16_grads_needing_unscale_with_stash.size() > 0) {
        std::vector<std::vector<at::Tensor>> tensor_lists {
            fp16_grads_needing_unscale_with_stash, 
            preexisting_fp32_grads ,preexisting_fp32_grads };
        multi_tensor_axpby_cuda(fixed_chunk_size, is_overflow_buffer, tensor_lists, inv_scale, float(1.0), 0);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("multi_tensor_adam",
          &multi_tensor_adam_cuda,
          "Compute and apply gradient update to parameters for Adam optimizer");
    m.def("unscale_fp16_grads_into_fp32_grads",
          &unscale_fp16_grads_into_fp32_grads,
          "Unscale those fp16 gradients into fp32 gradient buffers.");
}
