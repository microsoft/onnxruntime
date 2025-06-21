// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_tunning_utils.h"
#include <stdint.h>
#include <optional>
#include "opencl_program_manager.h"

namespace onnxruntime {
namespace opencl {
using param_type = size_t;
using RetType = std::optional<double>;
typedef std::function<RetType(const std::vector<param_type>&,
                                    std::vector<param_type>&)> GPURunOneIterFunc;

class EPPerfTuner {
 private:
  std::string tuned_param_file_path_;

 public:
  explicit EPPerfTuner(const std::string tuned_param_file_path = "") : tuned_param_file_path_(tuned_param_file_path) {

  }
  ~EPPerfTuner() {}

  EPPerfTuner(const EPPerfTuner&) = delete;
  EPPerfTuner& operator=(const EPPerfTuner&) = delete;

  //run multiple times for a more accurate time_cost statistic.
  std::optional<double> Run(const GPURunOneIterFunc &func,
              const std::vector<param_type> &params,
              int num_runs,
              std::vector<param_type> tuning_result) {
    RetType res_or_tc_us;
    int iter = 0;
    double total_time_us = 0;
    for (iter = 0; iter < num_runs; ++iter) {
      res_or_tc_us = func(params, tuning_result);
      if (!res_or_tc_us.has_value()) {
        return std::make_optional<double>(std::numeric_limits<double>::max());
      }
      total_time_us += res_or_tc_us.value();
    }
    return std::make_optional<double>(total_time_us / iter);
  }

  int32_t Tune(const std::function<std::vector<std::vector<param_type>>()>& param_generator,
               const std::vector<param_type>& global_param,
               const std::vector<param_type>& default_param,
               const GPURunOneIterFunc &func,
               std::vector<param_type>& opt_params) {
    double tmp_time = 0.0;
    double default_param_of_tc = Run(func, global_param, 1, default_param).value();
    double opt_time = std::numeric_limits<double>::max();
    auto params = param_generator();
    std::vector<param_type> tuning_result;
    for (auto param : params) {      
      // warm up
      if (Run(func, param, 1, tuning_result).has_value()==false){
        return -1;    
      }
      // run
      tmp_time = Run(func, global_param, 10, param).value();

      // Check the execution time
      if (tmp_time < opt_time) {
        opt_time = tmp_time;
        opt_params = tuning_result;
      }

      //early_return
      if (opt_time < 0.3 * default_param_of_tc) {
        break;
      }
    }
    return 0;
  }
};

// adreno local size calculate //reference to TNN
std::vector<size_t> AdrenoLocalSize2D(const opencl::NDRange& gws, const opencl::OpenCLDeviceInfo& gpu_info) {
  std::vector<size_t> lws;
  const size_t max_workgroup_size = gpu_info.max_work_group_size;
  const size_t subgroup_size = gpu_info.sub_group_size;
  size_t min_workgroup_count = gpu_info.compute_units_;
  // for the later verion gpu 1 SP can process more than one workgroup
  if (gpu_info.gpu_model >= 540)
    min_workgroup_count = 2 * gpu_info.compute_units_;

  // check gws[1] fisrt
  if (gws[1] % min_workgroup_count == 0) {
    lws.resize(2);
    lws[1] = std::min<size_t>(gws[1] / min_workgroup_count, max_workgroup_size);
    auto AdrenoLocalSizeValid = [](const opencl::NDRange& gws, std::vector<size_t>& lws,
                                   const size_t subgroup_size)->bool {
      return 0 == (lws[0] * lws[1]) % subgroup_size && 0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] &&
             ((lws[0] < lws[1]) == (gws[0] < gws[1]));
    };
    // if subgroup size is got, then use it
    if (0 != subgroup_size) {
      size_t min_workgroup_size = subgroup_size * 2;
      size_t max_val = std::max<size_t>(max_workgroup_size / lws[1], 1);
      size_t min_val = std::max<size_t>(min_workgroup_size / lws[1], 1);
      lws[0] = std::min<size_t>(gws[0], max_val);
      for (; lws[0] >= min_val; lws[0]--) {
        if (AdrenoLocalSizeValid(gws, lws, subgroup_size)) {
          return lws;
        }
      }
    }

    // another way to calculate lws[0]
    lws[0] = max_workgroup_size / lws[1];
    lws[0] = std::max<size_t>(std::min<size_t>(gws[0], lws[0]), 1);
    if (0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] && ((lws[0] < lws[1]) == (gws[0] < gws[1]))) {
      return lws;
    }
  }

  // check gws[0] later
  if (gws[0] % min_workgroup_count == 0) {
    lws.resize(2);
    lws[0] = std::min<size_t>(gws[0] / min_workgroup_count, max_workgroup_size);

    // if subgroup size is got, then use it
    if (0 != subgroup_size) {
      size_t min_workgroup_size = subgroup_size * 2;
      size_t max_val = std::max<size_t>(max_workgroup_size / lws[0], 1);
      size_t min_val = std::max<size_t>(min_workgroup_size / lws[0], 1);
      lws[1] = std::min<size_t>(gws[1], max_val);
      for (; lws[1] >= min_val; lws[1]--) {
        if (0 == (lws[0] * lws[1]) % subgroup_size && 0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] &&
            ((lws[0] < lws[1]) == (gws[0] < gws[1]))) {
          return lws;
        }
      }
    }

    // another way to calculate lws[1]
    lws[1] = max_workgroup_size / lws[0];
    lws[1] = std::max<size_t>(std::min<size_t>(gws[1], lws[1]), 1);
    if (0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] && ((lws[0] < lws[1]) == (gws[0] < gws[1]))) {
      return lws;
    }
  }
  lws.clear();
  return lws;
}

opencl::NDRange RunTuneLWS2D(const opencl::NDRange& gws, opencl::OpenCLDeviceInfo dev_info_, const opencl::TuneKernelWithTimeFunc& func, int32_t auto_tuning_level) {
  auto max_work_group_size = dev_info_.max_work_group_size;
  auto max_work_item_size = dev_info_.max_work_item_size;
  std::vector<size_t> lws_prefer(2, 1);
  std::vector<size_t> lws(2, 1);
  double default_lws_tc = func(opencl::NDRange(), gws);//get default tc
  double min_cost = default_lws_tc;
  auto valid_lws2d_checker = [&]()->bool {
    if (lws[0] > gws[0]) return false;
    if (lws[0] <= max_work_item_size[0] && lws[1] <= max_work_item_size[1] && lws[0] * lws[1] <= max_work_group_size) {
      return true;
    }
    return false;
  };
  bool satisfy_fast_return = false;
  auto update_lws_step = [&]() {
    // in some mobile GPUs, global size must be divisible by local-size
    std::vector<size_t> internalGlobalWS(2, 1);
    for (size_t i = 0; i < gws.Size(); ++i) {
      internalGlobalWS[i] = ROUND_UP(gws[i], std::max<size_t>(1, lws[i]));
    }
    opencl::NDRange g(internalGlobalWS);
    opencl::NDRange l(lws);
    double cost_time = func(l, g);
    if (cost_time < min_cost) {
      min_cost = cost_time;
      lws_prefer[0] = lws[0];
      lws_prefer[1] = lws[1];
    };
    if (min_cost <= 0.4 * default_lws_tc) {
      satisfy_fast_return = true;
    }
  };
  if (auto_tuning_level == 1){
    if (dev_info_.gpu_type == opencl::GpuType::ADRENO) {
      for (lws[0] = 1; !satisfy_fast_return && lws[0] <= 16; lws[0] *= 2) {  // dim0
        for (lws[1] = 32; !satisfy_fast_return && lws[1] <= 256; lws[1] *= 4) {  // dim1
          if (!valid_lws2d_checker() || lws[0] * lws[1]<128) {
            continue;
          }
          update_lws_step();
        }
      }
    } else {
      for (lws[0] = 2; !satisfy_fast_return && lws[0] <= 16; lws[0] *= 2) {  // dim0
        for (lws[1] = 2; !satisfy_fast_return && lws[1] <= 64; lws[1] *= 2) {  // dim1
          if (!valid_lws2d_checker() || lws[0] * lws[1] < 8) {
            continue;
          }
          update_lws_step();
        }
      }
    }
  } else if (auto_tuning_level == 2) {
    while(lws[1] <= gws[1] || lws[1] <= 6) {
        lws[0] = 1;
        while(lws[0] <= gws[0] || lws[0] <= 6) {
          if (lws[0] <= max_work_item_size[0] && lws[1] <= max_work_item_size[1] && lws[0] * lws[1] <= max_work_group_size) {
            update_lws_step();
          }
          lws[0] = (lws[0] + 2 + 1) / 2 * 2;
        }
        lws[1] = (lws[1] + 2 + 1) / 2 * 2;
    }
  } else if (auto_tuning_level == 3) {
    while(lws[1] <= gws[1] || lws[1] <= 6) {
            lws[0] = 1;
            while(lws[0] <= gws[0] || lws[0] <= 6) {
                if(lws[0] <= max_work_item_size[0] && lws[1] <= max_work_item_size[1] && lws[0]*lws[1] <= max_work_group_size) {
                  update_lws_step();
                }
                do {
                    lws[0]++;
                }
                while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] > 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[1]++;
            }
            while(((2*gws[1])%lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] > 6));//divisible powOfTwo lessThanSix
    }
  } else if (auto_tuning_level == 4) {
    while (lws[1] <= gws[1] && lws[1] <= 6) {
      lws[0] = 1;
      while (lws[0] <= gws[0] || lws[0] <= 6) {
        if (lws[0] <= max_work_item_size[0] && lws[1] <= max_work_item_size[1] && lws[0] * lws[1] <= max_work_group_size) {
          update_lws_step();
        }
        do {
          lws[0]++;
        } while (((2 * gws[0]) % lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] > 6));  // divisible powOfTwo lessThanSix
      }
      do {
        lws[1]++;
      } while (((2 * gws[1]) % lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] <= 6));  // divisible powOfTwo lessThanSix
    }
  } else if (auto_tuning_level == 5) {
    while (lws[1] <= gws[1] && lws[1] <= 6) {
      lws[0] = 1;
      while (lws[0] <= gws[0] && lws[0] <= 6) {
        if (lws[0] <= max_work_item_size[0] && lws[1] <= max_work_item_size[1] && lws[0] * lws[1] <= max_work_group_size) {
          update_lws_step();
        }
        do {
          lws[0]++;
        } while (((2 * gws[0]) % lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] <= 6));  // divisible powOfTwo lessThanSix
      }
      do {
        lws[1]++;
      } while (((2 * gws[1]) % lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] <= 6));  // divisible powOfTwo lessThanSix
    }
  }
  else if (auto_tuning_level == 6) {
    std::vector<std::vector<size_t>> candidates = {
        {max_work_group_size / 2, 2, 0}, {max_work_group_size / 4, 4, 0}, {max_work_group_size / 8, 8, 0}, {max_work_group_size / 16, 16, 0}, {max_work_group_size / 32, 32, 0}, {max_work_group_size / 64, 64, 0}, {max_work_group_size / 128, 128, 0}, {max_work_group_size / 256, 256, 0}, {max_work_group_size, 1, 0}, {1, max_work_group_size, 0}};
    for (auto param : candidates) {
      lws[0] = param[0];
      lws[1] = param[1];
      update_lws_step();
    }
  }
  if (lws_prefer == std::vector<size_t>{1, 1}) {
    return opencl::NDRange();
  }
  VLOGF_DEFAULT(2, "after tuning, kernel latency was improved from %f to %f", default_lws_tc, min_cost);
  printf("after tuning, kernel latency was improved from %f to %f\n", default_lws_tc, min_cost);
  return opencl::NDRange(lws_prefer);
}
}
}  // namespace onnxruntime
