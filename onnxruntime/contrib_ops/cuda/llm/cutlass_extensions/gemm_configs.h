/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "cute/tensor.hpp"

namespace onnxruntime::llm {
namespace cutlass_extensions {

// Note: The shapes are in the format MxNxK. The K shape of the runtime config MUST match the K shape
//       in the kernel layout details when doing weight only quantization.
enum class CutlassTileConfig {
  // Signals that we should run heuristics do choose a config
  Undefined,

  // Signals that we should run heuristics do choose a config
  ChooseWithHeuristic,

  // SiMT config
  CtaShape128x128x8_WarpShape64x64x8,

  // TensorCore configs CTA_N = 128, CTA_K = 64
  // Warp configs for M=16
  CtaShape16x128x64_WarpShape16x32x64,

  // Warp configs for M=32
  CtaShape32x128x64_WarpShape32x32x64,

  // Warp configs for M=64
  CtaShape64x128x64_WarpShape32x64x64,
  CtaShape64x64x128_WarpShape32x64x64,
  CtaShape64x128x64_WarpShape64x32x64,

  // Warp configs for M=128
  CtaShape128x64x64_WarpShape64x32x64,
  CtaShape128x128x64_WarpShape64x32x64,
  CtaShape128x128x64_WarpShape64x64x64,
  CtaShape128x128x64_WarpShape128x32x64,
  CtaShape128x256x64_WarpShape64x64x64,

  // Warp configs for M=256
  CtaShape256x128x64_WarpShape64x64x64,

  // TensorCore config CTA_N = 64, CTA_K = 128
  CtaShape128x64x128_WarpShape64x32x128,

  // TensorCore config CTA_N = 256, CTA_K = 64
  CtaShape16x256x64_WarpShape16x64x64,

  // TensorCore config CTA_N = 256, CTA_K = 128
  CtaShape16x256x128_WarpShape16x64x128

};

enum class SplitKStyle {
  NO_SPLIT_K,
  SPLIT_K_SERIAL,
  STREAM_K,  // Sm80+
             // SPLIT_K_PARALLEL // Not supported yet
};

enum class CutlassTileConfigSM90 {
  // Signals that we should run heuristics do choose a config
  Undefined,

  // Signals that we should run heuristics do choose a config
  ChooseWithHeuristic,

  // CTA configs for M=64
  CtaShape64x16x128B,
  CtaShape64x32x128B,
  CtaShape64x64x128B,
  CtaShape64x128x128B,
  CtaShape64x256x128B,

  // CTA configs for M=128
  CtaShape128x16x128B,
  CtaShape128x32x128B,
  CtaShape128x64x128B,
  CtaShape128x128x128B,
  CtaShape128x256x128B,

  // CTA configs for M=256
  CtaShape256x128x128B,
  CtaShape256x256x128B,
};

enum class CutlassTileConfigSM100 {
  // Signals that we should run heuristics do choose a config
  Undefined,

  // Signals that we should run heuristics do choose a config
  ChooseWithHeuristic,

  /*
   * Grouped GEMM
   */
  // M=64
  CtaShape64x32x128B,
  CtaShape64x64x128B,
  CtaShape64x128x128B,
  CtaShape64x256x128B,

  // M=128
  CtaShape128x8x256B,
  CtaShape128x16x128B,
  CtaShape128x32x128B,
  CtaShape128x64x128B,
  CtaShape128x128x128B,
  CtaShape128x256x128B,
  CtaShape128x128x256B,
  CtaShape128x256x256B,

  // M=256
  CtaShape256x64x128B,
  CtaShape256x128x128B,
  CtaShape256x256x128B,
};

enum class CutlassTileConfigSM120 {
  // Signals that we should run heuristics do choose a config
  Undefined,

  // Signals that we should run heuristics do choose a config
  ChooseWithHeuristic,

  CtaShape128x128x128B,
  CtaShape128x128x64B,
  CtaShape256x128x64B,
  CtaShape128x256x64B,
  CtaShape128x128x256B,
  CtaShape256x128x128B,
};

enum class MainloopScheduleType {
  AUTO,  // Automatically selects between pingpong and cooperative schedules on Hopper. On older architectures, this
         // defaults to the "legacy" main loop schedule.
  PINGPONG,
  COOPERATIVE,
  WARPSPECIALIZED
};

#if 0
static auto get_mainloop_schedule_name(MainloopScheduleType schedule) {
  if (schedule == MainloopScheduleType::AUTO) {
    return "auto";
  } else if (schedule == MainloopScheduleType::PINGPONG) {
    return "pingpong";
  } else if (schedule == MainloopScheduleType::COOPERATIVE) {
    return "cooperative";
  } else if (schedule == MainloopScheduleType::WARPSPECIALIZED) {
    return "warpspecialized";
  }
  return "unknown schedule";
}
#endif

enum class EpilogueScheduleType {
  AUTO,  // Automatically chooses an epilogue schedule compatible with the selected main loop schedule for Hopper. For
         // architectures older than hopper, the epilogue is always performed by the same thread block as the main
         // loop.
};

enum class TileShape {
  TileShape_64x16x128,
  TileShape_64x32x128,
  TileShape_64x64x128,
  TileShape_64x128x128,
  TileShape_64x256x128,
  TileShape_64x512x128,
  TileShape_128x16x128,
  TileShape_128x32x128,
  TileShape_128x64x128,
  TileShape_128x128x128,
  TileShape_128x256x128,
  TileShape_256x128x128,
  TileShape_256x256x128
};

template <TileShape Shape_MNK>
constexpr auto get_tile_shape() {
  using namespace cute;
  if constexpr (Shape_MNK == TileShape::TileShape_64x16x128) {
    return cute::Shape<_64, _16, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_64x32x128) {
    return cute::Shape<_64, _32, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_64x64x128) {
    return cute::Shape<_64, _64, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_64x128x128) {
    return cute::Shape<_64, _128, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_64x256x128) {
    return cute::Shape<_64, _256, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_64x512x128) {
    return cute::Shape<_64, _512, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_128x16x128) {
    return cute::Shape<_128, _16, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_128x32x128) {
    return cute::Shape<_128, _32, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_128x64x128) {
    return cute::Shape<_128, _64, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_128x128x128) {
    return cute::Shape<_128, _128, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_128x256x128) {
    return cute::Shape<_128, _256, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_256x128x128) {
    return cute::Shape<_256, _128, _128>{};
  } else if constexpr (Shape_MNK == TileShape::TileShape_256x256x128) {
    return cute::Shape<_256, _256, _128>{};
  }
}

#if 0
static auto get_tile_shape_name(TileShape Shape_MNK) {
  if (Shape_MNK == TileShape::TileShape_64x16x128) {
    return "64x16x128";
  } else if (Shape_MNK == TileShape::TileShape_64x32x128) {
    return "64x32x128";
  } else if (Shape_MNK == TileShape::TileShape_64x64x128) {
    return "64x64x128";
  } else if (Shape_MNK == TileShape::TileShape_64x128x128) {
    return "64x128x128";
  } else if (Shape_MNK == TileShape::TileShape_64x256x128) {
    return "64x256x128";
  } else if (Shape_MNK == TileShape::TileShape_64x512x128) {
    return "64x512x128";
  } else if (Shape_MNK == TileShape::TileShape_128x16x128) {
    return "128x16x128";
  } else if (Shape_MNK == TileShape::TileShape_128x32x128) {
    return "128x32x128";
  } else if (Shape_MNK == TileShape::TileShape_128x64x128) {
    return "128x64x128";
  } else if (Shape_MNK == TileShape::TileShape_128x128x128) {
    return "128x128x128";
  } else if (Shape_MNK == TileShape::TileShape_128x256x128) {
    return "128x256x128";
  }  else if (Shape_MNK == TileShape::TileShape_256x128x128)
    {
        return "256x128x128";
    }
    else if (Shape_MNK == TileShape::TileShape_256x256x128)
    {
        return "256x256x128";
    }
  return "Unknown shape";
}
#endif

enum class ClusterShape {
  ClusterShape_1x1x1,
  ClusterShape_2x1x1,
  ClusterShape_1x2x1,
  ClusterShape_2x2x1,
  ClusterShape_1x4x1,
  ClusterShape_4x1x1,
  ClusterShape_4x2x1,
  ClusterShape_2x4x1,
  ClusterShape_4x4x1,
  ClusterShape_1x8x1,
  ClusterShape_8x1x1
};

#if 0
static auto get_cluster_shape_name(ClusterShape Shape_MNK) {
  if (Shape_MNK == ClusterShape::ClusterShape_1x1x1) {
    return "1x1x1";
  } else if (Shape_MNK == ClusterShape::ClusterShape_2x1x1) {
    return "2x1x1";
  } else if (Shape_MNK == ClusterShape::ClusterShape_1x2x1) {
    return "1x2x1";
  } else if (Shape_MNK == ClusterShape::ClusterShape_2x2x1) {
    return "2x2x1";
  } else if (Shape_MNK == ClusterShape::ClusterShape_4x1x1)
    {
        return "4x1x1";
    } else if (Shape_MNK == ClusterShape::ClusterShape_1x8x1) {
    return "1x8x1";
  } else if (Shape_MNK == ClusterShape::ClusterShape_8x1x1) {
    return "8x1x1";
  }
  return "Unknown shape";
}

template <ClusterShape Shape_MNK>
constexpr auto get_cluster_shape() {
  using namespace cute;
  if constexpr (Shape_MNK == ClusterShape::ClusterShape_1x1x1) {
    return cute::Shape<_1, _1, _1>{};
  } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_2x1x1) {
    return cute::Shape<_2, _1, _1>{};
  } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_1x2x1) {
    return cute::Shape<_1, _2, _1>{};
  } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_2x2x1) {
    return cute::Shape<_2, _2, _1>{};
  } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_4x1x1)
    {
        return cute::Shape<_4, _1, _1>{};
    } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_1x8x1) {
    return cute::Shape<_1, _8, _1>{};
  } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_8x1x1) {
    return cute::Shape<_8, _1, _1>{};
  }
}
#endif

struct CutlassGemmConfig {
  enum CandidateConfigTypeParam : int {
    NONE = 0,
    WEIGHT_ONLY = 1u << 0,
    SIMT_ONLY = 1u << 1,
    INT8_ONLY = 1u << 2,
    HOPPER = 1u << 3,
    BLACKWELL = 1u << 4,
    GROUPED_GEMM = 1u << 5,
    FP8_ONLY = 1u << 6,
    FP4_ONLY = 1u << 7
  };

  CutlassTileConfig tile_config_sm80 = CutlassTileConfig::ChooseWithHeuristic;
  SplitKStyle split_k_style = SplitKStyle::NO_SPLIT_K;
  int split_k_factor = -1;
  int stages = -1;

  // config options for sm90
  CutlassTileConfigSM90 tile_config_sm90 = CutlassTileConfigSM90::ChooseWithHeuristic;
  CutlassTileConfigSM100 tile_config_sm100 = CutlassTileConfigSM100::ChooseWithHeuristic;
  CutlassTileConfigSM120 tile_config_sm120 = CutlassTileConfigSM120::ChooseWithHeuristic;
  MainloopScheduleType mainloop_schedule = MainloopScheduleType::AUTO;
  EpilogueScheduleType epilogue_schedule = EpilogueScheduleType::AUTO;
  ClusterShape cluster_shape = ClusterShape::ClusterShape_1x1x1;
  bool enableCudaKernel = false;
  int sm_version = 80;  // Use 80 as a catch all for <90
  bool is_tma_warp_specialized = false;

  CutlassGemmConfig() = default;

  CutlassGemmConfig(CutlassTileConfig tile_config, SplitKStyle split_k_style, int split_k_factor, int stages)
      : tile_config_sm80(tile_config), split_k_style(split_k_style), split_k_factor(split_k_factor), stages(stages), sm_version(80) {
  }

  CutlassGemmConfig(CutlassTileConfigSM90 tile_config_sm90, MainloopScheduleType mainloop_schedule,
                    EpilogueScheduleType epilogue_schedule, ClusterShape cluster_shape)
      : tile_config_sm90(tile_config_sm90), mainloop_schedule(mainloop_schedule), epilogue_schedule(epilogue_schedule), cluster_shape(cluster_shape), sm_version(90), is_tma_warp_specialized(true) {
  }

  CutlassGemmConfig(CutlassTileConfigSM100 tile_config_sm100, MainloopScheduleType mainloop_schedule,
                    EpilogueScheduleType epilogue_schedule, ClusterShape cluster_shape)
      : tile_config_sm100(tile_config_sm100), mainloop_schedule(mainloop_schedule), epilogue_schedule(epilogue_schedule), cluster_shape(cluster_shape), sm_version(100), is_tma_warp_specialized(true) {
  }

  CutlassGemmConfig(CutlassTileConfigSM120 tile_config_sm120, MainloopScheduleType mainloop_schedule,
                    EpilogueScheduleType epilogue_schedule, ClusterShape cluster_shape)
      : tile_config_sm120(tile_config_sm120), mainloop_schedule(mainloop_schedule), epilogue_schedule(epilogue_schedule), cluster_shape(cluster_shape), sm_version(120), is_tma_warp_specialized(true) {
  }

  int getTileConfigAsInt() const {
    if (sm_version == 120)
      return (int)tile_config_sm80;
    if (sm_version >= 100)
      return (int)tile_config_sm100;
    if (sm_version == 90)
      return (int)tile_config_sm90;
    if (sm_version < 90)
      return (int)tile_config_sm80;
    assert(false && "Invalid SM version");
    return -1;
  }

  std::string toString() const {
    std::stringstream tactic;
    tactic << "Cutlass GEMM Tactic";
    if (is_tma_warp_specialized && getTileConfigAsInt() != (int)CutlassTileConfigSM90::ChooseWithHeuristic) {
      assert(sm_version >= 90 && "Invalid cutlass GEMM config");
      tactic << "\n\tstyle=TMA Warp Specialized"
             << "\n\tsm: " << sm_version << "\n\ttile shape ID: " << getTileConfigAsInt()
             << "\n\tcluster shape ID: " << (int)cluster_shape
             << "\n\tmainloop sched: " << (int)mainloop_schedule << "\n\tepi sched: " << (int)epilogue_schedule
             << "\n\tenable cuda kernel: " << (enableCudaKernel ? "true" : "false");
    } else if (tile_config_sm80 != onnxruntime::llm::cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic) {
      assert(sm_version < 90 && "Invalid cutlass GEMM config");
      tactic << "\n\tstyle=compatible"
             << "\n\ttile shape ID: " << (int)tile_config_sm80 << "\n\tstages: " << (int)stages
             << "\n\tsplit k: " << (int)split_k_factor
             << "\n\tenable cuda kernel: " << (enableCudaKernel ? "true" : "false");
    } else if (enableCudaKernel) {
      tactic << "\n\tenable cuda kernel: " << (enableCudaKernel ? "true" : "false");
    } else {
      tactic << "\n\tundefined";
    }
    tactic << "\n";
    return tactic.str();
  }
};

inline std::ostream& operator<<(std::ostream& out, CutlassGemmConfig const& config) {
  if (config.is_tma_warp_specialized) {
    out << "tile_config_sm90_enum: " << config.getTileConfigAsInt()
        << ", mainloop_schedule_enum: " << int(config.mainloop_schedule)
        << ", epilogue_schedule_enum: " << int(config.epilogue_schedule)
        << ", cluster_shape_enum: " << int(config.cluster_shape)
        << ", enable_cuda_kernel: " << (config.enableCudaKernel ? "true" : "false");
  } else {
    out << "tile_config_enum: " << config.getTileConfigAsInt()
        << ", split_k_style_enum: " << int(config.split_k_style)
        << ", split_k_factor: " << config.split_k_factor
        << ", stages: " << config.stages
        << ", enable_cuda_kernel: " << (config.enableCudaKernel ? "true" : "false");
  }
  return out;
}

}  // namespace cutlass_extensions
}  // namespace onnxruntime::llm

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
