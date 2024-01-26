/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/common/logger.h"
//#include "contrib_ops/cuda/bert/tensorrt_llm_fmha/common/tllmException.h"
#include <cuda_runtime.h>

namespace tensorrt_llm::common
{

Logger::Logger()
{
    char* isFirstRankOnlyChar = std::getenv("TLLM_LOG_FIRST_RANK_ONLY");
    bool isFirstRankOnly = (isFirstRankOnlyChar != nullptr && std::string(isFirstRankOnlyChar) == "ON");

    int deviceId;
    cudaGetDevice(&deviceId);

    char* levelName = std::getenv("TLLM_LOG_LEVEL");
    if (levelName != nullptr)
    {
        std::map<std::string, Level> nameToLevel = {
            {"TRACE", TRACE},
            {"DEBUG", DEBUG},
            {"INFO", INFO},
            {"WARNING", WARNING},
            {"ERROR", ERROR},
        };
        auto level = nameToLevel.find(levelName);
        // If TLLM_LOG_FIRST_RANK_ONLY=ON, set LOG LEVEL of other device to ERROR
        if (isFirstRankOnly && deviceId != 0)
        {
            level = nameToLevel.find("ERROR");
        }
        if (level != nameToLevel.end())
        {
            setLevel(level->second);
        }
        else
        {
            fprintf(stderr,
                "[TensorRT-LLM][WARNING] Invalid logger level TLLM_LOG_LEVEL=%s. "
                "Ignore the environment variable and use a default "
                "logging level.\n",
                levelName);
            levelName = nullptr;
        }
    }
}

void Logger::log(std::exception const& ex, Logger::Level level)
{
    //log(level, "%s: %s", TllmException::demangle(typeid(ex).name()).c_str(), ex.what());
    log(level, "%s: %s", typeid(ex).name(), ex.what());
}

Logger* Logger::getLogger()
{
    thread_local Logger instance;
    return &instance;
}

} // namespace tensorrt_llm::common
