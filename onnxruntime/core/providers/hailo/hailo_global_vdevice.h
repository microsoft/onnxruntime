/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#pragma once
#include "hailo/hailort.hpp"
#include "utils.h"

namespace onnxruntime {

using hailort::VDevice;

class GlobalVDevice {
public:
    static GlobalVDevice& get_instance();
    std::shared_ptr<VDevice> get_vdevice();
    void release();

    static std::mutex m_mutex;

private:
    GlobalVDevice() : m_vdevice() {}
    GlobalVDevice(GlobalVDevice const&) = delete;
    void operator=(GlobalVDevice const&) = delete;

    std::shared_ptr<VDevice> create_vdevice();

    std::shared_ptr<VDevice> m_vdevice;
};

}  // namespace onnxruntime