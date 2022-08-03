/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "core/providers/shared_library/provider_api.h"
#include "hailo_global_vdevice.h"

namespace onnxruntime {

std::mutex GlobalVDevice::m_mutex;

GlobalVDevice& GlobalVDevice::get_instance()
{
    static GlobalVDevice instance;
    return instance;
}
    
std::shared_ptr<VDevice> GlobalVDevice::get_vdevice()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vdevice) {
        m_vdevice = create_vdevice();
    }

    return m_vdevice;
}

void GlobalVDevice::release()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_vdevice.use_count() <= 1) {
        m_vdevice.reset();
    }
}

std::shared_ptr<VDevice> GlobalVDevice::create_vdevice()
{
    hailo_vdevice_params_t params;
    auto status = hailo_init_vdevice_params(&params);
    HAILO_ORT_ENFORCE(HAILO_SUCCESS == status, "Failed init vdevice_params, status = ", status);
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;

    auto vdevice = VDevice::create(params);
    HAILO_CHECK_EXPECTED(vdevice, "Creating VDevice failed");
    return vdevice.release();
}

}  // namespace onnxruntime