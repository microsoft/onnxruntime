/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#pragma once
#include "core/framework/op_kernel.h"
#include "hailo_node_capability.h"
#include "hailo/hailort.hpp"

namespace onnxruntime {


#define HAILO_ORT_ENFORCE(condition, ...) ORT_ENFORCE(condition, "Hailo Error - ", __VA_ARGS__)
#define HAILO_ORT_THROW(...) ORT_THROW("Hailo Error - ", __VA_ARGS__)
#define HAILO_CHECK_EXPECTED(obj, ...)                                                                                                      \
    do {                                                                                                                                    \
        const auto &__check_expected_obj = (obj);                                                                                           \
        if(!__check_expected_obj.has_value())   {                                                                                           \
            ORT_THROW("HAILO_CHECK_EXPECTED failed with status=", __check_expected_obj.status(), ". ", __VA_ARGS__);                        \
        }                                                                                                                                   \
    } while(0)


class HailoUtils final
{
public:

    static TensorShape convert_hailo_shape(long int frames_count, hailo_3d_image_shape_t hailo_shape, hailo_format_order_t order)
    {
        switch (order) {
        case HAILO_FORMAT_ORDER_NC:
            return std::move(TensorShape({frames_count, hailo_shape.features}));
        case HAILO_FORMAT_ORDER_HAILO_NMS:
            HAILO_ORT_THROW("Hailo format order nms is not supported by HailoEP.");
        default:
            return std::move(TensorShape({frames_count, hailo_shape.features, hailo_shape.height, hailo_shape.width}));
        }
    }

    static hailo_format_type_t convert_ort_to_hailo_dtype(int ort_type)
    {
        switch (ort_type) {
        case ORT_DataType::type_float32:
            return HAILO_FORMAT_TYPE_FLOAT32;
            break;
        case ORT_DataType::type_uint16:
            return HAILO_FORMAT_TYPE_UINT16;
            break;
        case ORT_DataType::type_uint8:
            return HAILO_FORMAT_TYPE_UINT8;
            break;
        default:
            HAILO_ORT_THROW("ORT data type = ", ort_type, " is not supported by Hailort.");
            break;
        }
    };

private:

    template<typename T>
    static void transform_NCHW_to_NHWC(const T *src_ptr, T *dst_ptr, hailo_3d_image_shape_t *shape, uint32_t frames_count)
    {
        /* Validate arguments */
        HAILO_ORT_ENFORCE(NULL != src_ptr, "Invalid argument");
        HAILO_ORT_ENFORCE(NULL != dst_ptr, "Invalid argument");

        for (uint32_t n = 0; n < frames_count; n++) {
            for (uint32_t h = 0; h < shape->height ; h++) {
                for (uint32_t w = 0; w < shape->width; w++) {
                    for (uint32_t c = 0; c < shape->features; c++) {
                        dst_ptr[n * shape->height * shape->width * shape->features + h * shape->width * shape->features + w * shape->features + c] =
                            src_ptr[n * shape->height * shape->width * shape->features + shape->width * shape->height * c + shape->width * h + w];
                    }
                }
            }
        }
    }


    template<typename T>
    static void transform_NHWC_to_NCHW(const T *src_ptr, T *dst_ptr, hailo_3d_image_shape_t *shape, uint32_t frames_count)
    {
        /* Validate arguments */
        HAILO_ORT_ENFORCE(NULL != src_ptr, "Invalid argument");
        HAILO_ORT_ENFORCE(NULL != dst_ptr, "Invalid argument");

        for (uint32_t n = 0; n < frames_count; n++) {
            for (uint32_t h = 0; h < shape->height; h++) {
                for (uint32_t w = 0; w < shape->width; w++) {
                    for (uint32_t c = 0; c < shape->features; c++) {
                        dst_ptr[n * shape->height * shape->width * shape->features + shape->width * shape->height * c + shape->width * h + w] =
                            src_ptr[n * shape->height * shape->width * shape->features + h * shape->width * shape->features + w * shape->features + c];
                    }
                }
            }
        }
    }

public:
    static void transform_NCHW_to_NHWC(const void *src_ptr, void *dst_ptr, hailo_3d_image_shape_t *shape, hailo_format_type_t dtype, uint32_t frames_count)
    {
        switch (dtype)
        {
        case HAILO_FORMAT_TYPE_FLOAT32:
            transform_NCHW_to_NHWC<float32_t>((float32_t*)src_ptr, (float32_t*)dst_ptr, shape, frames_count);
            break;
        case HAILO_FORMAT_TYPE_UINT16:
            transform_NCHW_to_NHWC<uint16_t>((uint16_t*)src_ptr, (uint16_t*)dst_ptr, shape, frames_count);
            break;
        case HAILO_FORMAT_TYPE_UINT8:
            transform_NCHW_to_NHWC<uint8_t>((uint8_t*)src_ptr, (uint8_t*)dst_ptr, shape, frames_count);
            break;
        default:
            HAILO_ORT_THROW("Unsupported data type");
            break;
        }
    }

    static void transform_NHWC_to_NCHW(void *src_ptr, void *dst_ptr, hailo_3d_image_shape_t *shape, hailo_format_type_t dtype, uint32_t frames_count)
    {
        switch (dtype)
        {
        case HAILO_FORMAT_TYPE_FLOAT32:
            transform_NHWC_to_NCHW<float32_t>((float32_t*)src_ptr, (float32_t*)dst_ptr, shape, frames_count);
            break;
        case HAILO_FORMAT_TYPE_UINT16:
            transform_NHWC_to_NCHW<uint16_t>((uint16_t*)src_ptr, (uint16_t*)dst_ptr, shape, frames_count);
            break;
        case HAILO_FORMAT_TYPE_UINT8:
            transform_NHWC_to_NCHW<uint8_t>((uint8_t*)src_ptr, (uint8_t*)dst_ptr, shape, frames_count);
            break;
        default:
            HAILO_ORT_THROW("Unsupported data type");
            break;
        }
    }
};

}  // namespace onnxruntime