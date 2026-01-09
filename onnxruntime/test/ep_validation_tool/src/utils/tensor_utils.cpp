#include "tensor_utils.h"

#include "npy/npy.h"
#include "npy/npz.h"
#include "npy/tensor.h"
#include "npy/core.h"
#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <math.h>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

static constexpr size_t s_batch_size = 10000;
static size_t GetOrtValueElementCount(const Ort::Value& value)
{
    auto type_and_shape = value.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> value_shape = type_and_shape.GetShape();

    size_t element_count = 1;
    for (auto dim : value_shape)
    {
        element_count *= dim;
    }

    return element_count;
}

template<typename T>
static void CastOrtValueDataToFloat(const Ort::Value& tensor, std::vector<float>& buffer)
{
    size_t element_count = GetOrtValueElementCount(tensor);
    buffer.resize(element_count);

    const T* data = reinterpret_cast<const T*>(tensor.GetTensorRawData());
    if constexpr (std::is_same<T, float>())
    {
        std::copy(data, data + element_count, buffer.begin());
    }
    else
    {
        std::transform(data, data + element_count, buffer.begin(), [](const T& x) { return static_cast<float>(x); });
    }
}

void CastOrtValueData(const Ort::Value& value, std::vector<float>& buffer)
{
    auto tensor_info = value.GetTensorTypeAndShapeInfo();
    auto tensor_type = tensor_info.GetElementType();

    switch (tensor_type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        CastOrtValueDataToFloat<float>(value, buffer);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        CastOrtValueDataToFloat<std::int32_t>(value, buffer);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        CastOrtValueDataToFloat<std::int64_t>(value, buffer);
        break;
    }
}

template<typename T>
static Ort::Value ReadAndMakeORTValue(const std::string& path)
{
    auto npy_tensor = npy::load<T, npy::tensor>(path);

    // Create a tensor using an allocator
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Value tensor = Ort::Value::CreateTensor<T>(
        allocator, reinterpret_cast<const std::int64_t*>(npy_tensor.shape().data()), npy_tensor.shape().size());

    // Copy data into the tensor's buffer
    T* tensor_data = tensor.GetTensorMutableData<T>();
    std::copy(npy_tensor.values().begin(), npy_tensor.values().end(), tensor_data);

    return tensor;
}

Ort::Value ReadNumpy(const std::filesystem::path& path)
{
    const auto& file_path = path.string();
    npy::header_info header = npy::peek(file_path);

    switch (header.dtype)
    {
    case npy::data_type_t::FLOAT32:
        return ReadAndMakeORTValue<float>(file_path);
        break;
    case npy::data_type_t::INT32:
        return ReadAndMakeORTValue<std::int32_t>(file_path);
        break;
    case npy::data_type_t::INT64:
        return ReadAndMakeORTValue<std::int64_t>(file_path);
        break;
    }

    return Ort::Value(nullptr);
}

template<typename T>
static bool SaveOrtValue(const std::filesystem::path& path, const Ort::Value& value)
{
    try
    {
        auto type_and_shape = value.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape_int64 = type_and_shape.GetShape();
        std::vector<size_t> shape(shape_int64.begin(), shape_int64.end());

        npy::tensor<T> tensor(shape);
        size_t element_count = GetOrtValueElementCount(value);
        if (element_count == 0)
        {
            std::cerr << "ERROR: No elements in the processed tensor." << std::endl;
            return false;
        }

        const void* raw_data = value.GetTensorRawData();
        if (!raw_data)
        {
            std::cerr << "ERROR: No data in the processed tensor." << std::endl;
            return false;
        }
        tensor.copy_from(reinterpret_cast<const T*>(raw_data), element_count);

        npy::save(path.string(), tensor);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: Error saving ORT value: " << e.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "ERROR: Unknown error occurred saving ORT value." << std::endl;
        return false;
    }
}

bool SaveOrtValueAsNumpyArray(const std::filesystem::path& path, const Ort::Value& value)
{
    auto tensor_info = value.GetTensorTypeAndShapeInfo();
    auto tensor_type = tensor_info.GetElementType();

    switch (tensor_type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return SaveOrtValue<float>(path, value);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return SaveOrtValue<double>(path, value);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return SaveOrtValue<int8_t>(path, value);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return SaveOrtValue<uint8_t>(path, value);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return SaveOrtValue<int16_t>(path, value);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return SaveOrtValue<uint16_t>(path, value);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return SaveOrtValue<int32_t>(path, value);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return SaveOrtValue<uint32_t>(path, value);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return SaveOrtValue<int64_t>(path, value);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return SaveOrtValue<uint64_t>(path, value);

    default:
        std::cerr << "ERROR: Unsupported tensor data type." << std::endl;
        return false;
    }
}

// Template helper functions to eliminate switch statement duplication
template<typename T>
static inline void CopyTensorDataTyped(const Ort::Value& source, Ort::Value& dest, size_t element_count)
{
    std::copy(source.GetTensorData<T>(), source.GetTensorData<T>() + element_count, dest.GetTensorMutableData<T>());
}

static inline void CopyTensorData(const Ort::Value& source, Ort::Value& dest)
{
    const auto& tensor_info = source.GetTensorTypeAndShapeInfo();
    size_t element_count = tensor_info.GetElementCount();

    switch (tensor_info.GetElementType())
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        CopyTensorDataTyped<float>(source, dest, element_count);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        CopyTensorDataTyped<double>(source, dest, element_count);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        CopyTensorDataTyped<int8_t>(source, dest, element_count);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        CopyTensorDataTyped<uint8_t>(source, dest, element_count);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        CopyTensorDataTyped<int16_t>(source, dest, element_count);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        CopyTensorDataTyped<uint16_t>(source, dest, element_count);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        CopyTensorDataTyped<int32_t>(source, dest, element_count);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        CopyTensorDataTyped<uint32_t>(source, dest, element_count);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        CopyTensorDataTyped<int64_t>(source, dest, element_count);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        CopyTensorDataTyped<uint64_t>(source, dest, element_count);
        break;
    default:
        throw std::runtime_error("Unsupported tensor data type.");
    }
}

void CopyTensorMap(
    const std::unordered_map<std::string, Ort::Value>& source, std::unordered_map<std::string, Ort::Value>& dest)
{
    for (const auto& [name, tensor] : source)
    {
        const auto& tensor_info = tensor.GetTensorTypeAndShapeInfo();

        auto [it, inserted] = dest.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(name),
            std::forward_as_tuple(
                Ort::Value::CreateTensor(
                    Ort::AllocatorWithDefaultOptions(),
                    tensor_info.GetShape().data(),
                    tensor_info.GetShape().size(),
                    tensor_info.GetElementType())));

        CopyTensorData(tensor, it->second);
    }
}

Ort::Value CopyTensor(const Ort::Value& source)
{
    const auto& tensor_info = source.GetTensorTypeAndShapeInfo();
    Ort::Value dest = Ort::Value::CreateTensor(
        Ort::AllocatorWithDefaultOptions(),
        tensor_info.GetShape().data(),
        tensor_info.GetShape().size(),
        tensor_info.GetElementType());

    CopyTensorData(source, dest);
    return dest;
}

template<typename T, typename Predicate>
bool CheckAllInBatches(const Ort::Value& tensor, Predicate predicate, size_t batch_size = 10000)
{
    try
    {
        const T* data = tensor.GetTensorData<T>();
        const size_t num_elements = tensor.GetTensorTypeAndShapeInfo().GetElementCount();

        if (!data || num_elements == 0)
        {
            return false;
        }

        for (size_t i = 0; i < num_elements; i += batch_size)
        {
            const T* batch_start = data + i;
            const T* batch_end = data + std::min(i + batch_size, num_elements);
            if (!std::all_of(batch_start, batch_end, predicate))
            {
                return false;
            }
        }

        return true;
    }
    catch (const std::exception& e)
    {
        return false;
    }
}

template<typename T, typename Predicate>
bool CheckAnyInBatches(const Ort::Value& tensor, Predicate predicate, size_t batch_size = 10000)
{
    try
    {
        const T* data = tensor.GetTensorData<T>();
        const size_t num_elements = tensor.GetTensorTypeAndShapeInfo().GetElementCount();

        if (!data || num_elements == 0)
        {
            return false;
        }

        for (size_t i = 0; i < num_elements; i += batch_size)
        {
            const T* batch_start = data + i;
            const T* batch_end = data + std::min(i + batch_size, num_elements);
            if (std::any_of(batch_start, batch_end, predicate))
            {
                return true;
            }
        }

        return false;
    }
    catch (const std::exception& e)
    {
        return false;
    }
}

bool AllNans(const Ort::Value& tensor)
{
    try
    {
        const auto& tensor_info = tensor.GetTensorTypeAndShapeInfo();
        switch (tensor_info.GetElementType())
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return CheckAllInBatches<float>(tensor, [](float val) { return std::isnan(val); });

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return CheckAllInBatches<double>(tensor, [](double val) { return std::isnan(val); });

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return CheckAllInBatches<float>(tensor, [](float val) { return std::isnan(val); });

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return CheckAllInBatches<float>(tensor, [](float val) { return std::isnan(val); });

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return false;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return false;

        default:
            std::cerr << "ERROR: AllNans: Unsupported tensor element type "
                      << static_cast<int>(tensor_info.GetElementType()) << ", assuming no NaNs" << std::endl;
            return false;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: AllNans: Exception occurred: " << e.what() << std::endl;
        return false;
    }
}

bool AnyNans(const Ort::Value& tensor)
{
    try
    {
        const auto& tensor_info = tensor.GetTensorTypeAndShapeInfo();
        const size_t num_elements = tensor_info.GetElementCount();

        if (num_elements == 0)
        {
            return false;
        }

        switch (tensor_info.GetElementType())
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return CheckAnyInBatches<float>(tensor, [](float val) { return std::isnan(val); });

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return CheckAnyInBatches<double>(tensor, [](double val) { return std::isnan(val); });

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return CheckAnyInBatches<float>(tensor, [](float val) { return std::isnan(val); });

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return CheckAnyInBatches<float>(tensor, [](float val) { return std::isnan(val); });

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return false;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return false;

        default:
            std::cerr << "ERROR: AnyNans: Unsupported tensor element type "
                      << static_cast<int>(tensor_info.GetElementType()) << ", assuming no NaNs" << std::endl;
            return false;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: AnyNans: Exception occurred: " << e.what() << std::endl;
        return false;
    }
}
