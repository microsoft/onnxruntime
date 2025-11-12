#include "io_binding.h"

std::unordered_map<std::string, Ort::Value> CreateNamedInputs(const Ort::Session& session, OrtAllocator* allocator)
{
    std::unordered_map<std::string, Ort::Value> inputs;
    const size_t num = session.GetInputCount();
    for (size_t idx = 0; idx < num; ++idx)
    {
        const auto name = session.GetInputNameAllocated(idx, allocator);
        const auto type_info = session.GetInputTypeInfo(idx).GetTensorTypeAndShapeInfo();
        const auto shape = type_info.GetShape();
        inputs.emplace(
            name.get(), Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type_info.GetElementType()));
    }

    return inputs;
}

std::unordered_map<std::string, Ort::Value> CreateNamedOutputs(const Ort::Session& session, OrtAllocator* allocator)
{
    std::unordered_map<std::string, Ort::Value> outputs;
    const size_t num = session.GetOutputCount();
    for (size_t idx = 0; idx < num; ++idx)
    {
        const auto name = session.GetOutputNameAllocated(idx, allocator);
        const auto type_info = session.GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo();
        outputs.emplace(
            name.get(),
            Ort::Value::CreateTensor(
                allocator, type_info.GetShape().data(), type_info.GetShape().size(), type_info.GetElementType()));
    }

    return outputs;
}

Ort::IoBinding CreateBinding(
    Ort::Session& session,
    std::unordered_map<std::string, Ort::Value>& inputs,
    std::unordered_map<std::string, Ort::Value>& outputs)
{
    Ort::IoBinding io_binding(session);
    for (const auto& input : inputs)
    {
        io_binding.BindInput(input.first.c_str(), input.second);
    }
    for (const auto& output : outputs)
    {
        io_binding.BindOutput(output.first.c_str(), output.second);
    }

    return io_binding;
}

std::vector<std::string> CreateTensorNames(const Ort::Session& session, OrtAllocator* allocator, bool is_input)
{
    std::vector<std::string> tensor_names;
    const size_t num_tensors = is_input ? session.GetInputCount() : session.GetOutputCount();
    for (size_t idx = 0; idx < num_tensors; ++idx)
    {
        const auto name =
            is_input ? session.GetInputNameAllocated(idx, allocator) : session.GetOutputNameAllocated(idx, allocator);
        tensor_names.push_back(name.get());
    }

    return tensor_names;
}

std::vector<Ort::Value> CreateTensors(const Ort::Session& session, OrtAllocator* allocator, bool is_input)
{
    std::vector<Ort::Value> tensors;
    const size_t num_tensors = is_input ? session.GetInputCount() : session.GetOutputCount();
    for (size_t idx = 0; idx < num_tensors; ++idx)
    {
        const auto type_info = is_input ? session.GetInputTypeInfo(idx) : session.GetOutputTypeInfo(idx);
        const auto type_and_shape = type_info.GetTensorTypeAndShapeInfo();
        const auto shape = type_and_shape.GetShape();
        tensors.emplace_back(
            Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type_and_shape.GetElementType()));
    }

    return tensors;
}
