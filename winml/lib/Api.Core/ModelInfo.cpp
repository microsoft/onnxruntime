#include "pch.h"

#include "inc/ModelInfo.h"

#include <io.h>
#include <fcntl.h>

#include "FeatureDescriptorFactory.h"
#include "ZeroCopyInputStreamWrapper.h"

#include "google/protobuf/io/zero_copy_stream_impl.h"

using namespace Windows::AI::MachineLearning;
    
static
std::vector<const char*>
GetAllNodeOutputs(const onnx::ModelProto& model_proto)
{
    std::vector<const char*> nodes_outputs;
    auto& graph = model_proto.graph();
    auto& nodes = graph.node();
    for (auto& node : nodes)
    {
        for (auto& node_output : node.output())
        {
            nodes_outputs.push_back(node_output.c_str());
        }
    }
    return nodes_outputs;
}

static
std::vector<const char*>
GetInitializers(const onnx::ModelProto& model_proto)
{
    std::vector<const char*> initializers;
    auto& graph = model_proto.graph();
    auto& graph_initializers = graph.initializer();
    for (auto& initializer : graph_initializers)
    {
        initializers.push_back(initializer.name().c_str());
    }
    return initializers;
}

static
std::vector<const onnx::ValueInfoProto*>
GetInputsWithoutInitializers(const onnx::ModelProto& model_proto)
{
    auto initializers = GetInitializers(model_proto);

    std::vector<const onnx::ValueInfoProto*> inputs_without_initializers;
    auto& graph = model_proto.graph();
    auto& inputs = graph.input();
    for (auto& input : inputs)
    {
        if (input.has_name() && input.has_type())
        {
            auto found_it = std::find_if(
                std::begin(initializers),
                std::end(initializers),
                [&] (auto& initializer)
                {
                    return std::strcmp(initializer, input.name().c_str()) == 0;
                });

            auto is_initializer = found_it != std::end(initializers);
            if (!is_initializer)
            {
                inputs_without_initializers.push_back(&input);
            }
        }
    }
    return inputs_without_initializers;
}

static
std::vector<const onnx::ValueInfoProto*>
GetOutputs(const onnx::ModelProto& model_proto)
{
    std::vector<const onnx::ValueInfoProto*> outputs_with_name;
    auto& graph = model_proto.graph();
    auto& outputs = graph.output();
    for (auto& output : outputs)
    {
        if (output.has_name() && output.has_type())
        {
            outputs_with_name.push_back(&output);
        }
    }
    return outputs_with_name;
}

ModelInfo::ModelInfo(
    const onnx::ModelProto* model_proto)
{
    Initialize(model_proto);
}

void
ModelInfo::Initialize(
    const onnx::ModelProto* model_proto)
{
    // metadata
    for (auto& prop : model_proto->metadata_props())
    {
        model_metadata_[prop.key()] = prop.value();
    }

    WinML::FeatureDescriptorFactory builder(model_metadata_);

    // Create inputs
    auto inputs = GetInputsWithoutInitializers(*model_proto);
    input_features_ = builder.CreateDescriptorsFromValueInfoProtos(inputs);

    // Create outputs
    auto outputs = ::GetOutputs(*model_proto);
    output_features_ = builder.CreateDescriptorsFromValueInfoProtos(outputs);

    // author
    auto has_producer_name = model_proto->has_producer_name();
    author_ = has_producer_name
        ? model_proto->producer_name()
        : "";

    // domain
    auto has_domain = model_proto->has_domain();
    domain_ = has_domain
        ? model_proto->domain()
        : "";

    // name
    auto has_graph = model_proto->has_graph();
    auto graph_has_name = model_proto->graph().has_name();
    auto is_name_available = has_graph && graph_has_name;
    name_ = is_name_available
        ? model_proto->graph().name()
        : "";

    // description
    auto has_description = model_proto->has_doc_string();
    description_ = has_description
        ? model_proto->doc_string()
        : "";

    // version
    auto has_version = model_proto->has_model_version();
    version_ = has_version
        ? model_proto->model_version()
        : 0;
}

// factory methods for creating an ort model from a path 
std::unique_ptr<onnx::ModelProto>
WinML::CreateModelProto(
    const char* path)
{
    int file_descriptor;
    _sopen_s(
        &file_descriptor,
        path,
        O_RDONLY | _O_SEQUENTIAL | _O_BINARY,
        _SH_DENYWR,
        _S_IREAD | _S_IWRITE
    );

    WINML_THROW_HR_IF_TRUE_MSG(
        E_FAIL,
        0 > file_descriptor,
        "Failed"); //errno

    auto stream = google::protobuf::io::FileInputStream(file_descriptor);
    stream.SetCloseOnDelete(true);

    auto model_proto = std::make_unique<onnx::ModelProto>();
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        model_proto->ParseFromZeroCopyStream(&stream),
        "The stream failed to parse.");
    
    return model_proto;
}

// factory methods for creating an ort model from a stream 
std::unique_ptr<onnx::ModelProto>
WinML::CreateModelProto(
    const wss::IRandomAccessStreamReference& stream_reference)
{
    ZeroCopyInputStreamWrapper wrapper(stream_reference);

    auto model_proto = std::make_unique<onnx::ModelProto>();
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        model_proto->ParseFromZeroCopyStream(&wrapper),
        "The stream failed to parse.");

    return model_proto;
}
