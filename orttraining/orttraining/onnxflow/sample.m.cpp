#include "orttraining/onnxflow/csrc/load_parameters.h"
#include <filesystem>
#include <iostream>
#include <onnx/onnx_pb.h>

int main()
{
    std::string path_to_parameters_proto;
    std::cout << "Provide the absolute path to the parameters.of file\n";
    std::cin >> path_to_parameters_proto;
    std::filesystem::path path{path_to_parameters_proto};

    auto parameters = onnxflow::load_parameters(std::filesystem::absolute(path).string());

    std::cout << "The parameters are:\n";
    for (const auto& param : parameters.parameters())
    {
        if (param.is_parameter())
        {
            onnx::TensorProto tensor;
            param.data().UnpackTo(&tensor);
            std::cout << "<" << tensor.name() << ", requires_grad=" << (param.requires_grad() ? "True" : "False") << ">" << std::endl;
        } else
        {
            onnx::ValueInfoProto valueinfo;
            param.data().UnpackTo(&valueinfo);
            std::cout << "<" << valueinfo.name() << ", requires_grad=" << (param.requires_grad() ? "True" : "False") << ">" << std::endl;
        }
    }

    return 0;
}