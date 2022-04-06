
#include "load_parameters.h"
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <sstream>


namespace onnxflow {

OnnxFlowParameters load_parameters(const std::string& path_to_file)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    OnnxFlowParameters params;
    std::ifstream t(path_to_file);
    std::stringstream buffer;
    buffer << t.rdbuf();
    params.ParseFromString(buffer.str());

    return params;
}

} // end namespace onnxflow
