// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <unordered_map>
namespace openvino_ep{

class OpenVINOLayer{

private:
    static const std::unordered_map<std::string, std::tuple<std::string, int>> opMap;
    static const std::unordered_map<std::string, std::tuple<std::string, int>> createMap();
    std::string layerName;
    bool supportedOnCPU = true;
    bool supportedOnGPU = true;
    bool supportedOnFPGA = true;
    bool supportedOnVPU = true;
    int opsetVersion = 0;

public:
    static std::unordered_map<std::string, std::tuple<std::string, int>> getOpMap(){
        return opMap;
    };

    OpenVINOLayer(std::string name);

    bool supportedOnPlugin(std::string name);

    std::string getName();

    int getOpsetVersion();
};

}