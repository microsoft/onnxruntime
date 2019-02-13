#include <unordered_map>

namespace onnxruntime{

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