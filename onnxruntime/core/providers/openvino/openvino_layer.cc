#include "openvino_layer.h"
#include "opset_versions.h"
namespace onnxruntime{



const std::unordered_map<std::string, std::tuple<std::string, int>> OpenVINOLayer::createMap(){
    std::unordered_map<std::string, std::tuple<std::string,int>> mMap;

    //mMap.emplace("Add", std::make_tuple("EltwiseSum",OpenVINOEltwiseSum));
    //mMap.emplace("BatchNormalization",std::make_tuple("ScaleShift",OpenVINOScaleShift));
    mMap.emplace("Conv",std::make_tuple("Convolution",OpenVINOConvolution));
    //mMap.emplace("GlobalAveragePool",std::make_tuple("AveragePooling",OpenVINOAveragePooling));
    mMap.emplace("Relu",std::make_tuple("ReLU",OpenVINOReLU));
    //mMap.emplace("Reshape",std::make_tuple("Reshape",OpenVINOReshape));
    //mMap.emplace("Flatten",std::make_tuple("Reshape",OpenVINOReshape));
    // mMap.emplace("Gemm",std::make_tuple("FullyConnectedGemm",OpenVINOFullyConnected));
    //mMap.emplace("MaxPool",std::make_tuple("MaxPooling",OpenVINOMaxPooling));
    //mMap.emplace("AveragePool",std::make_tuple("AveragePooling",OpenVINOAveragePooling));
    //mMap.emplace("Concat",std::make_tuple("Concat",OpenVINOConcat));
    //mMap.emplace("Dropout",std::make_tuple("Ignored",10));
    //mMap.emplace("LRN",std::make_tuple("Normalize",OpenVINONormalize));
    //mMap.emplace("Softmax",std::make_tuple("SoftMax",OpenVINOSoftMax));
    // mMap.emplace("Mul",std::make_tuple("EltwiseMul",OpenVINOEltwiseMul));
    //mMap.emplace("Sum",std::make_tuple("EltwiseSum",OpenVINOEltwiseSum));
    mMap.emplace("Transpose",std::make_tuple("Permute",OpenVINOPermute));
    //mMap.emplace("Identity",std::make_tuple("Ignored",10));
    // mMap.emplace("MatMul",std::make_tuple("FullyConnected",OpenVINOFullyConnected));
    // mMap.emplace("Unsqueeze",std::make_tuple("Reshape",OpenVINOReshape));
    // mMap.emplace("ImageScaler",std::make_tuple("ScaleShift",OpenVINOScaleShift));

    return mMap;
}
const std::unordered_map<std::string, std::tuple<std::string,int>> OpenVINOLayer::opMap = OpenVINOLayer::createMap();

OpenVINOLayer::OpenVINOLayer(std::string name){

    auto it = OpenVINOLayer::opMap.find(name);
    if(it != OpenVINOLayer::opMap.end()){
        this->layerName = std::get<0>(it->second);
        this->opsetVersion = std::get<1>(it->second);
        if(this->layerName == "Reshape")
            supportedOnFPGA = false;
        else if(this->layerName == "Normalize")
            supportedOnFPGA = false;
        else if(this->layerName == "SoftMax")
            supportedOnFPGA = false;
    }
    else{
        this->layerName = "NotSupported";
    }
}

bool OpenVINOLayer::supportedOnPlugin(std::string plugin){
    if(plugin == "CPU")
        return supportedOnCPU;
    else if(plugin == "GPU")
        return supportedOnGPU;
    else if(plugin == "FPGA")
        return supportedOnFPGA;
    else if(plugin == "VPU")
        return supportedOnVPU;
    else
        return false;
}

std::string OpenVINOLayer::getName(){
    return layerName;
}

int OpenVINOLayer::getOpsetVersion(){
    return opsetVersion;
}

}

