// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <ctime>
#include "core/common/logging/logging.h"
#include "core/framework/op_kernel.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "core/platform/ort_mutex.h"
#include "cuda_runtime_api.h"

namespace onnxruntime {

namespace tensorrt_env_vars {
static const std::string kMaxPartitionIterations = "ORT_TENSORRT_MAX_PARTITION_ITERATIONS";
static const std::string kMinSubgraphSize = "ORT_TENSORRT_MIN_SUBGRAPH_SIZE";
static const std::string kMaxWorkspaceSize = "ORT_TENSORRT_MAX_WORKSPACE_SIZE";
static const std::string kFP16Enable = "ORT_TENSORRT_FP16_ENABLE";
}  // namespace tensorrt_env_vars

//slx INT8
inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

class IBatchStream
{
public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual float* getBatch() = 0;
    virtual float* getLabels() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims getDims() const = 0;
};

class BatchStream : public IBatchStream
{
public:
    BatchStream(int batchSize, int maxBatches, nvinfer1::Dims dims)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{dims} //!< We already know the dimensions of MNIST images.
    {
        //readDataFile(locateFile(dataFile, directories));
        //readLabelsFile(locateFile(labelsFile, directories));
    }

    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount >= mMaxBatches)
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {
        return mData.data() + (mBatchCount * mBatchSize * volume(mDims));
    }

    float* getLabels() override
    {
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
        return nvinfer1::Dims{4, {mBatchSize, mDims.d[0], mDims.d[1], mDims.d[2]}, {}};
    }

private:
/*
    void readDataFile(const std::string& dataFilePath)
    {
        std::ifstream file{dataFilePath.c_str(), std::ios::binary};

        int magicNumber, numImages, imageH, imageW;
        file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        // All values in the MNIST files are big endian.
        magicNumber = samplesCommon::swapEndianness(magicNumber);
        assert(magicNumber == 2051 && "Magic Number does not match the expected value for an MNIST image set");

        // Read number of images and dimensions
        file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        file.read(reinterpret_cast<char*>(&imageH), sizeof(imageH));
        file.read(reinterpret_cast<char*>(&imageW), sizeof(imageW));

        numImages = samplesCommon::swapEndianness(numImages);
        imageH = samplesCommon::swapEndianness(imageH);
        imageW = samplesCommon::swapEndianness(imageW);

        // The MNIST data is made up of unsigned bytes, so we need to cast to float and normalize.
        int numElements = numImages * imageH * imageW;
        std::vector<uint8_t> rawData(numElements);
        file.read(reinterpret_cast<char*>(rawData.data()), numElements * sizeof(uint8_t));
        mData.resize(numElements);
        std::transform(
            rawData.begin(), rawData.end(), mData.begin(), [](uint8_t val) { return static_cast<float>(val) / 255.f; });
    }

    void readLabelsFile(const std::string& labelsFilePath)
    {
        std::ifstream file{labelsFilePath.c_str(), std::ios::binary};
        int magicNumber, numImages;
        file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        // All values in the MNIST files are big endian.
        magicNumber = samplesCommon::swapEndianness(magicNumber);
        assert(magicNumber == 2049 && "Magic Number does not match the expected value for an MNIST labels file");

        file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        numImages = samplesCommon::swapEndianness(numImages);

        std::vector<uint8_t> rawLabels(numImages);
        file.read(reinterpret_cast<char*>(rawLabels.data()), numImages * sizeof(uint8_t));
        mLabels.resize(numImages);
        std::transform(
            rawLabels.begin(), rawLabels.end(), mLabels.begin(), [](uint8_t val) { return static_cast<float>(val); });
    }
*/
    int mBatchSize{0};
    int mBatchCount{0}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    nvinfer1::Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};
};

//! \class EntropyCalibratorImpl
//!
//! \brief Implements common functionality for Entropy calibrators.
//!
template <typename TBatchStream>
class EntropyCalibratorImpl
{
public:
    EntropyCalibratorImpl(
        TBatchStream stream, int firstBatch, std::string networkName, const char* inputBlobName, bool readCache = true)
        : mStream{stream}
        , mCalibrationTableName("CalibrationTable" + networkName)
        , mInputBlobName(inputBlobName)
        , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = volume(dims);
        cudaMalloc(&mDeviceInput, mInputCount * sizeof(float));//CUDA_RETURN_IF_ERROR
        mStream.reset(firstBatch);
    }

    virtual ~EntropyCalibratorImpl()
    {
        cudaFree(mDeviceInput);//CUDA_RETURN_IF_ERROR
    }

    int getBatchSize() const
    {
        return mStream.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings)
    {
        std::cout << "nbBindings: " << nbBindings << std::endl;//slx
        if (!mStream.next())
        {
            return false;
        }
        cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice);//CHECK
        assert(!strcmp(names[0], mInputBlobName));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length)
    {
        mCalibrationCache.clear();
        std::istringstream input(mCalibrationTableName, std::ios::binary);//ifstream
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length)
    {
        std::ostringstream output(mCalibrationTableName, std::ios::binary);//ofstream
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    TBatchStream mStream;
    size_t mInputCount;
    std::string mCalibrationTableName;
    const char* mInputBlobName;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
template <typename TBatchStream>
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(
        TBatchStream stream, int firstBatch, const char* networkName, const char* inputBlobName, bool readCache = true)
        : mImpl(stream, firstBatch, networkName, inputBlobName, readCache)
    {
    }

    int getBatchSize() const override
    {
        return mImpl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) override
    {
        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        mImpl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl<TBatchStream> mImpl;
};

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
class MyInt8EntropyCalibrator2: public nvinfer1::IInt8EntropyCalibrator {
public:
    MyInt8EntropyCalibrator2(int batchSize, int maxBatches, void* input_bindings,
                             const std::string networkName, const char* inputBlobName, bool readCache = true)
        : _batchSize(batchSize)
        , _maxBatches(maxBatches)
        //, _currentBatch(0)
        , input_bindings_(input_bindings)
        , _networkName(networkName)
        , _calibrationTableName("CalibrationTable" + networkName)
        , _inputBlobName(inputBlobName)
        , _readCache(readCache) {
            //Dims d = _stream.getInputDims();
            //_inputCount = _stream.getBatchSize() * d.d[1] * d.d[2] * d.d[3];
            //cudaMalloc(&_deviceInput, _inputCount * sizeof(float));
        }

    int getBatchSize() const override {return _batchSize;}

    //virtual ~Int8EntropyCalibrator() {cudaFree(_deviceInput);}

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
        /*
        if (!_stream.next())
            return false;

        cudaMemcpy(_deviceInput, _stream.getBatch(), _inputCount * sizeof(float), cudaMemcpyHostToDevice);
        bindings[0] = _deviceInput;
        */
        std::cout << "nbBindings: " << nbBindings << std::endl;
        if (_currentBatch == _maxBatches)
            return false;
        assert(!strcmp(names[0], _inputBlobName));//??
        bindings[0] = input_bindings_;
        _currentBatch++;
        return true;
    }

    const void* readCalibrationCache(size_t& length) {
        _calibrationCache.clear();
        std::istringstream input(_calibrationTableName, std::ios::binary);
        input >> std::noskipws;
        if (_readCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                      std::back_inserter(_calibrationCache));

        length = _calibrationCache.size();
        return length ? &_calibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) {
        std::ostringstream output(_calibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }
/* //??
    void reset() {
        _currentBatch = 0;
    }
*/
private:
    int _batchSize;
    int _maxBatches;
    int _currentBatch = 0;
    void* input_bindings_ {nullptr};//[]
    const std::string _networkName;
    std::string _calibrationTableName;
    const char* _inputBlobName;
    bool _readCache {true};
    //ImageStream _stream;

    const std::string _calibrationCacheName;
    std::vector<char> _calibrationCache;
    //size_t _inputCount;
    //void* _deviceInput {nullptr};

};
//

class TensorrtLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;

 public:
  TensorrtLogger(Severity verbosity = Severity::kWARNING)
      : verbosity_(verbosity) {}
  void log(Severity severity, const char* msg) override {
    if (severity <= verbosity_) {
      time_t rawtime = std::time(0);
      char buf[256];
      strftime(&buf[0], 256,
               "%Y-%m-%d %H:%M:%S",
               std::gmtime(&rawtime));
      const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR ? "  ERROR" : severity == Severity::kWARNING ? "WARNING" : severity == Severity::kINFO ? "   INFO" : "UNKNOWN");
      LOGS_DEFAULT(WARNING) << "[" << buf << " " << sevstr << "] " << msg;
    }
  }
};

// Information needed to construct trt execution providers.
struct TensorrtExecutionProviderInfo {
  int device_id{0};
};

// Information to construct kernel function state.
struct TensorrtFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  nvonnxparser::IParser* parser = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  nvinfer1::IBuilder* builder = nullptr;
  nvinfer1::INetworkDefinition* network = nullptr;
  std::vector<std::vector<int>> input_info;
  std::vector<std::vector<int>> output_info;
  std::unordered_map<int, std::unordered_map<int, std::pair<int64_t, int64_t>>> input_shape_ranges;
  std::vector<std::vector<int64_t>> output_shapes;
  OrtMutex* tensorrt_mu_ptr = nullptr;
  bool* fp16_enable_ptr = nullptr;
  size_t* max_workspace_size_ptr = nullptr;
};

// Logical device representation.
class TensorrtExecutionProvider : public IExecutionProvider {
 public:
  explicit TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info);
  virtual ~TensorrtExecutionProvider();

  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  int GetDeviceId() const { return device_id_; }

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

 private:
  size_t max_workspace_size_ = 1 << 30;  // 1GB
  int max_partition_iterations_ = 1000;
  int min_subgraph_size_ = 1;
  bool fp16_enable_ = false;

  struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
      if (obj) {
        obj->destroy();
      }
    }
  };

  template <typename T>
  using unique_pointer = std::unique_ptr<T, InferDeleter>;

  OrtMutex tensorrt_mu_;
  int device_id_;
  std::unordered_map<std::string, unique_pointer<nvonnxparser::IParser>> parsers_;
  std::unordered_map<std::string, unique_pointer<nvinfer1::ICudaEngine>> engines_;
  std::unordered_map<std::string, unique_pointer<nvinfer1::IExecutionContext>> contexts_;
  std::unordered_map<std::string, unique_pointer<nvinfer1::IBuilder>> builders_;
  std::unordered_map<std::string, unique_pointer<nvinfer1::INetworkDefinition>> networks_;
  std::unordered_map<std::string, std::vector<std::vector<int>>> input_info_;
  std::unordered_map<std::string, std::vector<std::vector<int>>> output_info_;
  std::unordered_map<std::string, std::unordered_map<int, std::unordered_map<int, std::pair<int64_t, int64_t>>>> input_shape_ranges_;
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> output_shapes_;

  /**Get IndexedSubGraph based on node list of the subgraph*/
  std::unique_ptr<IndexedSubGraph> GetSubGraph(SubGraph_t graph_nodes_index, int& kernels_index,
                                               const onnxruntime::GraphViewer& graph) const;

  /**
  Get TensorRT supported node lists by calling Onnx-TensorRT parser recursively. Since each time the parser
  can only detect first unsupported node failure, it needs to wait for Onnxruntime to partition the graph
  and then detect next failure again. If there are too many iterations, which means many nodes in the graph
  are not supported by TensorRT, the process will be terminated and the whole graph is simply assigned to
  other execution provider.
  */
  SubGraphCollection_t GetSupportedList(SubGraphCollection_t supported_nodes_list, int iterations, const int max_iterations,
                                        const onnxruntime::GraphViewer& graph, bool* early_termination) const;

  void RemoveTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const onnxruntime::GraphViewer& graph) const;

  AllocatorPtr allocator_;
};

}  // namespace onnxruntime
