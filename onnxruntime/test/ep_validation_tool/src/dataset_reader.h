#pragma once

#include "numpy_store.h"
#include "tensor_utils.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <iostream>
#include <vector>

class IDatasetReader
{
public:
    virtual size_t GetNumSamples() = 0;
    virtual bool Load(size_t sample_idx, const std::string& name, Ort::Value& tensor) = 0;
    virtual bool LoadOutput(size_t sample_idx, const std::string& output_name, Ort::Value& output_tensor) = 0;
    virtual bool LoadL2Norm(const std::string& output_name, std::vector<float>& l2norm_values) = 0;
    virtual bool StoreL2Norm(const std::unordered_map<std::string, std::vector<float>>& l2_norm_outputs) = 0;
    virtual std::unordered_map<std::string, std::string> GetOutputNames() = 0;

    virtual ~IDatasetReader() = default;
};

class DirPerInputNumpyDatasetReader : public IDatasetReader
{
public:
    DirPerInputNumpyDatasetReader(const std::filesystem::path& dataset_root);

    size_t GetNumSamples() override;
    bool Load(size_t sample_idx, const std::string& name, Ort::Value& tensor) override;

    bool LoadOutput(size_t sample_idx, const std::string& output_name, Ort::Value& output_tensor);

    bool LoadL2Norm(const std::string& output_name, std::vector<float>& l2norm_values);

    bool StoreL2Norm(const std::unordered_map<std::string, std::vector<float>>& l2_norm_outputs);

    std::unordered_map<std::string, std::string> GetOutputNames();

private:
    void ReadConfig(const std::filesystem::path& config_path);

private:
    std::unordered_map<std::filesystem::path, std::vector<std::filesystem::path>> m_files_in_dir;
    std::unordered_map<std::string, std::string> m_input_to_dir;
    std::unordered_map<std::string, std::string> m_output_to_dir;
    std::string m_input_data_version;
    std::string m_output_data_version;
    size_t m_num_samples = 0;
    std::unique_ptr<NumpyTensorsReader> m_input_numpy_reader;
    std::unique_ptr<NumpyTensorsReader> m_output_numpy_reader;
    std::unique_ptr<NumpyTensorsStore> m_l2_norm_reader_writer;
};

std::unique_ptr<IDatasetReader> CreateDatasetReader(const std::wstring& dataset_root);
