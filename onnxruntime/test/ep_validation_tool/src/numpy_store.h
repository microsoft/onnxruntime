#pragma once

#include "tensors_reader_writer.h"
#include "tensor_utils.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>

class NumpyTensorsWriter : public ITensorsWriter
{
public:
    NumpyTensorsWriter(
        const std::filesystem::path& out_dir, const std::unordered_map<std::string, std::string>& name_to_dir = {})
        : m_root_dir(out_dir),
          m_name_to_dir(name_to_dir)
    {
        EnsureDirectoryExists(m_root_dir);
    }

    bool Store(size_t sample_idx, const std::string& tensor_name, const Ort::Value& tensor) const override
    {
        const auto& mapped_name = GetMappedName(tensor_name);
        if (!EnsureDir(mapped_name))
        {
            return false;
        }

        std::stringstream out_file_name;
        out_file_name << mapped_name << "_" << std::setw(6) << std::setfill('0') << sample_idx << ".npy";

        std::filesystem::path out_path = m_root_dir / mapped_name / out_file_name.str();
        if (!SaveOrtValueAsNumpyArray(out_path, tensor))
        {
            std::cerr << "Error saving " << out_path << std::endl;
            return false;
        }
        return true;
    }

    bool Store(const std::string& tensor_name, const std::vector<float>& data) const
    {
        const auto& mapped_name = GetMappedName(tensor_name);
        std::filesystem::path out_path = m_root_dir / (mapped_name + ".npy");
        return SaveVectorAsNumpyArray(out_path, data);
    }

    bool Store(const std::unordered_map<std::string, std::vector<float>>& data_map) const
    {
        for (const auto& [tensor_name, data] : data_map)
        {
            if (!Store(tensor_name, data))
            {
                return false;
            }
        }
        return true;
    }

private:
    const std::string& GetMappedName(const std::string& tensor_name) const
    {
        auto it = m_name_to_dir.find(tensor_name);
        return (it != m_name_to_dir.end()) ? it->second : tensor_name;
    }

    bool EnsureDir(const std::string& name) const
    {
        std::filesystem::path dir = m_root_dir / name;
        return EnsureDirectoryExists(dir);
    }

    bool EnsureDirectoryExists(const std::filesystem::path& dir) const
    {
        if (!std::filesystem::exists(dir))
        {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec || !std::filesystem::exists(dir))
            {
                std::cerr << "Failed to create directory: " << dir << " (" << ec.message() << ")" << std::endl;
                return false;
            }
        }
        return true;
    }

    bool SaveVectorAsNumpyArray(const std::filesystem::path& file_path, const std::vector<float>& data) const
    {
        std::ofstream file(file_path, std::ios::binary);
        if (!file)
        {
            std::cerr << "ERROR: Failed to create file: " << file_path << std::endl;
            return false;
        }

        return WriteNpyHeader(file, data.size()) && WriteNpyData(file, data.data(), data.size());
    }

    bool WriteNpyHeader(std::ofstream& file, size_t array_size) const
    {
        // Write magic number and version
        file.write("\x93NUMPY\x01\x00", 8);

        // Create header dictionary
        std::string header_dict =
            "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(array_size) + ",), }";

        // Calculate padding for 64-byte alignment
        size_t total_size = 10 + header_dict.length();
        size_t padding = (64 - (total_size % 64)) % 64;
        header_dict.append(padding, ' ');
        header_dict += '\n';

        // Write header length and dictionary
        uint16_t header_len = static_cast<uint16_t>(header_dict.length());
        file.write(reinterpret_cast<const char*>(&header_len), 2);
        file.write(header_dict.c_str(), header_dict.length());

        return file.good();
    }

    bool WriteNpyData(std::ofstream& file, const float* data, size_t size) const
    {
        file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
        return file.good();
    }

private:
    std::filesystem::path m_root_dir;
    std::unordered_map<std::string, std::string> m_name_to_dir;
};

class NumpyTensorsReader : public ITensorsReader
{
public:
    NumpyTensorsReader(
        const std::filesystem::path& dir, const std::unordered_map<std::string, std::string>& name_to_dir = {})
        : m_root_dir(dir),
          m_name_to_dir(name_to_dir)
    {
        m_file_list = EnumerateFiles(m_root_dir);
        m_num_samples = ValidateAndGetSampleCount(m_file_list);
    }

    bool Load(size_t sample_idx, const std::string& tensor_name, Ort::Value& tensor) const override
    {
        if (sample_idx >= m_num_samples)
        {
            std::cerr << "Sample index " << sample_idx << " out of range (max: " << m_num_samples << ")" << std::endl;
            return false;
        }

        const auto& dir_name = GetMappedName(tensor_name);
        if (m_file_list.find(dir_name) == m_file_list.end())
        {
            std::cerr << "Directory '" << dir_name << "' not found" << std::endl;
            return false;
        }

        const auto& sample_paths = m_file_list.at(dir_name);
        if (sample_idx >= sample_paths.size())
        {
            std::cerr << "Sample " << sample_idx << " out of range for " << dir_name << std::endl;
            return false;
        }

        tensor = ReadNumpy(sample_paths[sample_idx]);
        return tensor.IsTensor();
    }

    bool Load(const std::string& tensor_name, std::vector<float>& data) const
    {
        const auto& mapped_name = GetMappedName(tensor_name);
        std::filesystem::path file_path = m_root_dir / (mapped_name + ".npy");

        if (!std::filesystem::exists(file_path))
        {
            std::cerr << "ERROR: File not found: " << file_path << std::endl;
            return false;
        }

        return LoadVectorFromNumpyArray(file_path, data);
    }

    size_t GetNumSamples() const override
    {
        return m_num_samples;
    }

private:
    const std::string& GetMappedName(const std::string& tensor_name) const
    {
        auto it = m_name_to_dir.find(tensor_name);
        return (it != m_name_to_dir.end()) ? it->second : tensor_name;
    }

    std::unordered_map<std::string, std::vector<std::filesystem::path>> EnumerateFiles(
        const std::filesystem::path& root_dir) const
    {
        std::unordered_map<std::string, std::vector<std::filesystem::path>> file_list;

        if (!std::filesystem::exists(root_dir))
        {
            std::cerr << "Root directory does not exist: " << root_dir << std::endl;
            return file_list;
        }

        for (const auto& entry : std::filesystem::directory_iterator(root_dir))
        {
            if (entry.is_directory())
            {
                const std::string dir_name = entry.path().filename().string();
                auto& files = file_list[dir_name];

                for (const auto& file : std::filesystem::directory_iterator(entry))
                {
                    if (file.is_regular_file() && file.path().extension() == ".npy")
                    {
                        files.push_back(file);
                    }
                }

                std::sort(files.begin(), files.end());
            }
        }

        return file_list;
    }

    size_t ValidateAndGetSampleCount(
        const std::unordered_map<std::string, std::vector<std::filesystem::path>>& file_list) const
    {
        if (file_list.empty())
        {
            return 0;
        }

        size_t expected_count = file_list.begin()->second.size();
        for (const auto& [dir_name, files] : file_list)
        {
            if (files.size() != expected_count)
            {
                std::cerr << "Directory " << dir_name << " has " << files.size() << " files, expected "
                          << expected_count << std::endl;
                return 0;
            }
        }
        return expected_count;
    }

    bool LoadVectorFromNumpyArray(const std::filesystem::path& file_path, std::vector<float>& data) const
    {
        std::ifstream file(file_path, std::ios::binary);
        if (!file)
        {
            std::cerr << "ERROR: Failed to open file: " << file_path << std::endl;
            return false;
        }

        size_t array_size;
        std::string dtype;
        if (!ReadNpyHeader(file, array_size, dtype))
        {
            std::cerr << "ERROR: Failed to read NPY header from: " << file_path << std::endl;
            return false;
        }

        if (dtype != "<f4" && dtype != "f4")
        {
            std::cerr << "ERROR: Unsupported data type '" << dtype << "' in: " << file_path << std::endl;
            return false;
        }

        data.resize(array_size);
        size_t data_size = array_size * sizeof(float);

        if (!file.read(reinterpret_cast<char*>(data.data()), data_size) ||
            file.gcount() != static_cast<std::streamsize>(data_size))
        {
            std::cerr << "ERROR: Failed to read data from: " << file_path << std::endl;
            return false;
        }

        return true;
    }

    bool ReadNpyHeader(std::ifstream& file, size_t& array_size, std::string& dtype) const
    {
        char header[8];
        file.read(header, 8);
        if (!file || std::string(header, 6) != std::string("\x93NUMPY", 6) || header[6] != 1 || header[7] != 0)
        {
            std::cerr << "ERROR: Invalid NPY file format" << std::endl;
            return false;
        }

        // Read header length
        uint16_t header_len;
        file.read(reinterpret_cast<char*>(&header_len), 2);
        if (!file)
        {
            std::cerr << "ERROR: Failed to read header length" << std::endl;
            return false;
        }

        // Read and parse header dictionary
        std::string header_dict(header_len, '\0');
        file.read(&header_dict[0], header_len);
        if (!file)
        {
            std::cerr << "ERROR: Failed to read header dictionary" << std::endl;
            return false;
        }

        return ParseHeaderDict(header_dict, array_size, dtype);
    }

    bool ParseHeaderDict(const std::string& header_dict, size_t& array_size, std::string& dtype) const
    {
        // Extract dtype
        size_t descr_pos = header_dict.find("'descr':");
        if (descr_pos == std::string::npos)
        {
            return false;
        }

        size_t dtype_start = header_dict.find("'", descr_pos + 8) + 1;
        size_t dtype_end = header_dict.find("'", dtype_start);
        if (dtype_start == std::string::npos || dtype_end == std::string::npos)
        {
            return false;
        }

        dtype = header_dict.substr(dtype_start, dtype_end - dtype_start);

        // Extract shape
        size_t shape_pos = header_dict.find("'shape':");
        if (shape_pos == std::string::npos)
        {
            return false;
        }

        size_t shape_start = header_dict.find("(", shape_pos) + 1;
        size_t shape_end = header_dict.find_first_of(",)", shape_start);
        if (shape_start == std::string::npos || shape_end == std::string::npos)
        {
            return false;
        }

        try
        {
            array_size = std::stoull(header_dict.substr(shape_start, shape_end - shape_start));
            return true;
        }
        catch (const std::exception&)
        {
            return false;
        }
    }

private:
    std::filesystem::path m_root_dir;
    std::unordered_map<std::string, std::string> m_name_to_dir;
    std::unordered_map<std::string, std::vector<std::filesystem::path>> m_file_list;
    size_t m_num_samples = 0;
};

class NumpyTensorsStore : public NumpyTensorsWriter, public NumpyTensorsReader
{
public:
    NumpyTensorsStore(const std::filesystem::path& dir, const std::string& type = "")
        : NumpyTensorsWriter(dir / type),
          NumpyTensorsReader(dir / type)
    {
    }

    NumpyTensorsStore(
        const std::filesystem::path& dir,
        const std::unordered_map<std::string, std::string>& name_to_dir,
        const std::string& type = "")
        : NumpyTensorsWriter(dir / type, name_to_dir),
          NumpyTensorsReader(dir / type, name_to_dir)
    {
    }
};
