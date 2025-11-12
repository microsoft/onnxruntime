#include "dataset_reader.h"

DirPerInputNumpyDatasetReader::DirPerInputNumpyDatasetReader(const std::filesystem::path& dataset_root)
{
    ReadConfig(dataset_root / L"dataset_config.json");

    auto input_dir = dataset_root / L"input_data" / std::filesystem::path(m_input_data_version);
    auto output_dir = dataset_root / L"output_data" / std::filesystem::path(m_output_data_version);
    auto l2_norm_dir = dataset_root / L"l2_norm" / std::filesystem::path(m_output_data_version);

    m_input_numpy_reader = std::make_unique<NumpyTensorsReader>(input_dir, m_input_to_dir);
    m_output_numpy_reader = std::make_unique<NumpyTensorsReader>(output_dir, m_output_to_dir);
    m_l2_norm_reader_writer = std::make_unique<NumpyTensorsStore>(l2_norm_dir, m_output_to_dir);
}

void DirPerInputNumpyDatasetReader::ReadConfig(const std::filesystem::path& config_path)
{
    nlohmann::json config;

    // Check if config file exists.
    if (!std::filesystem::is_regular_file(config_path))
    {
        std::cout << "WARNING: Config file not found at " << config_path << std::endl;
        std::cout << "Model inputs are expected to have the same name as subfolders." << std::endl;
    }
    else
    {
        // Load the config.
        std::ifstream config_file(config_path);
        if (!config_file.is_open())
        {
            std::cout << "WARNING: Failed to load config from " << config_path << std::endl;
        }
        config_file >> config;

        // Read input and output data versions.
        if (config.find("input_data_version") != config.end())
        {
            m_input_data_version = config["input_data_version"].get<std::string>();
        }
        else
        {
            std::cout << "WARNING: input_data_version not found in config: " << config_path << std::endl;
        }
        if (config.find("output_data_version") != config.end())
        {
            m_output_data_version = config["output_data_version"].get<std::string>();
        }
        else
        {
            std::cout << "WARNING: output_data_version not found in config: " << config_path << std::endl;
        }

        // Read input-to-dir name mappings.
        if (config.find("input_to_dir") != config.end())
        {
            for (const auto& entry : config["input_to_dir"].items())
            {
                m_input_to_dir[entry.key()] = entry.value().get<std::string>();
            }
        }
        else
        {
            std::cout << "WARNING: input_to_dir section not found in config: " << config_path << std::endl;
        }

        // Read output-to-dir name mappings.
        if (config.find("output_to_dir") != config.end())
        {
            for (const auto& entry : config["output_to_dir"].items())
            {
                m_output_to_dir[entry.key()] = entry.value().get<std::string>();
            }
        }
        else
        {
            std::cout << "WARNING: output_to_dir section not found in config: " << config_path << std::endl;
        }
    }
}

size_t DirPerInputNumpyDatasetReader::GetNumSamples()
{
    return m_input_numpy_reader->GetNumSamples();
}

std::unordered_map<std::string, std::string> DirPerInputNumpyDatasetReader::GetOutputNames()
{
    return m_output_to_dir;
}

bool DirPerInputNumpyDatasetReader::Load(size_t sample_idx, const std::string& input_name, Ort::Value& input_tensor)
{
    return m_input_numpy_reader->Load(sample_idx, input_name, input_tensor);
}

bool DirPerInputNumpyDatasetReader::LoadOutput(
    size_t sample_idx, const std::string& output_name, Ort::Value& output_tensor)
{
    return m_output_numpy_reader->Load(sample_idx, output_name, output_tensor);
}

bool DirPerInputNumpyDatasetReader::LoadL2Norm(const std::string& output_name, std::vector<float>& l2norm_values)
{
    return m_l2_norm_reader_writer->Load(output_name, l2norm_values);
}

bool DirPerInputNumpyDatasetReader::StoreL2Norm(
    const std::unordered_map<std::string, std::vector<float>>& l2_norm_outputs)
{
    return m_l2_norm_reader_writer->Store(l2_norm_outputs);
}

std::unique_ptr<IDatasetReader> CreateDatasetReader(const std::wstring& dataset_root)
{
    std::unique_ptr<IDatasetReader> dataset_reader = std::make_unique<DirPerInputNumpyDatasetReader>(dataset_root);
    if (dataset_reader == nullptr)
    {
        std::wcout << L"Failed to deduce data layout in " << dataset_root << std::endl;
    }
    return dataset_reader;
}
