#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Forked from
// https://devicesasg.visualstudio.com/PerceptiveShell/_git/PerceptiveShell?path=/src/perceptiveshell_private/perceptiveshell_private_platform/io.cpp&version=GC855fc8c9d8aef3990717adfca2057210a58382aa&line=176&lineEnd=233&lineStartColumn=1&lineEndColumn=2&lineStyle=plain&_a=contents
bool LoadEncryptedModelData(const std::string& modelPath, std::vector<unsigned char>& data)
{
    auto CheckFileAndGetSize = [](const std::filesystem::path& pathToCheck, uintmax_t& fileSize) -> bool
    {
        std::error_code ec;
        fileSize = std::filesystem::file_size(pathToCheck, ec);
        if (ec)
        {
            return false;
        }

        if (fileSize < 4)
        {
            return false;
        }
        return true;
    };

    uintmax_t fileSize(0);
    if (auto result = CheckFileAndGetSize(modelPath, fileSize); !result)
    {
        return result;
    }

    std::ifstream ifs(modelPath, std::ios::binary);
    if (!ifs.good())
    {
        return false;
    }

    // for paranoia, lets enable exceptions, in case the read fails for some obscure reason
    ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    // could use std::copy() with stream iterators... Runtime is slightly slower so stick with raw read.
    try
    {
        data.resize(fileSize);
        ifs.read(reinterpret_cast<char*>(&data[0]), fileSize);
        if (static_cast<uintmax_t>(ifs.gcount()) != fileSize)
        {
            return false;
        }
    }
    catch (const std::ios_base::failure& ex)
    {
        return false;
    }

    return true;
}
