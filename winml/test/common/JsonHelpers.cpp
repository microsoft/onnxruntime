#include "testPch.h"
#include "JsonHelpers.h"
#include <fstream>

bool JsonHelpers::ReadFile(const std::string & filePath, std::string & fileContents)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        throw std::invalid_argument(std::string("could not open: ") + filePath);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    fileContents.resize(size);
    file.read(&fileContents[0], size);

    return (fileContents.size() > 0);
}
