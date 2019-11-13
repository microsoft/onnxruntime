#pragma once
#include "Std.h"

#define RAPIDJSON_ENDIAN RAPIDJSON_LITTLEENDIAN 
#define RAPIDJSON_HAS_STDSTRING 1
#define RAPIDJSON_NO_SIZETYPEDEFINE 1
namespace rapidjson { using SizeType = uint32_t; }
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/error/en.h>

namespace JsonHelpers
{
    bool ReadFile(const std::string& filePath, std::string& fileContents);
}