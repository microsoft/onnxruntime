#include <iostream>
#include <unordered_map>
#include <gtest/gtest.h>

#include "runtimeParameters.h"

namespace RuntimeParameters
{
    std::unordered_map<std::string, std::string> Parameters;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    for (int i = 1; i < argc; i++)
    {
        std::string argument(argv[i]);
        if (argument.rfind("/p:", 0) == 0)
        {
            auto separatorIndex = argument.find('=');
            auto parameterName = argument.substr(3, separatorIndex - 3);
            auto parameterValue = argument.substr(separatorIndex + 1);
            RuntimeParameters::Parameters[parameterName] = parameterValue;
        }
        else
        {
            std::cerr << "Unrecognized argument " << argument << "\n";
            return -1;
        }
    }
    return RUN_ALL_TESTS();
}
