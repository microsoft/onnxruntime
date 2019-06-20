#pragma once

#include "server/environment.h"
#include "test/test_environment.h"

namespace onnxruntime{
    namespace server{
        namespace test{
            ServerEnvironment* ServerEnv();
            class TestServerEnvironment{
                public:
                TestServerEnvironment();
                ~TestServerEnvironment();

                TestServerEnvironment(const TestServerEnvironment&) = delete;
                TestServerEnvironment(TestServerEnvironment&&) = default;
               
            };
        }
    }
}