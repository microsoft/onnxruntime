// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <list>
#include <memory>

#include "test_util.h"

#if !defined(BUILD_MLAS_NO_ONNXRUNTIME)

MLAS_THREADPOOL* GetMlasThreadPool(void) {
  static auto threadpool = std::make_unique<onnxruntime::concurrency::ThreadPool>(
      &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, 2, true);
  return threadpool.get();
}

#else

MLAS_THREADPOOL* GetMlasThreadPool(void) {
  return nullptr;
}

#endif

// Singleton to avoid initialization order impact.
class LongShortExecuteManager {
 public:
  static LongShortExecuteManager& instance(void) {
    static LongShortExecuteManager s_instance;
    return s_instance;
  };

  void AddTestRegister(TestRegister test_register) {
    test_registers_.push_back(test_register);
  }

  size_t RegisterAll(bool is_short_execute) {
    size_t count = 0;
    for (const auto& r : instance().test_registers_) {
      count += r(is_short_execute);
    }
    return count;
  }

 private:
  LongShortExecuteManager() : test_registers_() {}
  LongShortExecuteManager(const LongShortExecuteManager&) = delete;
  LongShortExecuteManager& operator=(const LongShortExecuteManager&) = delete;

  std::list<TestRegister> test_registers_;
};

bool AddTestRegister(TestRegister test_register) {
  LongShortExecuteManager::instance().AddTestRegister(test_register);
  return true;
}

int main(int argc, char** argv) {
  bool is_short_execute = (argc <= 1 || strcmp("--long", argv[1]) != 0);
  std::cout << "-------------------------------------------------------" << std::endl;
  if (is_short_execute) {
    std::cout << "----Running normal quick check mode. To enable more complete test," << std::endl;
    std::cout << "----  run with '--long' as first argument!" << std::endl;
  }
  auto test_count = LongShortExecuteManager::instance().RegisterAll(is_short_execute);
  std::cout << "----Total " << test_count << " tests registered programmably!" << std::endl;
  std::cout << "-------------------------------------------------------" << std::endl;

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
