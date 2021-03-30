// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

#include <list>
#include <algorithm>

#if !defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)

MLAS_THREADPOOL* GetMlasThreadPool(void) {
  static MLAS_THREADPOOL* threadpool = new onnxruntime::concurrency::ThreadPool(
      &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, 2, true);
  return threadpool;
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

  void AddTestRegistor(TestRegistor test_registor) {
    test_registors_.push_back(test_registor);
  }

  size_t RegisterAll(bool is_short_execute) {
    size_t count = 0;
    for (const auto& r : instance().test_registors_) {
      count += r(is_short_execute);
    }
    return count;
  }

 private:
  LongShortExecuteManager() : test_registors_() {}
  LongShortExecuteManager(const LongShortExecuteManager&) = delete;
  LongShortExecuteManager& operator=(const LongShortExecuteManager&) = delete;

  std::list<TestRegistor> test_registors_;
};

bool AddTestRegistor(TestRegistor test_registor) {
  LongShortExecuteManager::instance().AddTestRegistor(test_registor);
  return true;
}

int main(int argc, char** argv) {
  bool is_short_execute = (argc <= 1 || strcmp("--long", argv[1]) != 0);
  std::cout << "-------------------------------------------------------" << std::endl;
  if (is_short_execute) {
    std::cout << "----Running quick check mode. Enable more complete test" << std::endl;
    std::cout << "----  with '--long' as first argument!" << std::endl;
  }
  auto test_count = LongShortExecuteManager::instance().RegisterAll(is_short_execute);
  std::cout << "----Total " << test_count << " tests registered programmablely!" << std::endl;
  std::cout << "-------------------------------------------------------" << std::endl;

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
