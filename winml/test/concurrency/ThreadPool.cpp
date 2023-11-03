#include "testPch.h"
#include "ThreadPool.h"
#include <ctime>

ThreadPool::ThreadPool(unsigned int initial_pool_size) : m_destruct_pool(false), m_threads() {
  for (unsigned int i = 0; i < initial_pool_size; i++) {
    m_threads.emplace_back([this]() {
      while (true) {
        std::unique_lock<std::mutex> lock(m_mutex);
        // thread listening for event and acquire lock if event triggered
        m_cond_var.wait(lock, [this] { return m_destruct_pool || !m_work_queue.empty(); });
        if (!m_work_queue.empty()) {
          auto work = m_work_queue.front();
          m_work_queue.pop();
          lock.unlock();
          work();
        } else {
          // Work queue is empty but lock acquired
          // This means we are destructing the pool
          break;
        }
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  m_destruct_pool = true;
  m_cond_var.notify_all();  // notify destruction to threads
  for (auto& thread : m_threads) {
    thread.join();
  }
}
