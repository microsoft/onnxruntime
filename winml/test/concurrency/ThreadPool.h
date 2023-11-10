#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <future>

class ThreadPool {
 private:
  std::condition_variable m_cond_var;
  bool m_destruct_pool;
  std::mutex m_mutex;
  std::vector<std::thread> m_threads;
  std::queue<std::function<void()>> m_work_queue;

 public:
  ThreadPool(unsigned int initial_pool_size);
  ~ThreadPool();
  template <typename F, typename... Args>
  inline auto SubmitWork(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
    auto func = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    auto task = std::make_shared<std::packaged_task<decltype(f(args...))()>>(std::forward<decltype(func)>(func));
    {
      std::lock_guard<std::mutex> lock(m_mutex);
            // wrap packed task into a void return function type so that it can be stored in queue
      m_work_queue.push([task]() { (*task)(); });
    }

    m_cond_var.notify_one(); // unblocks one of the waiting threads
    return task->get_future();
  }
};
