#pragma once

template <typename T>
std::shared_ptr<T> PheonixSingleton()
{
    static std::weak_ptr<T> wpInstance;
    static std::mutex m_lock;

    std::lock_guard<std::mutex> lock(m_lock);
    if (auto instance = wpInstance.lock())
    {
        return instance;
    }

    auto instance = std::make_shared<T>();
    wpInstance = instance;
    return instance;
}