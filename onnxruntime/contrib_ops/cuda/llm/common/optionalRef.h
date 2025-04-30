/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>

namespace ort_llm::common
{

/**
 * @brief Wrapper that holds an optional reference and integrates with std containers.
 * @details The wrapper uses a std::optional<std::reference_wrapper<T>> at its core.
            When constructed with a unique or shared ptr with a nullptr value, it is interpreted as not holding a value,
            meaning the std::optional of the wrapper object will be false.
 *
 * @tparam T
 */
template <typename T>
class OptionalRef
{
private:
    std::optional<std::reference_wrapper<T>> opt;

public:
    OptionalRef() = default;

    OptionalRef(T& ref)
        : opt(std::ref(ref))
    {
    }

    OptionalRef(std::nullopt_t)
        : opt(std::nullopt)
    {
    }

    OptionalRef(std::shared_ptr<T> const& ptr)
        : opt(ptr ? std::optional<std::reference_wrapper<T>>(std::ref(*ptr)) : std::nullopt)
    {
    }

    // Constructor for std::shared_ptr<std::remove_const_t<T>> when T is const-qualified
    template <typename U = T, typename = std::enable_if_t<std::is_const_v<U>>>
    OptionalRef(std::shared_ptr<std::remove_const_t<T>> const& ptr)
        : opt(ptr ? std::optional<std::reference_wrapper<T>>(std::ref(*ptr)) : std::nullopt)
    {
    }

    OptionalRef(std::unique_ptr<T> const& ptr)
        : opt(ptr ? std::optional<std::reference_wrapper<T>>(std::ref(*ptr)) : std::nullopt)
    {
    }

    // Constructor for std::unique_ptr<std::remove_const_t<T>> when T is const-qualified
    template <typename U = T, typename = std::enable_if_t<std::is_const_v<U>>>
    OptionalRef(std::unique_ptr<std::remove_const_t<T>> const& ptr)
        : opt(ptr ? std::optional<std::reference_wrapper<T>>(std::ref(*ptr)) : std::nullopt)
    {
    }

    T* operator->() const
    {
        return opt ? &(opt->get()) : nullptr;
    }

    T& operator*() const
    {
        return opt->get();
    }

    explicit operator bool() const
    {
        return opt.has_value();
    }

    bool has_value() const
    {
        return opt.has_value();
    }

    T& value() const
    {
        return opt->get();
    }
};

} // namespace ort_llm::common
