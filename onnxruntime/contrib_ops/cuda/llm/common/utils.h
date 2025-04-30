/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <algorithm>
#include <initializer_list>
#include <string>

#ifndef _WIN32
#include <pthread.h>
#endif

namespace ort_llm::common
{

inline bool setThreadName(std::string const& name)
{
#ifdef _WIN32
    return false;
#else
    auto const ret = pthread_setname_np(pthread_self(), name.c_str());
    return !ret;
#endif
}

template <typename T>
bool contains(std::initializer_list<T> const& c, T const& v)
{
    return std::find(c.begin(), c.end(), v) != c.end();
}

} // namespace ort_llm::common
