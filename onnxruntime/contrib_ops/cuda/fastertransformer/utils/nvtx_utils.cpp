/*
* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "nvtx_utils.h"

namespace nvtx
{
    std::string get_scope(){ return scope;}
    void add_scope(std::string name){ scope = scope + name + "/"; return;}
    void set_scope(std::string name){ scope = name + "/"; return;}
    void reset_scope(){ scope = ""; return;}
}
