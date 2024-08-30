cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

cmake_policy(SET CMP0092 NEW)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0117 NEW)
# Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
cmake_policy(SET CMP0104 OLD)

# Enable Hot Reload for MSVC compilers if supported.
cmake_policy(SET CMP0141 NEW)