# 
# Copyright (c) 2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set(CUTLASS_INC_PATHS
    /usr/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    /usr/local/include
    $ENV{CUTLASS_DIR}/include/
    )

set(CUTLASS_LIB_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    /usr/local/lib
    /usr/local/lib64
    $ENV{CUTLASS_DIR}/lib
    )

find_path(CUTLASS_INCLUDE_DIR NAMES cutlass.h PATHS ${CUTLASS_INC_PATHS}/cutlass)
find_library(CUTLASS_LIBRARIES NAMES cutlass PATHS ${CUTLASS_LIB_PATHS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUTLASS DEFAULT_MSG CUTLASS_INCLUDE_DIR CUTLASS_LIBRARIES)

if (CUTLASS_FOUND)
  message(STATUS "Found CUTLASS    (include: ${CUTLASS_INCLUDE_DIR}, library: ${CUTLASS_LIBRARIES})")
  mark_as_advanced(CUTLASS_INCLUDE_DIR CUTLASS_LIBRARIES)
endif ()
