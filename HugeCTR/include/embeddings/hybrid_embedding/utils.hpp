/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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


#include "HugeCTR/include/tensor2.hpp"
#include <vector>

namespace HugeCTR {


namespace hybrid_embedding {


enum class CommunicationType {IB_NVLink, NVLink};


template <typename dtype>
void download_tensor(std::vector<dtype>& h_tensor, Tensor2<dtype> tensor, cudaStream_t stream);


template <typename dtype>
void upload_tensor(std::vector<dtype>& h_tensor, Tensor2<dtype> tensor, cudaStream_t stream);


}

}