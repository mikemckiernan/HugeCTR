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

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/communication.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype, typename emtype>
void Communication<dtype, emtype>::initialize_communication() {}

// template <typename dtype, typename emtype>
// Communication<dtype, emtype>::Communication<dtype, emtype>() {

// }

// template <typename dtype, typename emtype>
// Communication<dtype, emtype>::~Communication() {

// }

// reduces the frequent embedding
template <typename dtype, typename emtype>
void Communication<dtype, emtype>::all_reduce() {}

// all-to-all forward and backward for the infrequent embedding
template <typename dtype, typename emtype>
void Communication<dtype, emtype>::all_to_all_v_forward() {}

template <typename dtype, typename emtype>
void Communication<dtype, emtype>::all_to_all_v_backward() {}

// performs all-to-all-reduce on the frequent embedding
template <typename dtype, typename emtype>
void Communication<dtype, emtype>::all_to_all_reduce_frequent() {}

// performs all-to-all-reduce on the frequent and infrequent
// embeddings simultaneously (IB)
template <typename dtype, typename emtype>
void Communication<dtype, emtype>::all_to_all_reduce() {}

template class Communication<uint32_t, __half>;
template class Communication<uint32_t, float>;
template class Communication<unsigned long, __half>;
template class Communication<unsigned long, float>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR