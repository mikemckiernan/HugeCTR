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
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::initialize_embedding_vectors() {
  // TODO: create intialize_embedding_vectors()

  // calculate table_offsets
  //
  // size_t network_batch_size = data_.batch_size / data_.num_networks;
  // const size_t num_tables = data_.table_sizes.size();
  // std::vector<dtype> embedding_offsets(num_tables);
  // dtype embedding_offset = (dtype)0;
  // for (size_t embedding = 0; embedding < num_tables; ++embedding) {
  //   embedding_offsets[embedding] = embedding_offset;
  //   embedding_offset += table_sizes[embedding];
  // }

  // Initialize frequent_embedding_vectors_,
  // for category, buffer_location in enumerate(model_.category_frequent_index):
  //   if buffer_location < num_categories:
  //     # initialize category;
  //     # lookup to which table category belongs using table_offset array
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::forward_network(const emtype *interaction_layer_input) {
  // concatenate the embedding vectors into the buffer for
  // top-mlp input

  // Kefeng: type here, use FrequentEmbedding::frequent_sample_indices
  // in short this is what it should do:
  //   for index in frequent_sample_indices:
  //      output[index][0..em_vec_size-1] =
  //      frequent_embedding_vectors_[category_frequent_index[samples[index]]][0..em_vec_size-1]
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::update_network() {
  // TODO: create update()
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::update_model() {
  // TODO: create update()
}

template class FrequentEmbedding<uint32_t, __half>;
template class FrequentEmbedding<uint32_t, float>;
template class FrequentEmbedding<unsigned long, __half>;
template class FrequentEmbedding<unsigned long, float>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR