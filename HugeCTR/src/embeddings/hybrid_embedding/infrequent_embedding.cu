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

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/utils.cuh"


namespace hybrid_embedding {


template <typename dtype, typename emtype>
void InfrequentEmbedding::initialize_embedding_vectors() {
  // TODO: create initialize_embedding_vectors()
}


template <typename T>
static bool lesser_by_first(const std::pair<T, T>& a, const std::pair<T, T>& b) {
  return (a.first < b.first);
}


/// TODO: GPU version
template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::calculate_model_indices(cudaStream_t stream) {
  std::cout << "WARNING: calculate_model_indices must be done on GPU!" << std::endl;

  size_t local_batch_size = ceildiv<size_t>(data_.batch_size, data_.num_networks);
  const size_t num_tables = data_.table_sizes.size();

  std::vector<dtype> h_samples;
  download_tensor<dtype>(h_samples, data_.samples, stream);
  std::vector<dtype> h_category_location;
  download_tensor<dtype>(h_category_location, model_.category_location, stream);

  std::vector<uint32_t> h_model_indices = std::vector<dtype>(data_.batch_size * num_tables);
  std::vector<uint32_t> h_model_indices_offsets = std::vector<dtype>(data_.num_networks + 1);

  // Prefix sum
  size_t sum = 0;
  for (size_t j = 0; j < data_.batch_size; j++) {
    if (j % local_batch_size == 0) {
      h_model_indices_offsets[j / local_batch_size] = sum;
    }
    for (size_t i = 0; i < num_tables; i++) {
      size_t idx = j * num_tables + i;

      dtype category = h_samples[idx];
      dtype network_id = h_category_location[2 * category];
      bool mask = network_id == model_.global_network_id;

      sum += static_cast<size_t>(mask);

      if (mask) h_model_indices[sum - 1] = idx;
    }
  }
  // Total size stored at the end of the offsets vector
  h_model_indices_offsets[data_.num_networks] = sum;

  upload_tensor(h_model_indices, model_indices_, stream);
  upload_tensor(h_model_indices_offsets, model_indices_offsets_, stream);
}


/// TODO: GPU version
template <typename dtype, typename emtype>
void InfrequentEmbedding<dtype, emtype>::calculate_network_indices(cudaStream_t stream) {
  std::cout << "WARNING: calculate_network_indices must be done on GPU!" << std::endl;

  size_t local_batch_size = ceildiv<size_t>(data_.batch_size, data_.num_networks);
  const size_t num_tables = data_.table_sizes.size();

  std::vector<dtype> h_samples;
  download_tensor<dtype>(h_samples, data_.samples, stream);
  std::vector<dtype> h_category_location;
  download_tensor<dtype>(h_category_location, model_.category_location, stream);

  std::vector<std::pair<uint32_t, uint32_t>> h_network_sources_indices =
      std::vector<std::pair<uint32_t, uint32_t>>(local_batch_size * num_tables);

  // Prefix sum only of this GPU's sub-batch
  size_t sum = 0;
  for (size_t j = local_batch_size * model_.global_network_id;
       j < std::min(data_.batch_size, local_batch_size * (model_.global_network_id + 1)); j++) {
    for (size_t i = 0; i < num_tables; i++) {
      size_t idx = j * num_tables + i;

      dtype category = h_samples[idx];
      dtype network_id = h_category_location[2 * category];
      bool mask = network_id < data_.num_networks;

      sum += static_cast<size_t>(mask);

      uint32_t local_mlp_index = (j - local_batch_size * model_.global_network_id) * num_tables + i;

      if (mask)
        h_network_sources_indices[sum - 1] =
            std::make_pair(static_cast<uint32_t>(model_.global_network_id), local_mlp_index);
    }
  }

  // Sort by source only, otherwise stable
  std::sort(h_network_sources_indices.begin(), h_network_sources_indices.begin() + sum,
            lesser_by_first<dtype>);

  // Retrieve indices
  std::vector<dtype> h_network_indices = std::vector<dtype>(local_batch_size * num_tables);
  for (size_t idx = 0; idx < sum; idx++) {
    h_network_indices[idx] = h_network_sources_indices[idx].second;
  }
  // Compute offsets
  std::vector<dtype> h_network_indices_offsets = std::vector<dtype>(data_.num_networks + 1);
  for (size_t i = 0; i < data_.num_networks; i++) {
    h_network_indices_offsets[i] =
        std::lower_bound(h_network_sources_indices.begin(), h_network_sources_indices.begin() + sum,
                         std::make_pair<uint32_t, uint32_t>(i, 0), lesser_by_first<uint32_t>) -
        h_network_sources_indices.begin();
  }
  // Total size stored at the end of the offsets vector
  h_network_indices_offsets[data_.num_networks] = sum;

  upload_tensor(h_network_indices, network_indices_, stream);
  upload_tensor(h_network_indices_offsets, network_indices_offsets_, stream);
}


template class InfrequentEmbedding<uint32_t, __half>;
template class InfrequentEmbedding<uint32_t, float>;
template class InfrequentEmbedding<size_t, __half>;
template class InfrequentEmbedding<size_t, float>;
}  // namespace HugeCTR
