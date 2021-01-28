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
#include "HugeCTR/include/data_simulator.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::initialize_embedding_vectors() {
  CudaDeviceContext context(gpu_resource_->get_device_id());

  const size_t num_tables = data_.global_table_sizes.size();
  for (size_t i = 0; i < num_tables; i++) {
    float up_bound = sqrt(1.f / data_.global_table_sizes[i]);
    UniformGenerator::fill(
        frequent_embedding_vectors_[i], -up_bound, up_bound, gpu_resource_->get_sm_count(),
        gpu_resource_->get_replica_uniform_curand_generator(), gpu_resource_->get_stream());
  }
}
template <typename dtype, typename emtype>
__global__ void forward_network_frequent_embedding(
    uint32_t embedding_vec_size, uint32_t global_sample_index_base,
    const uint32_t* frequent_sample_indices,  // local
    const dtype* samples,                     // global
    const dtype* category_frequent_index,     // global
    const float* frequent_embedding_vectors,  // global
    emtype* interaction_layer_input)          // local
{
  int bid = blockIdx.x;   // each block corresponding to one category
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector
  uint32_t index = frequent_sample_indices[bid];
  dtype category = samples[index + global_sample_index_base];
  dtype frequent_index = category_frequent_index[category];
  interaction_layer_input[index * embedding_vec_size + tid] =
      frequent_embedding_vectors[frequent_index * embedding_vec_size + tid];
}

template <typename dtype, typename emtype>
void FrequentEmbedding<dtype, emtype>::forward_network(emtype* interaction_layer_input,
                                                       cudaStream_t stream) {
  // concatenate the embedding vectors into the buffer for
  // top-mlp input

  // Kefeng: type here, use FrequentEmbedding::frequent_sample_indices
  // in short this is what it should do:
  //   for index in frequent_sample_indices:
  //      output[index][0..em_vec_size-1] =
  //      frequent_embedding_vectors_[category_frequent_index[samples[index]]][0..em_vec_size-1]

  // carefully check
  uint32_t samples_per_instance = data_.samples.get_num_elements() / model_.num_instances;
  uint32_t global_sample_index_base = model_.global_instance_id * samples_per_instance;

  const size_t block_size = embedding_vec_size_;
  const size_t grid_size = frequent_sample_indices_.get_num_elements();
  forward_network_frequent_embedding<<<grid_size, block_size, 0, stream>>>(
      embedding_vec_size_, global_sample_index_base, frequent_sample_indices_.get_ptr(),
      data_.samples.get_ptr(), model_.category_frequent_index.get_ptr(),
      frequent_embedding_vectors_block_.get_ptr(), interaction_layer_input);
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