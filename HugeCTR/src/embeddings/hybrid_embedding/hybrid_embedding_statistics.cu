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

#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_statistics.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_utils.hpp"
 
#include <algorithm>
#include <iostream>
#include <vector>
  
namespace HugeCTR {


///
/// Perform count of categories within the samples and sort the categories by count
///
template <typename dtype>
void EmbeddingStatistics::sort_categories_by_count(
  Tensor2<dtype> samples,
  cudaStream_t stream
) {
  dtype *d_samples = samples.get_ptr();
  uint32_t num_samples = samples.get_size_in_bytes() / sizeof(dtype);
  dtype *d_categories = categories_sorted.get_ptr();
  uint32_t *d_counts = counts_sorted.get_ptr();
  sort_categories_by_count(
      d_samples, num_samples, d_categories, d_counts,
      num_unique_categories, stream); // Kefengs' function
}


// Kefeng, place your implementation here:
template <typename dtype>
void EmbeddingStatistics::sort_categories_by_count(
  dtype *samples,
  uint32_t num_samples,
  dtype *categories_sorted,
  uint32_t *counts_sorted,
  uint32_t &num_unique_categories,
  cudaStream_t stream);


template class EmbeddingStatistics<uint32_t>;
template class EmbeddingStatistics<size_t>;
}