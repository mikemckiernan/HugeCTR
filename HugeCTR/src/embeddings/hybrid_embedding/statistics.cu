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
#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "cub/cub/device/device_radix_sort.cuh"
#include "cub/cub/device/device_run_length_encode.cuh"

namespace HugeCTR {

namespace hybrid_embedding {

///
/// Perform count of categories within the samples and sort the categories by count
///
template <typename dtype>
void Statistics<dtype>::sort_categories_by_count(Tensor2<dtype> samples, cudaStream_t stream) {
  dtype *d_samples = samples.get_ptr();
  uint32_t num_samples = samples.get_size_in_bytes() / sizeof(dtype);
  dtype *d_categories = categories_sorted.get_ptr();
  uint32_t *d_counts = counts_sorted.get_ptr();
  sort_categories_by_count(d_samples, num_samples, d_categories, d_counts, num_unique_categories,
                           stream);  // Kefengs' function
}

// Kefeng, place your implementation here:
inline size_t align_offset(size_t offset) { return offset + (32 - (offset % 32)); }

template <typename dtype>
void Statistics<dtype>::sort_categories_by_count(dtype *samples, uint32_t num_samples,
                                                 dtype *categories_sorted, uint32_t *counts_sorted,
                                                 uint32_t &num_unique_categories,
                                                 cudaStream_t stream) {
  if (num_samples > 0x0fffffff) {
    CK_THROW_(Error_t::WrongInput, "num_samples is too large, overflow for int type");
  }

  size_t size_sort_keys_temp = 0;
  CK_CUDA_THROW_(cub::DeviceRadixSort::SortKeys((void *)nullptr, size_sort_keys_temp,
                                                (dtype *)nullptr, (dtype *)nullptr,
                                                (int)num_samples, 0, sizeof(dtype) * 8, stream));
  size_sort_keys_temp = align_offset(size_sort_keys_temp);
  size_t size_sort_keys_out = align_offset(num_samples * sizeof(dtype));

  size_t size_unique_categories_temp = 0;
  CK_CUDA_THROW_(cub::DeviceRunLengthEncode::Encode(
      (void *)nullptr, size_unique_categories_temp, (dtype *)nullptr, (dtype *)nullptr,
      (uint32_t *)nullptr, (uint32_t *)nullptr, (int)num_samples, stream));

  size_unique_categories_temp = align_offset(size_unique_categories_temp);
  size_t size_unique_categories_out = align_offset(num_samples * sizeof(dtype));
  size_t size_unique_categories_counts = align_offset(num_samples * sizeof(uint32_t));
  size_t size_num_unique_categories = align_offset(sizeof(uint32_t));

  size_t size_sort_pairs_temp = 0;
  CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairsDescending(
      (void *)nullptr, size_sort_pairs_temp, (uint32_t *)nullptr, (uint32_t *)nullptr,
      (dtype *)nullptr, (dtype *)nullptr, (int)num_samples, 0, sizeof(uint32_t) * 8, stream));
  size_sort_pairs_temp = align_offset(size_sort_pairs_temp);
  size_t max_temp_size = size_sort_keys_temp + size_sort_keys_out + size_unique_categories_temp +
                         size_unique_categories_out + size_unique_categories_counts +
                         size_num_unique_categories + size_sort_pairs_temp;
  unsigned char *ptr0 = nullptr;
  CK_CUDA_THROW_(cudaMalloc(&ptr0, max_temp_size));

  size_t offset = 0;
  unsigned char *p_sort_keys_temp = ptr0;  // void*
  offset += size_sort_keys_temp;
  unsigned char *p_sort_keys_out = ptr0 + offset;  // dtype
  offset += size_sort_keys_out;
  unsigned char *p_unique_categories_temp = ptr0 + offset;  // void
  offset += size_unique_categories_temp;
  unsigned char *p_unique_categories_out = ptr0 + offset;  // dtype
  offset += size_unique_categories_out;
  unsigned char *p_unique_categories_counts = ptr0 + offset;  // int
  offset += size_unique_categories_counts;
  unsigned char *p_num_unique_categories = ptr0 + offset;  // int
  offset += size_num_unique_categories;
  unsigned char *p_sort_pairs_temp = ptr0 + offset;  // void

  CK_CUDA_THROW_(cub::DeviceRadixSort::SortKeys((void *)p_sort_keys_temp, size_sort_keys_temp,
                                                samples, (dtype *)p_sort_keys_out, (int)num_samples,
                                                0, sizeof(dtype) * 8, stream));

  CK_CUDA_THROW_(cub::DeviceRunLengthEncode::Encode(
      (void *)p_unique_categories_temp, size_unique_categories_temp, (dtype *)p_sort_keys_out,
      (dtype *)p_unique_categories_out, (uint32_t *)p_unique_categories_counts,
      (uint32_t *)p_num_unique_categories, (int)num_samples, stream));
  CK_CUDA_THROW_(cudaMemcpyAsync((void *)&num_unique_categories, (void *)p_num_unique_categories,
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));

  CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairsDescending(
      (void *)p_sort_pairs_temp, size_sort_pairs_temp, (uint32_t *)p_unique_categories_counts,
      (uint32_t *)counts_sorted, (dtype *)p_unique_categories_out, (dtype *)categories_sorted,
      (int)num_unique_categories, 0, sizeof(uint32_t) * 8, stream));

  CK_CUDA_THROW_(cudaFree(ptr0));
}

template class Statistics<uint32_t>;
template class Statistics<unsigned long>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR