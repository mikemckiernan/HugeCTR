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
 #include <cub/cub.cuh>
 #include "HugeCTR/include/common.hpp"
 #include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
 #include "HugeCTR/include/tensor2.hpp"
 
 namespace HugeCTR {
 namespace hybrid_embedding {
 
 ///
 /// Perform count of categories within the samples and sort the categories by count
 ///
 template <typename dtype>
 void Statistics<dtype>::sort_categories_by_count(const Tensor2<dtype> &samples,
                                                  cudaStream_t stream) {
   const dtype *d_samples = samples.get_ptr();
   uint32_t num_samples = samples.get_size_in_bytes() / sizeof(dtype);
   dtype *d_categories = categories_sorted.get_ptr();
   uint32_t *d_counts = counts_sorted.get_ptr();
   sort_categories_by_count(d_samples, num_samples, d_categories, d_counts, num_unique_categories,
                            stream);  // Kefengs' function
   //categories_sorted.reset_shape({num_unique_categories, 1});
   //counts_sorted.reset_shape({num_unique_categories, 1});
 }
 
 template <typename dtype>
 void Statistics<dtype>::reserve_temp_storage(std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf) {
   size_t size_sort_keys_temp = 0;
   sort_categories_by_count_temp_storages_.resize(7);
   CK_CUDA_THROW_(cub::DeviceRadixSort::SortKeys((void *)nullptr, size_sort_keys_temp,
                                                 (dtype *)nullptr, (dtype *)nullptr,
                                                 (int)num_samples, 0, sizeof(dtype) * 8, 0));
   buf->reserve({size_sort_keys_temp, 1}, &sort_categories_by_count_temp_storages_[0]);
   buf->reserve({num_samples * sizeof(dtype), 1}, &sort_categories_by_count_temp_storages_[1]);
   size_t size_unique_categories_temp = 0;
   CK_CUDA_THROW_(cub::DeviceRunLengthEncode::Encode(
       (void *)nullptr, size_unique_categories_temp, (dtype *)nullptr, (dtype *)nullptr,
       (uint32_t *)nullptr, (uint32_t *)nullptr, (int)num_samples, 0));
 
   buf->reserve({size_unique_categories_temp, 1}, &sort_categories_by_count_temp_storages_[2]);
   buf->reserve({num_samples * sizeof(dtype), 1}, &sort_categories_by_count_temp_storages_[3]);
   buf->reserve({num_samples * sizeof(uint32_t), 1}, &sort_categories_by_count_temp_storages_[4]);
   buf->reserve({sizeof(uint32_t), 1}, &sort_categories_by_count_temp_storages_[5]);
 
   size_t size_sort_pairs_temp = 0;
   CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairsDescending(
       (void *)nullptr, size_sort_pairs_temp, (uint32_t *)nullptr, (uint32_t *)nullptr,
       (dtype *)nullptr, (dtype *)nullptr, (int)num_samples, 0, sizeof(uint32_t) * 8, 0));
   buf->reserve({size_sort_pairs_temp, 1}, &sort_categories_by_count_temp_storages_[6]);
 };
 
 template <typename dtype>
 void Statistics<dtype>::sort_categories_by_count(const dtype *samples, uint32_t num_samples,
                                                  dtype *categories_sorted, uint32_t *counts_sorted,
                                                  uint32_t &num_unique_categories,
                                                  cudaStream_t stream) {
   if (num_samples > 0x0fffffff) {
     CK_THROW_(Error_t::WrongInput, "num_samples is too large, overflow for int type");
   }
   void *p_sort_keys_temp =
       reinterpret_cast<void *>(sort_categories_by_count_temp_storages_[0].get_ptr());  // void*
   dtype *p_sort_keys_out =
       reinterpret_cast<dtype *>(sort_categories_by_count_temp_storages_[1].get_ptr());  // dtype*
   void *p_unique_categories_temp =
       reinterpret_cast<void *>(sort_categories_by_count_temp_storages_[2].get_ptr());  // void*
   dtype *p_unique_categories_out =
       reinterpret_cast<dtype *>(sort_categories_by_count_temp_storages_[3].get_ptr());  // dtype*
   uint32_t *p_unique_categories_counts = reinterpret_cast<uint32_t *>(
       sort_categories_by_count_temp_storages_[4].get_ptr());  // uint32_t*
   uint32_t *p_num_unique_categories = reinterpret_cast<uint32_t *>(
       sort_categories_by_count_temp_storages_[5].get_ptr());  // uint32*
   void *p_sort_pairs_temp =
       reinterpret_cast<void *>(sort_categories_by_count_temp_storages_[6].get_ptr());  // void*
 
   size_t temp_size = sort_categories_by_count_temp_storages_[0].get_size_in_bytes();
   CK_CUDA_THROW_(cub::DeviceRadixSort::SortKeys(p_sort_keys_temp, temp_size, samples,
                                                 p_sort_keys_out, (int)num_samples, 0,
                                                 sizeof(dtype) * 8, stream));
 
   temp_size = sort_categories_by_count_temp_storages_[2].get_size_in_bytes();
   CK_CUDA_THROW_(cub::DeviceRunLengthEncode::Encode(
       p_unique_categories_temp, temp_size, p_sort_keys_out, p_unique_categories_out,
       p_unique_categories_counts, p_num_unique_categories, (int)num_samples, stream));
   CK_CUDA_THROW_(cudaMemcpyAsync((void *)&num_unique_categories, (void *)p_num_unique_categories,
                                  sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
 
   temp_size = sort_categories_by_count_temp_storages_[6].get_size_in_bytes();
   CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairsDescending(
       p_sort_pairs_temp, temp_size, p_unique_categories_counts, counts_sorted,
       p_unique_categories_out, categories_sorted, (int)num_unique_categories, 0,
       sizeof(uint32_t) * 8, stream));
 }
 
 template class Statistics<uint32_t>;
 template class Statistics<long long>;
 template class Statistics<size_t>;

 }  // namespace hybrid_embedding
 
 }  // namespace HugeCTR