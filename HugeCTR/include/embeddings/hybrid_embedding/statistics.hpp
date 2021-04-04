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

#include <cuda_runtime.h>
#include <vector>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {
namespace hybrid_embedding {
template <typename dtype>
struct Statistics {
 public:
  Statistics() : num_samples(0), num_unique_categories(0) {}
  ~Statistics() {}
  Statistics(const Data<dtype> &data)
      : num_samples(data.table_sizes.size() * data.batch_size * data.num_iterations),
        num_unique_categories(0) {
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
    reserve(buf);
    buf->allocate();
  }
  Statistics(dtype num_samples_in) : num_samples(num_samples_in), num_unique_categories(0) {
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
    reserve(buf);
    buf->allocate();
  }
  Statistics(dtype num_samples_in, std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf)
      : num_samples(num_samples_in), num_unique_categories(0) {
    reserve(buf);
  }
  void reserve(std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf) {
    buf->reserve({num_samples, 1}, &categories_sorted);
    buf->reserve({num_samples, 1}, &counts_sorted);
    reserve_temp_storage(buf);
  }

  size_t num_samples;              // input
  uint32_t num_unique_categories;  // to be calculated

  // top categories sorted by count
  Tensor2<dtype> categories_sorted;
  Tensor2<uint32_t> counts_sorted;
  std::vector<Tensor2<unsigned char>> sort_categories_by_count_temp_storages_;
  void reserve_temp_storage(std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf);
  void sort_categories_by_count(const dtype *samples, uint32_t num_samples,
                                dtype *categories_sorted, uint32_t *counts_sorted,
                                uint32_t &num_unique_categories, cudaStream_t stream);
  void sort_categories_by_count(const Tensor2<dtype> &samples, cudaStream_t stream);
};

}  // namespace hybrid_embedding
}  // namespace HugeCTR