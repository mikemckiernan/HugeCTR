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
#include "HugeCTR/include/tensor2.hpp"


namespace HugeCTR {


namespace hybrid_embedding {


template <typename dtype>
struct Statistics {
 public:
  Statistics(size_t num_samples) {
      // TODO:
      // allocate num_samples categories of data 
      // for categories_sorted and counts_sorted
  }
  ~Statistics();

  uint32_t num_unique_categories;

  // top categories sorted by count
  Tensor2<dtype> categories_sorted;
  Tensor2<uint32_t> counts_sorted;

  void sort_categories_by_count(
    dtype *samples,
    uint32_t num_samples,
    dtype *categories_sorted,
    uint32_t *counts_sorted,
    uint32_t &num_unique_categories,
    cudaStream_t stream);
  void sort_categories_by_count(
    Tensor2<dtype> samples, 
    cudaStream_t stream);
};


}


}