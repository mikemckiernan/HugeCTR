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
struct Data {
  std::vector<uint32_t> global_table_sizes;
  std::vector<uint32_t> local_table_sizes;
  size_t num_tables;
  size_t batch_size;
  size_t num_iterations;
  size_t num_networks;

  Tensor2<dtype> samples;

  // convert raw input data such that categories of different
  // categorical features have unique indices
  void data_to_unique_categories(Tensor2<dtype> data, cudaStream_t stream);
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR