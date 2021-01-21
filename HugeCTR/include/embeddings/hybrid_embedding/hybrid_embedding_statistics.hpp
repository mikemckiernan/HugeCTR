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

#include "HugeCTR/include/tensor2.hpp"
#include <vector>

namespace HugeCTR {


template <typename dtype>
struct EmbeddingStatistics {
  EmbeddingStatistics();
  ~EmbeddingStatistics();

  uint32_t num_unique_categories;

  // top categories sorted by count
  Tensor2<dtype> categories_sorted;
  Tensor2<uint32_t> counts_sorted;

  void calculate_statistics(Tensor2<dtype> samples);
  void get_top_categories();                       // from existing sorted categories
  void get_top_categories(  
    Tensor2<dtype> samples                 // in
    Tensor2<dtype> &categories_sorted_out, // out
    Tensor2<dtype> &counts_sorted_out      // out
  );

};


}
