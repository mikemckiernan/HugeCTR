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
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/calibration_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"


namespace HugeCTR{


namespace hybrid_embedding {


template <typename dtype, typename emtype>
class Communication {
  FrequentEmbedding<dtype, emtype> frequent_embedding_;
  InfrequentEmbedding<dtype, emtype> infrequent_embedding_;

  void initialize_communication();

public:
  Communication() {}
  ~Communication() {}

  // reduces the frequent embedding
  void all_reduce();
  // all-to-all forward and backward for the infrequent embedding
  void all_to_all_v_forward();
  void all_to_all_v_backward();

  // performs all-to-all-reduce on the frequent embedding
  void all_to_all_reduce_frequent();
  // performs all-to-all-reduce on the frequent and infrequent
  // embeddings simultaneously (IB)
  void all_to_all_reduce();
};


}


}