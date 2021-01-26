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
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

// One FrequentEmbedding instance per gpu
template <typename dtype, typename emtype>
class FrequentEmbedding {
  // copy of the model parameters and the input data
  Model<dtype> model_;
  Data<dtype> data_;

  // locally stored embedding vectors for the data-parallel part of the embedding
  Tensor2<emtype> frequent_embedding_vectors_;
  // locally stored reduced gradients: input for the all-reduce
  Tensor2<emtype> frequent_partial_gradients_;

  Tensor2<uint32_t> frequent_sample_indices_;

  void init();

 public:
  FrequentEmbedding() {}
  ~FrequentEmbedding() {}

  void initialize_embedding_vectors();
  void forward_network(const emtype *interaction_layer_input);
  // update on the gpu where the sample gradients are calculated
  void update_network();
  // update on the gpu where the embedding vectors are stored
  void update_model();
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR