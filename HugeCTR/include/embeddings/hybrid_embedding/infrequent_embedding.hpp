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

#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_model.hpp"

namespace HugeCTR {


template <typename dtype, typename TypeEmbedding>
class InfrequentEmbedding {
  // copy of the model parameters and the input data, managed by HybridSparseEmbedding
  HybridEmbeddingModel<dtype> model_;
  HybridEmbeddingData<dtype> data_;

  Tensor2<TypeEmbedding> infrequent_embedding_vectors_;

  // forward-send, backward-receive
  Tensor2<uint32_t> model_indices_;
  Tensor2<uint32_t> model_indices_offsets_;
  // forward-receive, backward-send
  Tensor2<uint32_t> network_indices_;
  Tensor2<uint32_t> network_indices_offsets_;

 public:
  InfrequentEmbedding();
  ~InfrequentEmbedding();  

  void initialize_embedding_vectors();
  void calculate_model_indices(cudaStream_t stream);
  void calculate_network_indices(cudaStream_t stream);

  void all_to_all_forward();
  void all_to_all_backward();
};


}