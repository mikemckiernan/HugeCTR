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
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype, typename emtype>
class InfrequentEmbedding {
 public:
  // copy of the model parameters and the input data, managed by HybridSparseEmbedding
  Model<dtype> model_;
  Data<dtype> data_;

  // locally stored infrequent embedding vectors for the model-parallel part of the embedding for each table
  std::vector<Tensor2<float>> infrequent_embedding_vectors_;
  Tensor2<float> infrequent_embedding_vectors_block_; //  memory block for infrequent_embedding_vectors_

  // forward-send, backward-receive
  Tensor2<uint32_t> model_indices_;
  Tensor2<uint32_t> model_indices_offsets_;
  // forward-receive, backward-send
  Tensor2<uint32_t> network_indices_;
  Tensor2<uint32_t> network_indices_offsets_;

  std::shared_ptr<GPUResource> gpu_resource_;
  
  // to do, we need to initialize it in the constructor
  uint32_t embedding_vec_size_ = 128;
  // requires model_ and data_ to be set
  void init();

  InfrequentEmbedding(const std::shared_ptr<GPUResource> &gpu_resource)
      : gpu_resource_(gpu_resource) {}
  ~InfrequentEmbedding() {}

  void initialize_embedding_vectors();
  void forward_network(const emtype *message_buffer, emtype *interaction_layer_input, cudaStream_t stream);
  // only update on the gpu where the embedding vectors are stored
  void update_model();
  void calculate_model_indices(cudaStream_t stream);
  void calculate_network_indices(cudaStream_t stream);
};

}  // namespace hybrid_embedding

}  // namespace HugeCTR