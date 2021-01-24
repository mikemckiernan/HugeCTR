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

#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_utils.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

namespace HugeCTR {


template <typename dtype, typename TypeEmbedding>
void InfrequentEmbedding::initialize_embedding_vectors() {
  // TODO: create initialize_embedding_vectors()
}


template <typename dtype, typename TypeEmbedding>
void InfrequentEmbedding::calculate_model_indices() {
  // TODO: create calculate_model_indices()
}


template <typename dtype, typename TypeEmbedding>
void InfrequentEmbedding::calculate_network_indices() {
  // TODO: create calculate_network_indices()
}
  

template <typename dtype, typename TypeEmbedding>
void InfrequentEmbedding::all_to_all_forward() {
  // TODO: create all_to_all_forward()
}


template <typename dtype, typename TypeEmbedding>
void InfrequentEmbedding::all_to_all_backward() {
  // TODO: create all_to_all_backward()
}


#include "HugeCTR/include/embeddings/hybrid_embedding_template_defs/infrequent_embedding_template_defs.cuh"
}