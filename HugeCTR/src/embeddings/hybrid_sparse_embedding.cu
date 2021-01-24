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

#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_sparse_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_utils.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_calibration.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_statistics.hpp"


namespace HugeCTR {


template <typename dtype, typename TypeEmbedding>
void HybridSparseEmbedding<dtype, TypeEmbedding>::initialize_model() {
  // TODO: create initialize_model()
  //
  // allocate memory and initialize hybrid model objects...
  // 
}


template <typename dtype, typename TypeEmbedding>
void HybridSparseEmbedding<dtype, TypeEmbedding>::forward(bool is_train) {
  // TODO: create forward()
}
  

template <typename dtype, typename TypeEmbedding>
void HybridSparseEmbedding<dtype, TypeEmbedding>::backward() {
  // TODO: create backward()
}


template <typename dtype, typename TypeEmbedding>
void HybridSparseEmbedding<dtype, TypeEmbedding>::update_params() {
  // TODO: create update_params()
}


template <typename dtype, typename TypeEmbedding>
void HybridSparseEmbedding<dtype, TypeEmbedding>::init_params() {
  // TODO: createe init_params()
}


template <typename dtype, typename TypeEmbedding>
void HybridSparseEmbedding<dtype, TypeEmbedding>::load_parameters(std::istream& stream) {
  // TODO: create load_parameters()
}


template <typename dtype, typename TypeEmbedding>
void HybridSparseEmbedding<dtype, TypeEmbedding>::dump_parameters(std::ostream& stream) const {
  // TODO: create dump_parameters()
}


template <typename dtype, typename TypeEmbedding>
void HybridSparseEmbedding<dtype, TypeEmbedding>::set_learning_rate(float lr) {
  // TODO: create set_learning_rate()
}


template <typename dtype, typename TypeEmbedding>
size_t HybridSparseEmbedding<dtype, TypeEmbedding>::get_params_num() const {

}


template <typename dtype, typename TypeEmbedding>
size_t HybridSparseEmbedding<dtype, TypeEmbedding>::get_vocabulary_size() const {
  // TODO: create get_vocabulary_size()
}


template <typename dtype, typename TypeEmbedding>
size_t HybridSparseEmbedding<dtype, TypeEmbedding>::get_max_vocabulary_size() const {
  // TODO: create get_max_vocabulary_size()
}


template <typename dtype, typename TypeEmbedding>
std::vector<TensorBag2> HybridSparseEmbedding<dtype, TypeEmbedding>::get_train_output_tensors() const {
  // TODO: create get_train_output_tensors()
}


template <typename dtype, typename TypeEmbedding>
std::vector<TensorBag2> HybridSparseEmbedding<dtype, TypeEmbedding>::get_evaluate_output_tensors() const {
  // TODO: create get_evaluate_output_tensors
}


#include "HugeCTR/include/embeddings/hybrid_embedding_template_defs/hybrid_sparse_embedding_template_defs.cuh"
}
