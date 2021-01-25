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

#include <cuda_runtime.h>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_sparse_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/calibration_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/tensor2.hpp"


namespace HugeCTR {


template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::initialize_model() {
  // TODO: create initialize_model()
  //
  // allocate memory and initialize hybrid model objects...
  // 
}


template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::forward(bool is_train) {
  // TODO: create forward()
}
  

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::backward() {
  // TODO: create backward()
}


template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::update_params() {
  // TODO: create update_params()
}


template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::init_params() {
  // TODO: createe init_params()
}


template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::load_parameters(std::istream& stream) {
  // TODO: create load_parameters()
}


template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::dump_parameters(std::ostream& stream) const {
  // TODO: create dump_parameters()
}


template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::set_learning_rate(float lr) {
  // TODO: create set_learning_rate()
}


template <typename dtype, typename emtype>
size_t HybridSparseEmbedding<dtype, emtype>::get_params_num() const {

}


template <typename dtype, typename emtype>
size_t HybridSparseEmbedding<dtype, emtype>::get_vocabulary_size() const {
  // TODO: create get_vocabulary_size()
}


template <typename dtype, typename emtype>
size_t HybridSparseEmbedding<dtype, emtype>::get_max_vocabulary_size() const {
  // TODO: create get_max_vocabulary_size()
}


template <typename dtype, typename emtype>
std::vector<TensorBag2> HybridSparseEmbedding<dtype, emtype>::get_train_output_tensors() const {
  // TODO: create get_train_output_tensors()
}


template <typename dtype, typename emtype>
std::vector<TensorBag2> HybridSparseEmbedding<dtype, emtype>::get_evaluate_output_tensors() const {
  // TODO: create get_evaluate_output_tensors
}


template class HybridSparseEmbedding<uint32_t, __half>;
template class HybridSparseEmbedding<size_t, float>;
template class HybridSparseEmbedding<uint32_t, __half>;
template class HybridSparseEmbedding<size_t, float>;
}
