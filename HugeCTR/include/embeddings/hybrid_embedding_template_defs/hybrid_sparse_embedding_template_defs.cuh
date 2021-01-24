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

template
void HybridSparseEmbedding<uint32_t, __half>::initialize_model();
template
void HybridSparseEmbedding<uint32_t, float>::initialize_model();

template
void HybridSparseEmbedding<uint32_t, __half>::forward(bool is_train);
template
void HybridSparseEmbedding<uint32_t, float>::forward(bool is_train);

template
void HybridSparseEmbedding<uint32_t, __half>::backward();
template
void HybridSparseEmbedding<uint32_t, float>::backward();

template 
void HybridSparseEmbedding<uint32_t, __half>::update_params();
template
void HybridSparseEmbedding<uint32_t, float>::update_params();

template
void HybridSparseEmbedding<uint32_t, __half>::init_params();
template
void HybridSparseEmbedding<uint32_t, float>::init_params();

template
void HybridSparseEmbedding<uint32_t, __half>::load_parameters(std::istream& stream);
template
void HybridSparseEmbedding<uint32_t, float>::load_parameters(std::istream& stream);

template
void HybridSparseEmbedding<uint32_t, __half>::dump_parameters(std::ostream& stream) const;
template
void HybridSparseEmbedding<uint32_t, float>::dump_parameters(std::ostream& stream) const;

template
void HybridSparseEmbedding<uint32_t, __half>::set_learning_rate(float lr);
template
void HybridSparseEmbedding<uint32_t, float>::set_learning_rate(float lr);

template
size_t HybridSparseEmbedding<uint32_t, __half>::get_params_num() const;
template
size_t HybridSparseEmbedding<uint32_t, float>::get_params_num() const;

template
size_t HybridSparseEmbedding<uint32_t, __half>::get_vocabulary_size() const;
template
size_t HybridSparseEmbedding<uint32_t, float>::get_vocabulary_size() const;

template
size_t HybridSparseEmbedding<uint32_t, __half>::get_max_vocabulary_size() const;
template
size_t HybridSparseEmbedding<uint32_t, float>::get_max_vocabulary_size() const;

template
std::vector<TensorBag2> HybridSparseEmbedding<uint32_t, __half>::get_train_output_tensors() const;
template
std::vector<TensorBag2> HybridSparseEmbedding<uint32_t, float>::get_train_output_tensors() const;
 
template
std::vector<TensorBag2> HybridSparseEmbedding<uint32_t, __half>::get_evaluate_output_tensors() const;
template
std::vector<TensorBag2> HybridSparseEmbedding<uint32_t, float>::get_evaluate_output_tensors() const;
