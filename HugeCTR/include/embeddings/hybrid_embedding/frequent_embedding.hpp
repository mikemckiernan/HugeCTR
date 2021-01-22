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

namespace HugeCTR {


// One FrequentEmbedding instance per gpu
template <typename dtype, typename TypeEmbedding>
class FrequentEmbedding {
  // copy of the model parameters and the input data
  HybridEmbeddingModel<dtype> model_;
  HybridEmbeddingData<dtype> data_;

  // locally stored embedding vectors for the data-parallel part of the embedding
  Tensor2<TypeEmbedding> frequent_embedding_vectors_;
  // locally stored reduced gradients: input for the all-reduce
  Tensor2<TypeEmbedding> frequent_partial_gradients_;

  void init();
public:
  FrequentEmbedding() {}
  ~FrequentEmbedding() {}

  void initialize_embedding_vectors();

  void reduce();

  // when using IB & NVLink
  void all_reduce();
  // for NVLink only
  void all_to_all_reduce();

  void update();
};


}