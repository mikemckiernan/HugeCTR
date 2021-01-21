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

#include "HugeCTR/include/embeddings/hybrid_sparse_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/"


namespace HugeCTR {


template <typename dtype, typename TypeEmbedding>
void FrequentEmbedding::reduce() {
  switch(model_.communication_type) {
    case IB_NVLink:
      // internode using IB and intra node using NVLink: perform all-reduce
      all_reduce();
    break;
    case NVLink:
      // internode and intranode direct access: update fequent category
      // embedding vector on one gpu, 
      // on all gpus: update the embedding cache for the embedding vectors 
      // that are needed in next iteration.
      all_to_all_reduce();
    break;
    default:
      CK_THROW(Errot_t::WrongInput, "Not a valid communication type, should be IB_NVLink or NVLink");
  }
}


#include "HugeCTR/include/embeddings/hybrid_embedding_includes/hybrid_sparse_embedding_includes.cuh"
}