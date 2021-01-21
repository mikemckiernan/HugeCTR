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

#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_utils.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
 
namespace HugeCTR {


template <typename dtype>
void HybridEmbeddingModel::init_model(
  const CalibrationData& calibration,
  const HybridEmbeddingData<dtype>& embedding_data
) {
    HybridEmbeddinStatistics statistics(embedding_data);
    calculate_num_frequent_categories(calibration, statistics);

}

#include "HugeCTR/include/embeddings/hybrid_embedding_includes/hybrid_embedding_model_includes.cuh"
}