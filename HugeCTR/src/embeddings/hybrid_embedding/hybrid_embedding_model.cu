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


/// init_model calculates the optimal number of frequent categories 
/// given the calibration of the all-to-all and all-reduce.
template<dtype>
void HybridEmbeddingModel::init_model(
    const CalibrationData& calibration,
    const HybridEmbeddingData<dtype>& embedding_data
) {

  size_t num_nodes = (double) num_networks_per_node.size();
  size_t num_networks = (size_t) 0;
  for (size_t i = 0; i < num_networks_per_node.size(); ++i) {
    num_networks += num_networks_per_nodex[i];
  }

  size_t batch_size = embedding_data.batch_size;
  size_t num_networks = embedding_data.num_networks;
  size_t num_iterations = embedding_data.num_iterations;
  size_t num_tables = embedding_data.table_sizes.size();

  Tensor2<dtype> samples = embedding_data.samples;
  EmbeddingStatistics statistics(samples.get_size_in_bytes() / sizeof(dtype));
  statistics.calculate_statistics(samples);

  if (calibration.all_to_all_times.size() > 0) {
    // calibration is given, perform fully optimized hybrid model
    CK_THROW(Error_t::WrongInput, "initialization hybrid model from communication calibration not available yet");
  } else {

    // Use threshold to determine number of frequent categories,
    // calculates optimal number of frequent categories when the all-to-all 
    // and all-reduce are both bandwidth limited.
    float count_threshold = 1.f;
    double count_threshold = calibration.calculate_threshold(
        communication_type, batch_size, num_networks, num_iterations, num_tables);

    // Initialize frequent_category_index

    // Get first index in category frequent array that is smaller than threshold
    // find first element smaller than n_threshold
    std::vector<uint32_t> h_counts_sorted;
    download_tensor(h_counts_sorted, statistics.counts_sorted, stream);

    // num_frequent must be smaller than batch_size
    size_t n;
    for (n = 0; n < h_counts_sorted.size(); ++n) {
        if (h_counts_sorted[n] < count_threshold) break;
    }
    num_frequent = n;

    std::vector<dtype> h_categories_sorted;
    download_tensor(h_categories_sorted, statistics.categories_sorted, stream);

    std::vector<dtype> h_category_frequent_index(num_categories, num_categories);
    for (size_t i = 0; i < num_frequent; ++i) {
      dtype category = h_categories_sorted[i]
      h_category_frequent_index[category] = (dtype) i;
    }

    // 
    // initialize category_location
    // node_id, gpu_id, buffer_index
    std::vector<dtype> category_location(2*num_categories, num_categories);
    size_t indx = 0;
    for (size_t category = 0; category < num_categories; ++category) {
      if (h_category_frequent_index[category] == num_categories) {
        h_category_location[2*category] = indx % num_networks;
        h_category_location[2*category+1] = indx / num_networks;
      }
    }

    upload_tensor(h_category_frequent_index, category_frequent_index, stream);
    upload_tensor(h_category_location, category_location, stream);
  }
}


#include "HugeCTR/include/embeddings/hybrid_embedding_includes/hybrid_embedding_model_includes.cuh"
}