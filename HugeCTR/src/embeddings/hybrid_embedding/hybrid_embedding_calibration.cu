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

#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_calibration.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_utils.hpp"
  
#include <algorithm>
#include <iostream>
#include <vector>


namespace HugeCTR {


///
/// interpolate data_size using the two calibration data 
///   calibrated_data_size, calibrated_times
///   return communication_times
///
float CalibrationData::interpolate(
    const Tensor2<float> &calibrated_data_size,
    const Tensor2<float> &calibrated_times,
    const Tensor2<float> &data_size,
    Tensor2<float> &communication_times
) {
  // TODO: implement this
}


///
/// Convenience function for interpolating all-to-all communication times from 
/// calibrated data
///
float CalibrationData::interpolate_all_reduce(
  const Tensor2<float> &data_size,
  Tensor2<float> &communication_times
) {
  interpolate(
    all_reduce_data_size, all_reduce_times, data_size, communication_times
  );
}


///
/// Convenience function for interpolating all-to-all communication times from 
/// calibrated data
///
float CalibrationData::interpolate_all_to_all(
  const Tensor2<float> &data_size,
  Tensor2<float> &communication_times
) {
  interpolate(
    all_to_all_data_size, all_to_all_times, data_size, communication_times
  );
}


/// 
/// Calculate the number of frequent categories from data
///
template <typename dtype>
uint32_t calculate_num_frequent_categories(
  CommunicationType communication_type,
  CalibrationData<dtype> calibration,
  HybridEmbeddingStatistics<dtype> statistics,
  HybridEmbeddingData<dtype> data,
  cudaStream_t stream
) {
  size_t num_frequent = 0;

  if (calibration.all_to_all_times.size() > 0) {
    // calibration is given, perform fully optimized hybrid model
    CK_THROW(Error_t::WrongInput, "initialization hybrid model from communication calibration not available yet");
  } else {
    float count_threshold = 1.f;

    size_t num_networks = (size_t) 0;
    for (size_t i = 0; i < num_networks_per_node.size(); ++i) {
      num_networks += num_networks_per_nodex[i];
    }

    size_t batch_size = data.batch_size;
    size_t num_networks = data.num_networks;
    size_t num_iterations = data.num_iterations;
    size_t num_tables = data.table_sizes.size();

    // Use threshold to determine number of frequent categories,
    // calculates optimal number of frequent categories when the all-to-all 
    // and all-reduce are both bandwidth limited.
    double count_threshold = calibration.calculate_threshold(
        communication_type, batch_size, num_networks, num_iterations, num_tables);

    /// === TODO: PERFORM ON GPU ===
    /// ============================
    std::vector<dtype> h_categories_sorted;
    std::vector<uint32_t> h_counts_sorted;
    download_tensor(h_categories_sorted, statistics.categories_sorted, stream);
    download_tensor(h_counts_sorted, statistics.counts_sorted, stream);

    size_t num_top_categories = statistics.categories_sorted.get_size_in_bytes() / sizeof(dtype);
    for (num_frequent = 0; num_frequent < num_top_categories; ++num_frequent) {
      if (h_counts_sorted[num_frequent] < count_threshold) break;
    }
    /// ============================
    /// ============================
  }

  return num_frequent;
}


// Calculate threshold such that for the worst case distribution there will 
// be one duplication per network on average
float CalibrationData::calculate_threshold(
  CommunicationType communication_type,
  size_t batch_size, 
  size_t num_networks,
  size_t num_iterations,
  size_t num_tables
) {
  float count_threshold = 1.f;

  // for NVLink capture effectively all duplications with number of categories
  double M = (double) batch_size / (double) num_networks;
  double p_dup_max = 1.0 / 100.; // maximum 1 % of samples the category will be duplicated
  double count_threshold = 1.;
  switch(communication_type) {
  case IB_NVLink:
    count_threshold = (double) num_nodes * all_to_all_bandwidth / all_reduce_bandwidth 
                    * (double) num_nodes / ((double) num_nodes - 1.);
    break;
  case NVLink:
    // count threshold such that the probability of duplication is less than p_dup_max
    //   even if there are batch size number of categories that occur more often,
    //   there will be a duplication at most once every iteration per gpu
    //
    // d_duplication(category) \approx 1/2 M (M-1) \left( \frac{count}{batch_sizexnum_iterations} \right)^2
    count_threshold = (float) ((double) batch_size * (double) num_iterations
                    * sqrt(2.0 * p_dup_max / (M *(M-1))));
    break;
  default:
    CK_THROW(Error_t::WrongInput, "Unknown communication type, expecting IB_NVLink or NVLink");
  }

  return count_threshold;
}


#include "HugeCTR/include/embeddings/hybrid_embedding_template_defs/hybrid_embedding_calibration_template_defs.cuh"
}
