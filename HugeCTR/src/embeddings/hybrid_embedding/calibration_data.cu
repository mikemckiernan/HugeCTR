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

#include <algorithm>
#include <iostream>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/calibration_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

///
/// interpolate data_size using the two calibration data
///   calibrated_data_size, calibrated_times
///   return communication_times
///
void CalibrationData::interpolate(const Tensor2<float> &calibrated_data_size,
                                  const Tensor2<float> &calibrated_times,
                                  const Tensor2<float> &data_size,
                                  Tensor2<float> &communication_times) {
  // TODO: implement this
}

///
/// Convenience function for interpolating all-to-all communication times from
/// calibrated data
///
void CalibrationData::interpolate_all_reduce(const Tensor2<float> &data_size,
                                             Tensor2<float> &communication_times) {
  interpolate(all_reduce_data_size, all_reduce_times, data_size, communication_times);
}

///
/// Convenience function for interpolating all-to-all communication times from
/// calibrated data
///
void CalibrationData::interpolate_all_to_all(const Tensor2<float> &data_size,
                                             Tensor2<float> &communication_times) {
  interpolate(all_to_all_data_size, all_to_all_times, data_size, communication_times);
}

// Calculate threshold such that for the worst case distribution there will
// be one duplication per network on average
template <typename dtype>
double ModelInitializationFunctors<dtype>::calculate_threshold(
    const CommunicationType communication_type, double all_to_all_bandwidth,
    double all_reduce_bandwidth, size_t num_nodes, size_t batch_size, size_t num_networks,
    size_t num_iterations, size_t num_tables) {
  float count_threshold = 1.f;

  // for NVLink capture effectively all duplications with number of categories
  double M = (double)batch_size / (double)num_networks;
  double p_dup_max = 1.0 / 100.;  // maximum 1 % of samples the category will be duplicated
  switch (communication_type) {
    case CommunicationType::IB_NVLink:
      count_threshold = (double)num_iterations * (double)num_nodes * all_to_all_bandwidth /
                        all_reduce_bandwidth * (double)num_nodes / ((double)num_nodes - 1.);
      break;
    case CommunicationType::NVLink:
      // count threshold such that the probability of duplication is less than p_dup_max
      //   even if there are batch size number of categories that occur more often,
      //   there will be a duplication at most once every iteration per gpu
      //
      // d_duplication(category) \approx 1/2 M (M-1) \left( \frac{count}{batch_sizexnum_iterations}
      // \right)^2
      count_threshold = (double)((double)batch_size * (double)num_iterations *
                                 sqrt(2.0 * p_dup_max / (M * (M - 1))));
      break;
    default:
      CK_THROW_(Error_t::WrongInput, "Unknown communication type, expecting IB_NVLink or NVLink");
  }

  return count_threshold;
}

///
/// Calculate the number of frequent categories from data
///
template <typename dtype>
uint32_t ModelInitializationFunctors<dtype>::calculate_num_frequent_categories(
    const CommunicationType &communication_type, const CalibrationData &calibration,
    const Statistics<dtype> &statistics, const Data<dtype> &data, cudaStream_t stream) {
  size_t num_frequent = 0;

  if (calibration.all_to_all_times.get_size_in_bytes() > 0) {
    // calibration is given, perform fully optimized hybrid model
    CK_THROW_(Error_t::WrongInput,
              "initialization hybrid model from communication calibration not available yet");
  } else {
    size_t num_nodes = calibration.num_nodes;
    size_t batch_size = data.batch_size;
    size_t num_networks = data.num_networks;
    size_t num_iterations = data.num_iterations;
    size_t num_tables = data.num_tables;

    // Use threshold to determine number of frequent categories,
    // calculates optimal number of frequent categories when the all-to-all
    // and all-reduce are both bandwidth limited.
    double count_threshold = ModelInitializationFunctors::calculate_threshold(
        communication_type, calibration.max_all_to_all_bandwidth,
        calibration.max_all_reduce_bandwidth, num_nodes, batch_size, num_networks, num_iterations,
        num_tables);

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

template class ModelInitializationFunctors<uint32_t>;
template class ModelInitializationFunctors<unsigned long>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR