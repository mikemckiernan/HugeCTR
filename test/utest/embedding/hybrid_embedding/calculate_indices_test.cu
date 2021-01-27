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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/utils.cuh"

// #include "HugeCTR/include/embeddings/hybrid_sparse_embedding.hpp"

#include <utest/test_utils.h>

#include "calculate_indices_test_data.hpp"

using namespace HugeCTR;
using namespace hybrid_embedding;

namespace {

//----------------------------------------------------------------------------
// global params for all testing

constexpr int num_nodes = 2;
constexpr int num_instances_per_node = 4;
constexpr int num_slots = 10;

// computed from the previous parameters
constexpr int num_instances = num_nodes * num_instances_per_node;

//----------------------------------------------------------------------------
// test

template <typename dtype, typename emtype = float>
void calculate_indices_test(const CalculateIndicesTestInputs<dtype>& inputs) {
  cudaStream_t stream;
  CK_CUDA_THROW_(cudaStreamCreate(&stream));
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  /* Create objects */
  std::vector<InfrequentEmbedding<dtype, emtype>> infrequent_embeddings;
  for(size_t i = 0; i < num_instances; i++) {
    infrequent_embeddings.emplace_back(test::get_default_gpu());
  }
  size_t local_batch_size = ceildiv<size_t>(inputs.batch_size, num_instances);
  for (size_t i = 0; i < num_instances; i++) {
    Model<dtype>& model = infrequent_embeddings[i].model_;
    Data<dtype>& data = infrequent_embeddings[i].data_;

    /// TODO: variable number of networks per node?
    model.global_instance_id = i;
    model.node_id = i / num_instances_per_node;
    model.instance_id = i % num_instances_per_node;
    model.num_frequent = inputs.num_frequent;
    model.num_categories = inputs.num_categories;
    model.num_instances = num_instances;

    data.batch_size = inputs.batch_size;
    data.local_table_sizes = inputs.table_sizes;
    data.num_networks = num_instances;

    std::vector<size_t> batch_dims = {inputs.batch_size * num_slots};
    std::vector<size_t> local_batch_dims = {local_batch_size * num_slots};
    std::vector<size_t> offsets_dims = {num_instances + 1};

    buff->reserve(batch_dims, &infrequent_embeddings[i].model_indices_);
    buff->reserve(offsets_dims, &infrequent_embeddings[i].model_indices_offsets_);
    buff->reserve(local_batch_dims, &infrequent_embeddings[i].network_indices_);
    buff->reserve(offsets_dims, &infrequent_embeddings[i].network_indices_offsets_);

    /// TODO: 2D tensors?
    std::vector<size_t> locations_dims = {2 * inputs.num_categories};
    std::vector<size_t> samples_dims = {inputs.batch_size * inputs.num_categories};
    buff->reserve(locations_dims, &model.category_location);
    buff->reserve(samples_dims, &data.samples);
  }
  buff->allocate();

  /* Upload data */
  for (size_t i = 0; i < num_instances; i++) {
    Model<dtype>& model = infrequent_embeddings[i].model_;
    Data<dtype>& data = infrequent_embeddings[i].data_;
    upload_tensor(inputs.category_location, model.category_location, stream);
    upload_tensor(inputs.samples, data.samples, stream);
  }

  std::vector<std::vector<uint32_t>> actual_model_indices =
      std::vector<std::vector<uint32_t>>(num_instances);
  std::vector<std::vector<uint32_t>> actual_model_indices_offsets =
      std::vector<std::vector<uint32_t>>(num_instances);
  std::vector<std::vector<uint32_t>> actual_network_indices =
      std::vector<std::vector<uint32_t>>(num_instances);
  std::vector<std::vector<uint32_t>> actual_network_indices_offsets =
      std::vector<std::vector<uint32_t>>(num_instances);

  /* Execute */
  for (size_t i = 0; i < num_instances; i++) {
    infrequent_embeddings[i].calculate_model_indices(stream);
    download_tensor(actual_model_indices[i], infrequent_embeddings[i].model_indices_, stream);
    download_tensor(actual_model_indices_offsets[i],
                    infrequent_embeddings[i].model_indices_offsets_, stream);
    infrequent_embeddings[i].calculate_network_indices(stream);
    download_tensor(actual_network_indices[i], infrequent_embeddings[i].network_indices_, stream);
    download_tensor(actual_network_indices_offsets[i],
                    infrequent_embeddings[i].network_indices_offsets_, stream);
  }
  for (size_t i = 0; i < num_instances; i++) {
    actual_model_indices[i].resize(actual_model_indices_offsets[i][num_instances]);
    actual_network_indices[i].resize(actual_network_indices_offsets[i][num_instances]);
  }

  /* Compare */
  for (size_t i = 0; i < num_instances; i++) {
    EXPECT_THAT(actual_model_indices[i],
                ::testing::ElementsAreArray(inputs.expected_model_indices[i]));
    EXPECT_THAT(actual_model_indices_offsets[i],
                ::testing::ElementsAreArray(inputs.expected_model_indices_offsets[i]));
    EXPECT_THAT(actual_network_indices[i],
                ::testing::ElementsAreArray(inputs.expected_network_indices[i]));
    EXPECT_THAT(actual_network_indices_offsets[i],
                ::testing::ElementsAreArray(inputs.expected_network_indices_offsets[i]));
  };

  /* Cleanup */
  CK_CUDA_THROW_(cudaStreamDestroy(stream));
}

}  // namespace

//----------------------------------------------------------------------------
// instantiation

TEST(calculate_indices_test, uint32_0) { calculate_indices_test<uint32_t>(inputs_uint32[0]); }
