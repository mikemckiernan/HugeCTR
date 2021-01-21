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
void EmbeddingStatistics<uint32_t>::calculate_statistics(
  Tensor2<uint32_t> samples, 
  cudaStream_t stream);

template
void sort_categories_by_count<uint32_t>(
    uint32_t *samples,
    uint32_t num_samples,
    uint32_t *categories_sorted,
    uint32_t *counts_sorted,
    uint32_t &num_unique_categories,
    cudaStream_t stream);
