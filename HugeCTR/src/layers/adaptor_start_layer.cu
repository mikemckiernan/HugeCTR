/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <algorithm>
#include <functional>
#include <include/utils.cuh>
#include <layers/element_wise_function.hpp>
#include <layers/adaptor_start_layer.hpp>
#include <linalg/binary_op.cuh>
#include <linalg/unary_op.cuh>
#include <utils.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

namespace HugeCTR {

template <typename T>
AdaptorStartLayer<T>::AdaptorStartLayer(const Tensor2<T>& in_tensor,
        Tensor2<T>& fprop_out_tensors,
        Tensor2<T>& bprop_out_tensors,
        const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(gpu_resource) {
  // assert(in_tensor.get_num_elements() == out_tensor.get_num_elements());
  assert(in_tensor.get_num_elements() % 2 == 0);

  fprop_out_tensors = in_tensor;
  bprop_out_tensors = in_tensor;
  in_tensors_.push_back(in_tensor);
  out_tensors_.push_back(fprop_out_tensors);
  out_tensors_.push_back(bprop_out_tensors);
}

template <typename T>
void AdaptorStartLayer<T>::fprop(bool is_train) {
}

template <typename T>
void AdaptorStartLayer<T>::bprop() {
}

template class AdaptorStartLayer<float>;
template class AdaptorStartLayer<__half>;

}  // namespace HugeCTR
