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

#pragma once

#include <layer.hpp>

namespace HugeCTR {

/**
 * Adaptor Start activation function as a derived class of Layer
 */
template <typename T>
class AdaptorEndLayer : public Layer {
  /*
   * stores the references to the input tensors of this layer.
   */
  Tensors2<T> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  Tensors2<T> out_tensors_;

 public:
  /**
   * Ctor of ReluLayer.
   * @param fprop_in_tensor the input tensor
   * @param bprop_in_tensor the output tensor for forward propogation
   * @param out_tensor the output tensor for backward propogation
   * @param device_id the id of GPU where this layer belongs
   */
  AdaptorEndLayer(const Tensor2<T>& fprop_in_tensor,
          const Tensor2<T>& bprop_in_tensor,
          Tensor2<T>& out_tensor,
          const std::shared_ptr<GPUResource>& gpu_resource);

  /**
   * A method of implementing the forward pass of Adaptor Start
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(bool is_train) override;
  /**
   * A method of implementing the backward pass of Adaptor Start
   * @param stream CUDA stream where the backward propagation is executed
   */
  void bprop() override;
};

}  // namespace HugeCTR
