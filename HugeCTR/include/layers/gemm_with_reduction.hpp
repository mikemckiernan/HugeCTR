/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/gemm_complex.h"

// #include "testbed_utils.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace HugeCTR {

template <typename Gemm>
struct TestbedGemmWithReduction {

  using ElementAccumulator = typename Gemm::ElementAccumulator;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_D;
  cutlass::HostTensor<typename Gemm::ElementAccumulator, typename Gemm::LayoutC> tensor_Reduction;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_Tensor;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> reference_D;

  Gemm gemm_op_;

  int m_;
  int n_;
  int k_;

  //
  // Methods
  //

  TestbedGemmWithReduction(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }


  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view, 
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    } 
    else {
      // TODO: Implement the rest
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void copyin_val(const __half* W,
      const __half* dRelu_top,
      __half* dRelu_bottom,
      __half* db,
      const __half* mask,
      cudaStream_t stream = 0) {
    int len_A = tensor_A.size();
    int len_B = tensor_B.size();
    int len_D = tensor_D.size();
    int len_T = tensor_Tensor.size();

    cutlass::half_t* W_ptr            = tensor_A.device_data();
    cutlass::half_t* dRelu_top_ptr    = tensor_B.device_data();
    cutlass::half_t* dRelu_bottom_ptr = tensor_D.device_data();
    cutlass::half_t* mask_ptr         = tensor_Tensor.device_data();
    
    cudaMemcpyAsync(W_ptr,            W,            len_A*sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dRelu_top_ptr,    dRelu_top,    len_B*sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dRelu_bottom_ptr, dRelu_bottom, len_D*sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(mask_ptr,         mask,         len_T*sizeof(__half), cudaMemcpyDeviceToDevice, stream);
    // cudaMemset(mask_ptr, 1, len_T*sizeof(__half));

    // tensor_A.sync_host();
    // tensor_B.sync_host();
    // tensor_D.sync_host();
    // tensor_Tensor.sync_host();
  }

  /// Initializes data structures
  void initialize(
    const __half* W,
    const __half* dRelu_top,
    __half* dRelu_bottom,
    float* db,
    const __half* mask,
    cutlass::gemm::GemmUniversalMode mode,
    cutlass::gemm::GemmCoord problem_size, 
    int batch_count = 1,
    ElementAccumulator alpha = ElementAccumulator(1), 
    ElementAccumulator beta = ElementAccumulator(0),
    cudaStream_t stream=0) {
    //
    // Allocate the GEMM workspace
    //

    this->m_ = problem_size.m();
    this->n_ = problem_size.n();
    this->k_ = problem_size.k();

    tensor_A.resize(problem_size.mk());
    tensor_B.resize(problem_size.kn());
    tensor_C.resize(problem_size.mn());
    tensor_D.resize(problem_size.mn());

    tensor_Reduction.resize({
      problem_size.m(), 
      (problem_size.n() - 1 + Gemm::ThreadblockShape::kN) / Gemm::ThreadblockShape::kN
    });

    tensor_Tensor.resize(problem_size.mn());
    reference_D.resize(problem_size.mn(), false);

    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments{
      mode,
      problem_size,
      batch_count,
      {alpha, beta},
      W,
      dRelu_top,
      dRelu_bottom,
      dRelu_bottom,
      db,
      mask,
      problem_size.m() * problem_size.k(),
      problem_size.n() * problem_size.k(),
      problem_size.m() * problem_size.n(),
      problem_size.m() * problem_size.n(),
      tensor_A.layout().stride(0),
      tensor_B.layout().stride(0),
      tensor_C.layout().stride(0),
      tensor_D.layout().stride(0),
      tensor_Reduction.layout().stride(0),
      tensor_Tensor.layout().stride(0),
    };


    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    gemm_op_.initialize(arguments, workspace.get(), stream);

    // std::cout<<"A: ("<<tensor_A.size(0)<<","<<tensor_A.size(1)<<")"<<std::endl;
    // std::cout<<"tensor_A: "<<tensor_A.size()<<std::endl;
    // std::cout<<"tensor_B: "<<tensor_B.size()<<std::endl;
    // std::cout<<"tensor_C: "<<tensor_C.size()<<std::endl;
    // std::cout<<"tensor_D: "<<tensor_D.size()<<std::endl;
    // std::cout<<"tensor_Reduction: "<<tensor_Reduction.size()<<std::endl;
    // std::cout<<"tensor_Tensor: "<<tensor_Tensor.size()<<std::endl;

    // initialize_tensor(tensor_A.host_view(), init_A, seed + 2019);
    // initialize_tensor(tensor_B.host_view(), init_B, seed + 2018);
    // initialize_tensor(tensor_C.host_view(), init_C, seed + 2017);
    // initialize_tensor(tensor_Tensor.host_view(), init_C, seed + 2020);

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    // tensor_A.host_view().at({0, 0}) = typename Gemm::ElementA(1);
    // tensor_B.host_view().at({0, 0}) = typename Gemm::ElementB(1);
    // tensor_C.host_view().at({0, 0}) = typename Gemm::ElementC(1);

    // cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    // tensor_A.sync_device();
    // tensor_B.sync_device();
    // tensor_C.sync_device();
    // tensor_D.sync_device();
    // tensor_Reduction.sync_device();
    // tensor_Tensor.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
    cutlass::gemm::GemmCoord problem_size, 
    ElementAccumulator alpha, 
    ElementAccumulator beta) {

    tensor_Reduction.sync_host();
    tensor_D.sync_host();

    assert(cutlass::reference::host::TensorNorm(tensor_A.host_view()) > 0);
    assert(cutlass::reference::host::TensorNorm(tensor_B.host_view()) > 0);
    assert(cutlass::reference::host::TensorNorm(tensor_C.host_view()) > 0);
    
    assert(cutlass::reference::host::TensorNorm(tensor_D.host_view()) > 0);
    assert(cutlass::reference::host::TensorNorm(reference_D.host_view()) > 0);
    assert(cutlass::reference::host::TensorNorm(tensor_Reduction.host_view()) > 0);

    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    if (!passed) {

      /*
      std::stringstream fname;

      fname << "error_Gemm_device_"
        << problem_size.m() << "x"
        << problem_size.n() << "x"
        << problem_size.k() << "_"
        << Gemm::ThreadblockShape::kM << "x"  
        << Gemm::ThreadblockShape::kN << "x"  
        << Gemm::ThreadblockShape::kK << "_"
        << Gemm::WarpShape::kM << "x"  
        << Gemm::WarpShape::kN << "x"  
        << Gemm::WarpShape::kK << ".txt";

      std::ofstream file(fname.str());
      */

      std::ofstream file("testbed_universal_errors.txt");

      file
        << "problem: " << problem_size 
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      file 
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\nT = \n" << tensor_Tensor.host_view()
        << "\n\nReference =\n" << reference_D.host_view()
        << "\nComputed =\n" << tensor_D.host_view();
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
    cutlass::gemm::GemmCoord problem_size, 
    ElementAccumulator alpha, 
    ElementAccumulator beta) {

    //
    // Verify
    //

    cutlass::reference::host::GemmComplex<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC, 
        ElementAccumulator, ElementAccumulator
    >(
      problem_size,
      alpha, 
      tensor_A.host_ref(),
      Gemm::kTransformA,
      tensor_B.host_ref(),
      Gemm::kTransformB,
      beta, 
      tensor_C.host_ref(), 
      reference_D.host_ref(), 
      ElementAccumulator(0)
    );

    using ElementC = typename Gemm::ElementC;

    // compute relu
    for (int m = 0; m < problem_size.m(); ++m) {
      for (int n = 0; n < problem_size.n(); ++n) {
        if (tensor_Tensor.at({m, n}) < ElementC()) {
          reference_D.at({m, n}) = ElementC();
        }
      }
    }

    return compare_reference(problem_size, alpha, beta);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool sufficient() const {

    //
    // Determine SMEM requirements and waive if not satisfied
    //

    unsigned smem_size = unsigned(sizeof(typename Gemm::GemmKernel::SharedStorage));

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerMultiprocessor < smem_size) {
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(float* db, cudaStream_t stream=0) {

    // this->initialize(problem_size);
    // this->copyin_val(W, dRelu_top, dRelu_bottom, db, mask, stream);

    //
    // Run the GEMM
    //
    this->gemm_op_(stream);
    //
    // Verify
    //
    // int len = tensor_Reduction.size();
    // for (int i = 0; i < len; i++)
    // {
    //   /* code */
    // }
//     int len = tensor_Reduction.size();
//     float* db_host = new float[len];
//     cudaMemcpyAsync(db_host, db, len*sizeof(float), cudaMemcpyDeviceToHost, stream);
//     cutlass::half_t* sum = new cutlass::half_t[this->m_];
//     int num = (this->n_ - 1 + Gemm::ThreadblockShape::kN) / Gemm::ThreadblockShape::kN;
//     for (int i = 0; i < this->m_; i++)
//     {
//       float temp = 0.0f;
//       for(int j=0;j<num;j++)
//       {
//         temp += db_host[i+j*this->m_];
//       }
// //      sum[i] = (cutlass::half_t)temp;
//       sum[i] = (cutlass::half_t)db_host[i];;
//     }
//     cudaMemcpyAsync(db, sum, this->m_*sizeof(cutlass::half_t), cudaMemcpyHostToDevice, stream);
    
    

    // bool passed = this->verify(problem_size, alpha, beta);

    // if (!passed) {
    //   std::cout << "Failed with batch_count/split_k_slices = " << batch_count << std::endl;
    // } else
    // {
    //     std::cout << "Pass" << std::endl;
    // }

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// template <typename Gemm>
// bool cutlassGemmWithReduction(
//   const __half* W,
//   const __half* dRelu_top,
//   __half* dRelu_bottom,
//   __half* db,
//   const __half* mask,
//   cutlass::gemm::GemmCoord const & problem_size,
//   cutlass::gemm::GemmUniversalMode mode,  
//   int batch_count,
//   double alpha = 1.0, 
//   double beta = 2.0) {

//   bool passed = true;

//   TestbedGemmWithReduction<Gemm> testbed;

//   using ElementAccumulator = typename Gemm::ElementAccumulator;
  

//   passed = testbed.run(W, dRelu_top, dRelu_bottom, db, mask,
//     mode,
//     problem_size, 
//     batch_count,
//     cutlass::from_real<ElementAccumulator>(alpha), 
//     cutlass::from_real<ElementAccumulator>(beta)
//   );

//   return passed;
// }

}  // namespace HugeCTR
