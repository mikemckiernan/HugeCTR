#pragma once

#include <vector>
#include <cstdlib>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <sstream>
#include <mutex>

#include <cuda.h>
#include <cuda_runtime.h>

#include <common.hpp>

#ifdef ENABLE_PROFILING
#define PROFILE_RECORD(...) do               \
{                                            \
  global_profiler.record_event(__VA_ARGS__); \
} while (0)
#else
#define PROFILE_RECORD(...) do {} while (0)
#endif

namespace HugeCTR {
class Profiler {
  struct Event {
    std::string name;
    unsigned int start_index;
    unsigned int  end_index;
    // std::vector<unsigned int> on_iters;
    std::vector<float> measured_times;
  };

  struct GPUEvent : Event {
    unsigned int device_id;
    unsigned int stream_id;
  };

  struct CPUEvent : Event { };

  class GPUTimer {
   private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaStream_t stream_;

   public:
    GPUTimer(cudaStream_t);
    ~GPUTimer();
    // stream is a pointer itself
    void event_start();
    void event_stop();
    float get_result();
  };

  class CPUTimer {};

 private:
  std::string host_name_;
  unsigned int warmup_iterations_;
  unsigned int current_iteration_;
  unsigned int current_schedule_idx_;

  std::vector<std::pair<std::string, unsigned int>> scheduled_events_;

  std::map<cudaStream_t, std::shared_ptr<GPUTimer>> map_stream_gpu_timer_;
  std::map<cudaStream_t, unsigned int> map_stream_id_;

  std::vector<std::shared_ptr<Event>> events_;
  unsigned int events_num_;

  std::map<int, std::shared_ptr<GPUTimer>> map_event_id_current_gpu_timer_;

  std::mutex mtx_; // for thread safe

 public:
  void initialize(const char* schedule_file);
  void record_event(const char* event_label_char, cudaStream_t stream);
  void iter_start();
  void iter_end();
  int find_event(std::string& event_name, cudaStream_t stream);
  std::string write_result(const char* result_dir);

  static std::vector<std::string> split_string(std::string str, char delim = '.') {
    std::stringstream ss(str);
    std::vector<std::string> result;
    std::string token;
    while (std::getline(ss, token, delim)) {
        result.push_back(token);
    }
    return result;
  }

  static int get_device_id(cudaStream_t stream) {
    // TBD, below code seems problem
    // std::cout << "getting in !!" << std::endl;
    // CUcontext ctx;
    // std::cout << "1" << std::endl;
    // CUdevice device;
    // std::cout << "2" << std::endl;
    // CK_CU_RESULT_(cuStreamGetCtx(stream, &ctx));
    // std::cout << "3" << std::endl;
    // CK_CU_RESULT_(cuCtxPushCurrent(ctx));
    // std::cout << "4" << std::endl;
    // CK_CU_RESULT_(cuCtxGetDevice(&device));
    // std::cout << "5" << std::endl;
    // CK_CU_RESULT_(cuCtxPopCurrent(&ctx));
    // std::cout << "6" << std::endl;
    // int device_id;
    // //CUdevice_attribute attr = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
    // CUdevice_attribute attr = CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID;
    // std::cout << "7" << std::endl;
    // CK_CU_RESULT_(cuDeviceGetAttribute(&device_id, attr, device));
    // std::cout << "8" << std::endl;
    return 1;
  }
};

extern Profiler global_profiler;

}  // namespace HugeCTR
