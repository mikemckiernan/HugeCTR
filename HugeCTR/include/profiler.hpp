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
    std::string layer_name;
    std::string same_name_events_occured_order_in_code;
    int start_index;
    int end_index;
    // std::vector<int> on_iters;
    std::vector<float> measured_times_ms;
  };

  struct GPUEvent : Event {
    int device_id;
    cudaStream_t stream;
  };

  struct CPUEvent : Event { };

  class GPUTimer {
   private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    float measured_time_ms_;

   public:
    GPUTimer();
    ~GPUTimer();
    // stream is a pointer itself
    void event_start(cudaStream_t stream);
    void event_stop(cudaStream_t stream);
    float get_result();
  };

  class CPUTimer {};

 private:
  std::string host_name_;
  int warmup_iterations_;
  int current_iteration_;
  int current_schedule_idx_;

  std::vector<std::tuple<std::string, int, std::string, int>> scheduled_events_;
  std::map<cudaStream_t, std::shared_ptr<GPUTimer>> map_stream_to_gpu_timer_;
  std::vector<std::shared_ptr<Event>> events_;
  std::map<std::string, int> map_event_key_to_event_idx;
  int events_num_;

  // event_name : {stream : how many times does record_event meet this event_name within this stream }
  std::map<std::string, std::map<cudaStream_t, int>> map_internal_;
  // for thread safe
  std::mutex mtx_;

 public:
  void initialize(const char* schedule_file);
  void record_event(const char* event_label_char, cudaStream_t stream, int device_id);
  void iter_start();
  void iter_end();
  int find_event(std::string& event_key);
  int safe_access_map_internel(std::string& event_name, cudaStream_t stream);
  std::string write_result(const char* result_dir);

  static std::vector<std::string> split_string(std::string& str, char delim = '.') {
    std::stringstream ss(str);
    std::vector<std::string> result;
    std::string token;
    while (std::getline(ss, token, delim)) {
        result.push_back(token);
    }
    return result;
  }

  static std::string stream_str(cudaStream_t stream) {
    const void * address = static_cast<const void*>(stream);
    std::stringstream ss;
    ss << address;
    return ss.str();
  }

  static std::string gen_event_key(std::string& event_name, cudaStream_t stream, int same_name_events_occured_order_in_code) {
    return event_name + "_" + stream_str(stream) + "_" + std::to_string(same_name_events_occured_order_in_code);
  }



//   static int get_device_id() {
//     // TBD, below code seems problem
//     // std::cout << "get_device_id" << std::endl;
//     // CUcontext ctx;
//     // std::cout << "1" << std::endl;
//     // CUdevice device;
//     // std::cout << "2" << std::endl;
//     // CK_CU_RESULT_(cuStreamGetCtx(stream, &ctx));
//     // std::cout << "3" << std::endl;
//     // CK_CU_RESULT_(cuCtxPushCurrent(ctx));
//     // std::cout << "4" << std::endl;
//     // CK_CU_RESULT_(cuCtxGetDevice(&device));
//     // std::cout << "5" << std::endl;
//     // CK_CU_RESULT_(cuCtxPopCurrent(&ctx));
//     // std::cout << "6" << std::endl;
//     // int device_id;
//     // //CUdevice_attribute attr = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
//     // CUdevice_attribute attr = CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID;
//     // std::cout << "7" << std::endl;
//     // CK_CU_RESULT_(cuDeviceGetAttribute(&device_id, attr, device));
//     // std::cout << "8" << std::endl;
// 
//     //char device_info[20];
//     //CK_CU_RESULT_(cuDeviceGetAttribute(&device_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device));
//     //CK_CU_RESULT_(cuDeviceGetPCIBusId(device_info, 20, device));
//     //CK_CU_RESULT_(cuDeviceGetAttribute(&device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device));
//     // MESSAGE_(std::to_string(device_id));
// 
//     int device_id;
//     cudaGetDevice(&device_id);
//     return device_id;
//   }
};

extern Profiler global_profiler;

}  // namespace HugeCTR
