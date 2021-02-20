#pragma once

#include <vector>
#include <cstdlib>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <sstream>
#include <chrono>
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

#define PROFILER_DEBUG_(msg)                                                              \
  do {                                                                                    \
    MESSAGE_(std::string(msg) + " on thread " + std::to_string(omp_get_thread_num()) +    \
                           ", on stream " + stream_str(stream) +                          \
                           ", on device " + std::to_string(device_id) +                   \
                           ", iter " + std::to_string(current_iteration_));               \
  } while (0)                                                                             \

namespace HugeCTR {
class Profiler {
  struct Event {
    std::string event_name;
    int start_index;
    int end_index;
    // std::vector<int> on_iters;
    std::vector<float> iter_start_to_event_start_times_ms;
    std::vector<float> measured_times_ms;
  };

  struct GPUEvent : Event {
    int device_id;
    int met_times_within_this_stream;
    cudaStream_t stream;
  };

  struct CPUEvent : Event { };

  class GPUTimer {
   private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaEvent_t iter_start_;

   public:
    GPUTimer();
    ~GPUTimer();
    // stream is a pointer itself
    void iter_start(cudaStream_t stream, bool use_cuda_graph = false);
    void event_start(cudaStream_t stream, bool use_cuda_graph = false);
    void event_stop(cudaStream_t stream, bool use_cuda_graph = false);
    float get_measured_time_ms();
    float get_iter_start_to_event_start_ms();
    void sync_stop();

  };

  class CPUTimer {};

 private:
  bool use_cuda_graph_;
  std::string profiling_dir_;
  int repeat_times_;
  int current_reapted_times_;
  int warmup_after_cudagraph_reinit_;
  std::string host_name_;
  std::vector<float> iter_time_ms_;
  std::chrono::time_point<std::chrono::steady_clock> iter_check_;

  int warmup_iterations_;
  int current_iteration_;
  int current_event_idx_;

  std::vector<std::string> interested_events_;
  std::map<cudaStream_t, std::shared_ptr<GPUTimer>> map_stream_to_gpu_timer_;
  std::vector<std::shared_ptr<Event>> events_;
  std::map<std::string, int> map_event_key_to_event_idx;
  int events_num_;

  std::map<cudaStream_t, std::shared_ptr<std::map<std::string, int>>> map_internal_;

  // for thread safe
  std::mutex mtx_;

 public:
  bool init_cuda_graph_this_iter;

  void initialize(bool use_cuda_graph = false);
  void record_event(const char* event_label_char, cudaStream_t stream, int device_id);
  void iter_check();
  void prepare_iter_start();
  int find_event(std::string& event_key);
  void write_result();

  static std::string stream_str(cudaStream_t stream) {
    const void * address = static_cast<const void*>(stream);
    std::stringstream ss;
    ss << address;
    return ss.str();
  }

  static std::string gen_event_key(std::string& event_name, cudaStream_t stream, int same_name_events_occured_order_in_code) {
    return event_name + "_" + stream_str(stream) + "_" + std::to_string(same_name_events_occured_order_in_code);
  }

};

extern Profiler global_profiler;

}  // namespace HugeCTR
