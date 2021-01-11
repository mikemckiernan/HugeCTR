

#ifdef ENABLE_PROFILING
#define PROFILE_RECORD(...) do \
{ \
  global_profiler.record_event(__VA_ARGS__); \
} while (0)
#else
#define PROFILE_RECORD(x) do {} while (0)
#endif

namespace HugeCTR {
class InlineProfiler {
  struct Event {
    char* name;
    unsigned int start_index;
    unsigned int  end_index;
    // std::vector<unsigned int> on_iters;
    std::vector<float> measured_time;
  };

  struct GPUEvent : Event {
    unsigned int device_id;
    unsigned int stream_id;
  };

  struct CPUEvent : Event { };

  class Timer {
   public:
    virtual Event* record_event() = 0;
    virtual void event_start() = 0;
    virtual void event_stop() = 0;
    virtual float get_result() = 0;
  };

  class GPUTimer : public Timer {
   private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaStream_t stream_;

   public:
    void GPUTimer(cudaStream_t);
    void reset();
    // stream is a pointer itself
    void event_start(cudaStream_t stream);
    void event_stop(cudaStream_t stream);
  };

  class CPUTimer : public Timer { };

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
  InlineProfiler();
  void initialize(std::string schedule_file);
  void record_event(std::string event_label, cudaStream_t streams);
  void iter_start();
  void iter_end()
  int find_event(std::string event_name);
}

namespace Helpers {
  std::shared_ptr<std::vector<std::string>> split_string(std::string str, char delim = '.') {
    std::stringstream ss(str);
    std::shared_ptr<std::vector<std::string>> result = std::make_shared(new std::vector<std::string>());
    std::string token;
    while (std::getline(ss, token, delim)) {
        result->push_back(token);
    }
    return result;
  }

  int get_device_id(cudaStream_t stream) {
    CUcontext* pctx;
    CUdevice device;
    cuStreamGetCtx(stream, pctx);
    cuCtxPushCurrent(*pctx);
    cuCtxGetDevice(&device);
    cuCtpopCurrent(pctx);
    unsigned int device_id;
    //CUdevice_attribute attr = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
    CUdevice_attribute attr = CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID;
    cuDeviceGetAttribute(&device_id, attr, device);
    return device_id;
  }

}  // namespace Helpers
}  // namespace HugeCTR
