

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
    unsigned int stream_name;
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
  unsigned int current_iter_;
  unsigned int warmup_iters_;
  std::vector<std::string> interested_event_names_;
  //std::queue<unsigned int> measure_times;
  // stream
  std::vector<cudaStream_t> gpu_streams_;
  std::vector<std::string> gpu_stream_names_;
  std::vector<Event> events_;
  unsigned int events_num_;
  std::mutex mtx_; // for thread safe
  std::vector<std::shared_ptr<GPUTimer>> gpu_timers_;
  std::vector<std::shared_ptr<CPUTimer>> cpu_timers_;

 public:
  InlineProfiler();
  void initialize(std::string schedule_file);
  void prepare_settings();
  // expect the event_name to be in format of 'xxx.xxx.start' or 'xxx.xxx.stop', number of x
  // can be variant
  void record_event(std::string event_label, cudaStream_t streams);
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
