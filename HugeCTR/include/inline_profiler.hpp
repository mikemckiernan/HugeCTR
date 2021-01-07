#ifdef ENABLE_INLINE_PROFILE
#define INLINE_PROFILE_EVENT_START(x) do \
{ \
  global_inline_profiler.event_start(x); \
} while (0)
#define INLINE_PROFILE_EVENT_END(x) do \
{ \
  global_inline_profiler.event_end(x); \
} while (0)
#else
#define INLINE_PROFILE_EVENT_START(x) do {} while (0)
#define INLINE_PROFILE_EVENT_END(x) do {} while (0)
#endif

namespace HugeCTR {
  class InlineProfiler {

    enum EventType {
      INLINE_PROFILER_EVENT_START = 0,
      INLINE_PROFILER_EVENT_END = 1
    };

    struct Event {
      char* name;
      EventType type;
      float elapsed_time;
    };

    struct GPUEvent : Event {
      char* device_id;
      char* stream_name;
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
     public:
      cudaEvent_t start;
      cudaEvent_t stop;
      cudaStream_t *current_stream;

      void reset();
      void event_start(cudaStream_t& stream);
      void event_stop(cudaStream_t& stream);
    };

    class CPUTimer : public Timer { };

   public:
    std::string host_name;
    unsigned int current_iteration = 0;
    unsigned int warmup_iterarions = 1000;
    unsigned int events_num = 0;
    std::queue<std::string> interested_event_names;
    std::queue<unsigned int> measure_times;

    std::vector<*cudaStream_t> gpu_streams;
    std::vector<std::string> gpu_stream_names;
    std::vector<Event*> events;

    std::mutex mtx; // for thread safe

    GPUTimer gpu_timer;
    CPUTimer cpu_timer;

    unsigned std::string current_event_name;
    unsigned int current_measure_times;
    std::vector<float> current_measure_results;


    InlineProfiler();
    void initialize(std::string schedule_file);
    void reset();
    void gpu_event_start();
    void gpu_event_end();

  }


}  // namespace HugeCTR
