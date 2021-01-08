#include <inline_profiler.hpp>

namespace HugeCTR {

namespace InlineProfiler {

  void GPUTimer::GPUTimer(cudaStream_t stream) {
    stream_ = stream;
  }

  void GPUTimer::reset() {
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->stop);
    cudaEventCreate(this->start);
    cudaEventCreate(this->stop);
    this->current_stream = nullptr;
  }

  void GPUTimer::event_start(cudaStream_t stream) {
    this->current_stream = &stream;
    cudaEventRecord(this->start, stream);
  }

  void GPUTimer::event_end(cudaStream_t stream) {
    cudaEventRecord(this->end, stream);
  }

  float GPUTimer::get_result() {
    float elapsed_time;
    cudaElapsedTime(this->start, this->stop);
    return elapsed_time;
  }


  InlineProfiler() {};

  void initialize(std::string schedule_file) {
    // read from a schedule file, schedule file format:
    //    event_name_1, measure_times_1
    //    event_name_2, measure_times_2
    //    ...
    // the file represent the event that user want to profile and show in the final result
    // parse the file, and enqueue interested_event_names and measure_times
    
  }

  void record_event(std::string event_label, cudaStream_t stream) {
    int dot_pos = event_label.find_last_of(std::string('.'));
    std::string event_type = event_label.substr(dot_pos + 1);
    if (event_type != "start" || event_type != "stop") {
      throw internal_runtime_error(HugeCTR::Error_t::UnspecificError, \
      std::string("Invalid event name. Should end with .start or .end"))
    }
    std::string event_name = event_label.substr(0, dot_pos);
    if (std::find(interested_event_names.begin(), interested_event_names.end(), event_name) \
      == interested_event_names.end()) {
        return;
      }
    if (current_iter_ <= warmup_iters_) {
      mtx_.lock();
      if (std::find(gpu_streams_.begin(), gpu_streams_.end(), stream) == gpu_streams_.end()) {
        std::shared_ptr<GPUTimer> gpu_timer = make_shared<GPUTimer>(stream);
        gpu_streams_.append(stream)
        gpu_timers_.append(gpu_timer)
      }
      // if event is already existed

      // get device id from stream
      unsigned int device_id = get_device_id(stream)

      if (event_type == "start") {
        // create new event

        GPUEvent event = {
          event_name,
          events_num_,
          0, // wait for stop event to set
        }
        events_num_++;
      } else {
        
      }



    } else {
      
    }
  }



}  // namespace InlineProfiler

}  // namespace sHugeCTR

// A global variable global_inline_profiler;
HugeCTR::InlineProfiler global_inline_profiler();