#include <inline_profiler.hpp>

namespace HugeCTR {

namespace InlineProfiler {

  void GPUTimer::reset() {
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->stop);
    cudaEventCreate(this->start);
    cudaEventCreate(this->stop);
    this->current_stream = nullptr;
  }

  void GPUTimer::event_start(cudaStream_t& stream) {
    this->current_stream = &stream;
    cudaEventRecord(this->start, stream);
  }

  void GPUTimer::event_end(cudaStream_t& stream) {
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

  void gpu_event_start(std::string& event_name, cudaStream_t& stream) {
    // workflow:
    //   if event_name is not in the interested_event_names:
    //      return;
    //   if current_iteration < warmup_iterarions:
    //      lock mutex, avoid multiple thread issue
    //      this.event_num += 1
    //      look up stream in this->gpu_streams by pointer comparing. If not found,
    //      append this stream to this->gpu_streams. And then use gpu_streams.size() - 1
    //      as stream_name and append it to this->gpu_stream_names;
    //      create a new GPUEvent object with event_name, type = INLINE_PROFILER_EVENT_START
    //      stream_name = stream_name. Obtain gpu device id from stream using cuda driver api:
    //      https://stackoverflow.com/questions/31474784/are-cuda-streams-device-associated-and-how-do-i-get-a-streams-device
    //      insert the GPUEvent into events
    //   if current_iteration > warmup_iterarions:
    //      start actual profile
    //      
  }



}  // namespace InlineProfiler

}  // namespace sHugeCTR

// A global variable global_inline_profiler;
HugeCTR::InlineProfiler global_inline_profiler();