# Inline Profiler

Inline Profiler is designed to make better profiling experience than NsightSystem in some aspect and focusing on specific important modules, finally let developer know whether the optimization is really scaling.

Currently NsightSystem has several disadvatanges when profiling HugeCTR:

1. It introduce some overheads.
2. There are many blanks in timeline view and too many kernels. It makes hard for developers to have a clear vision on things they are interested.

So, we come up with this inline profiler, it:

1. Try to reduce overhead as much as possible by running profiling on only 1 kernel per iteration with CudaEvent.
2. Let user define what they are interested and try to provide a more clear vision on how things are optimized.
3. Introduce zero overhead for the code in the formal release version.

## Usage

1. Add `-DENABLE_PROFILING=ON` when pass args to Cmake. And then do the normal `make` process.

2. In `tools/profiler/schedule.py`, there defines all the optional events you may want to profile in each layer. You can just comment out some events you are not inerested in the `dlrm_perf_schedule`. If you want to add new event, please contact Randy Wang or Daniel Abel. It would be better to insert the profile macro `PROFILE_RECORD(label, stream, device_id)` by the profiler developer. We will try to meet developers' need and add necessary events.

3. `cd HugeCTR_PROJECT_ROOT` and `python3 tools/profiler/schedule.py`. This will generate a `prof.schedule` file under the project root. It is used to instruct profiler to schedule profiling.

4. Then just train the hugectr like normal, for example `NCCL_LAUNCH_MODE=PARALLEL ./build//bin/huge_ctr --train ./experiment/dlrm_fp16_16k.json`. The profiler will collect data during training and prof only 1 kernel per iteration and the program will terminate after profiling complete.

5. A file named 'prof.json' will appear in the project root. It records the kernel execution order, stream, device, prof time, etc.