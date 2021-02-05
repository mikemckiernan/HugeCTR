# Inline Profiler

Inline Profiler is designed to make better profiling experience than NsightSystem in some aspect and focusing on specific important modules, finally let developer know whether the optimization is really scaling.

Currently NsightSystem has several disadvatanges when profiling HugeCTR:

1. It introduce some overheads.
2. There are many blanks in timeline view and too many kernels. It makes hard for developers to have a clear vision on things they are interested.

So, we come up with this inline profiler, it:

1. Try to reduce overhead as much as possible by running profiling on only 1 kernel per iteration with CudaEvent.
2. Let user define what they are interested and try to provide a more clear vision on how things are optimized.
3. Introduce zero overhead for the code in the formal release version.

## Current limitation

1. Not support CUDA Graph yet. Automatically turn it off in code when complied with `DENABLE_PROFILING=ON`.
2.`PROFILE_RECORD(xxx)` inserted in an `if` or `for` statement probably will not work properly. Each `PROFILE_RECORD(xxx)`
has to be executed once and only once in each layer, within a iteration.

## Overhead
On Selene single node DGXA100. With `55926_8gpus.json` as config:

With cudaGraph, 2.21ms per iter

Without cudaGraph 2.29ms per iter

With profiler(cudagraph off) 2.32ms (so nearly 0.03 ms overhead)

## Usage

For jupyter notebook usage, please start the jupyter notebook server, open the `profiling.ipynb` and follow the usage there.

For command line usage, please go to `profiling.py` and read the guide comment from top to bottom, change what you
need and then execute `python profiling.py`.