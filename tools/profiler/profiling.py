import sys
import os
from datetime import datetime
from shutil import copy
import json

project_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.append(project_root)

from tools.profiler import generate_schedule, parse_result

'''
Should you have any question, please contact Randy Wang(ruotongw@nvidia.com).

Usage Guide

1. Add -DENABLE_PROFILING=ON when pass args to Cmake. And then do the normal make process.
2. Comment out anything you are not interested in dlrm_perf_schedule below. Just comment the event label line,
   don't comment out the forward_events, backward_events or BottomMLP.fc1 line. You can check for all labels
   in the cpp code. And also you can add your own label and recompile it, then insert correspond label in dlrm_perf_schedule.
'''

dlrm_perf_schedule = {
    # interested event name
    'BottomMLP.fc1': {
        'forward_events': [
            'fused_relu_bias_fully_connected.fprop',
            'fused_relu_bias_fully_connected.fprop.cublasLtMatmul',
        ],
        'backward_events': [
            'fused_relu_bias_fully_connected.bprop',
            'fused_relu_bias_fully_connected.bprop.initialize_array',
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
            'fused_relu_bias_fully_connected.bprop.convert_array',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_1',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_2'
        ]
    },
    'BottomMLP.fc2': {
        'forward_events': [
            'fused_relu_bias_fully_connected.fprop',
            'fused_relu_bias_fully_connected.fprop.cublasLtMatmul',
        ],
        'backward_events': [
            'fused_relu_bias_fully_connected.bprop',
            'fused_relu_bias_fully_connected.bprop.initialize_array',
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
            'fused_relu_bias_fully_connected.bprop.convert_array',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_1',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_2'
        ]
    },
    'BottomMLP.fc3': {
        'forward_events': [
            'fused_relu_bias_fully_connected.fprop',
            'fused_relu_bias_fully_connected.fprop.cublasLtMatmul',
        ],
        'backward_events': [
            'fused_relu_bias_fully_connected.bprop',
            'fused_relu_bias_fully_connected.bprop.initialize_array',
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
            'fused_relu_bias_fully_connected.bprop.convert_array',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_1',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_2'
        ]
    },
    'sparse_embedding1': {
        'forward_events': [
            'localized_slot_sparse_embedding_one_hot.forward.mapping_and_fuse'
        ],
        'backward_events': []
    },
    'interaction1': {
        'forward_events': [],
        'backward_events': []
    },
    'TopMLP.fc4': {
        'forward_events': [
            'fused_relu_bias_fully_connected.fprop',
            'fused_relu_bias_fully_connected.fprop.cublasLtMatmul',
        ],
        'backward_events': [
            'fused_relu_bias_fully_connected.bprop',
            'fused_relu_bias_fully_connected.bprop.initialize_array',
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
            'fused_relu_bias_fully_connected.bprop.convert_array',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_1',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_2'
        ]
    },
    'TopMLP.fc5': {
        'forward_events': [
            'fused_relu_bias_fully_connected.fprop',
            'fused_relu_bias_fully_connected.fprop.cublasLtMatmul',
        ],
        'backward_events': [
            'fused_relu_bias_fully_connected.bprop',
            'fused_relu_bias_fully_connected.bprop.initialize_array',
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
            'fused_relu_bias_fully_connected.bprop.convert_array',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_1',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_2'
        ]
    },
    'TopMLP.fc6': {
        'forward_events': [
            'fused_relu_bias_fully_connected.fprop',
            'fused_relu_bias_fully_connected.fprop.cublasLtMatmul',
        ],
        'backward_events': [
            'fused_relu_bias_fully_connected.bprop',
            'fused_relu_bias_fully_connected.bprop.initialize_array',
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
            'fused_relu_bias_fully_connected.bprop.convert_array',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_1',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_2'
        ]
    },
    'TopMLP.fc7': {
        'forward_events': [
            'fused_relu_bias_fully_connected.fprop',
            'fused_relu_bias_fully_connected.fprop.cublasLtMatmul',
        ],
        'backward_events': [
            'fused_relu_bias_fully_connected.bprop',
            'fused_relu_bias_fully_connected.bprop.initialize_array',
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
            'fused_relu_bias_fully_connected.bprop.convert_array',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_1',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_2'
        ]
    },
    'TopMLP.fc8': {
        'forward_events': [],
        'backward_events': []
    },
}

'''
3. Define profiling_dir for this profiling session. Specify the config file you want to use.
'''
config_file = os.path.join(project_root, 'mlperf', 'configs', '55296_8gpus.json')
working_dir = os.path.abspath(os.path.dirname(__file__))
profiling_dir = os.path.join(working_dir, 'test')

def gen_schedule():
    # create if profiling_dir non-exist.
    os.makedirs(profiling_dir, exist_ok=True)
    # Copy config to profiling_dir, for backup
    copy(config_file, profiling_dir)
    # Create a prof.schedule in profiling_dir. This file will instruct cpp profiler how to prof. 
    generate_schedule(dlrm_perf_schedule, profiling_dir)

if __name__ == '__main__':
    '''
    4. First you should create a profiling_dir. Just uncomment it.
    '''
    #gen_schedule()
    '''
    5. Run the training

    If you are not on cluster, you should upload the profiling_dir to corresponding location on cluster. Then you
    should run the training process like:

        export PROFILING_DIR=${profiling dir defined above}
        numactl ./build/bin/hugectr --train ${PROFILING_DIR}/55926_8gpus.json

    Or you can run from the login node like:

        ## DL params
        export CONTAINER_NAME=hugectr_inline_profiler
        export IMAGE=gitlab-master.nvidia.com/dl/mlperf/optimized:recommendation.hugectr.2035814
        export CONFIG="tools/profiler/test/55296_8gpus.json"
        export MOUNTS=/raid:/raid,/lustre/fsw/mlperft-dlrm/ruotongw/hugectr:/etc/workspace/home
        # For Profiler
        export PROFILING_DIR=tools/profiler/test
        ## NCCL WAR
        export NCCL_LAUNCH_MODE=PARALLEL
        # Setup container
        srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${IMAGE}" --container-name="${CONTAINER_NAME}" true
        srun \
            --mpi=pmix \
            --ntasks="${SLURM_JOB_NUM_NODES}" \
            --ntasks-per-node=1 \
            --container-workdir /etc/workspace/home \
            --container-name="${CONTAINER_NAME}" \
            --container-mounts="${MOUNTS}" \
            --export=NCCL_LAUNCH_MODE=PARALLEL,PROFILING_DIR=${PROFILING_DIR} \
            numactl --interleave=all /bin/bash -c "./build/bin/huge_ctr --train ${CONFIG}"


    The hugectr will exit after the profiling is completed, usually only run for 1000 - 3000 iters, depends on how many
    interested events you defined in the dlrm_perf_schedule. The raw result will appear in profiling_dir as ${host_name}.prof.json.
    If you use multiple nodes, there will be several jsons appear. The result json is not human readable, so please use function below to parse it.
    '''

    '''
    6. Parse the result into more human readable format. Just uncomment it.

    And you can do anything you like from the result, for instance save it as a file.
    '''
    #result = parse_result(profiling_dir)
    #print(json.dumps(result, indent=2))

