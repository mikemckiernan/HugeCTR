import sys
import os
from datetime import datetime
from shutil import copy
import json

project_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.append(project_root)

from tools.profiler import gen_prof_config, parse_result

'''
Should you have any question, please contact Randy Wang(ruotongw@nvidia.com).

Usage Guide

1. Insert Macro and Complie

a. You can insert macro `PROFILE_RECORD(xxx)` to CPP code to instruct profiler to profile this interval. There are some
macro inserted already, please have a check if you are looking for example. Remeber to insert a pair of labels `xxx.start`
and `xxx.stop`.

b. Add `-DENABLE_PROFILING=ON` when pass args to Cmake. And then do the normal `make` process.
'''

dlrm_interesed_events = {
    # interested event name
    'High_Level': {
        'forward_events': [
            'BottomMLP.fprop',
            'TopMLP.fprop',
            'Bottom&TopMLP.fprop'
        ],
        'backward_events': [
            'BottomMLP.bprop',
            'TopMLP.bprop',
            'Bottom&TopMLP.bprop'
        ]
    },
    'BottomMLP.fc1': {
        'forward_events': [
            'fused_relu_bias_fully_connected.fprop',
            'fused_relu_bias_fully_connected.fprop.cublasLtMatmul',
        ],
        'backward_events': [
            'fused_relu_bias_fully_connected.bprop',
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
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
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
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
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_1',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_2'
        ]
    },
    'Embedding': {
        'forward_events': [
            'localized_slot_sparse_embedding_one_hot.forward',
#            'all2all_forward',
#            'inter_node_hier_a2a.fprop'
        ],
        'backward_events': [
            'localized_slot_sparse_embedding_one_hot.backward',
#            'all2all_backward',
#            'inter_node_hier_a2a.bprop',
            'localized_slot_sparse_embedding_one_hot.update_params'
        ]
    },
    'Interaction': {
        'forward_events': [
            'interaction.fprop'
        ],
        'backward_events': [
            'interaction.bprop'
        ]
    },
    'TopMLP.fc4': {
        'forward_events': [
            'fused_relu_bias_fully_connected.fprop',
            'fused_relu_bias_fully_connected.fprop.cublasLtMatmul',
        ],
        'backward_events': [
            'fused_relu_bias_fully_connected.bprop',
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
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
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
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
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
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
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_1',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_2'
        ]
    },
    'TopMLP.fc8': {
        'forward_events': [
            'fused_relu_bias_fully_connected.fprop',
            'fused_relu_bias_fully_connected.fprop.cublasLtMatmul',
        ],
        'backward_events': [
            'fused_relu_bias_fully_connected.bprop',
            'fused_relu_bias_fully_connected.bprop.reverse_add_bias_and_re_kernel',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_1',
            'fused_relu_bias_fully_connected.bprop.cublasGemmEx_2'
        ]
    },
    'Loss': {
        'forward_events': [
#            'compute'
        ]
    },
    'AllReduce_wgrads': {
        'forward_events': [
#            'exchange_wgrad'
        ]
    },
    'Update_Params': {
        'forward_events': [
#            'update'
        ]
    }
}

'''
3. Set related variable
'''

# Define profiling_dir for this profiling session.
# Specify the config file you want to use. Also other configs, like slurm related.
working_dir = os.path.join('tools', 'profiler')
# profiler related
profiling_dir_name = 'test'
profiling_dir = os.path.join(working_dir, 'results', profiling_dir_name)

# train configs
config_file = os.path.join('mlperf', 'configs', '55296_8gpus.json')
# slurm related
nodes_num = 1
container_name = 'hugectr-dlrm-profiling'
image = 'gitlab-master.nvidia.com/dl/mlperf/optimized:recommendation.hugectr.2035814'
mounts_str = '/raid:/raid,/lustre/fsw/mlperf/mlperft-dlrm/ruotongw/hugectr:/etc/workspace/home'
account = 'mlperf'
jobid = '1069007'

if __name__ == '__main__':
    '''
    4. Generate the profiling dir (optional)

    By default, profiler will prof every label inserted in CPP code, it usually cost 3min - 5min to finish the profiling,
    depends on how many labels inserted. And the `parse_result` will use `dlrm_interesed_events` to filer the result. So
    don't worry too many labels in CPP code will ruin the result. 

    But, if you truly want to profile anything you care about, and to shorten the profiling time maybe, you can use below
    function to generate profiling dir and a `prof.events` file in it. It will instruct profiler to strictly only profile
    what you list in `dlrm_interesed_events`. And you need to somehow upload the profiling dir to the corresponding
    location on cluster or container.
    '''
    #gen_prof_config(os.path.join(project_root, profiling_dir), interested_events=dlrm_interesed_events)
    '''
    5. Run the training

    5.a on the cluter, in the login node

    You may want to first
    ```
    salloc -p luna -A {account} -N{nodes_num} bash
    ```
    to apply for resources in advance. Remeber the jobid and fill in above.

    The hugectr will exit after the profiling is completed, usually only run for 1000 - 3000 iters, depends on how many
    interested events you defined in the dlrm_perf_schedule. The raw result will appear in profiling_dir as ${host_name}.prof.json.
    If you use multiple nodes, there will be several jsons appear. The result json is not human readable, so please use function
    below to parse it.
    '''

    cmd = '''
        srun --mpi=pmix --ntasks="{nodes_num}" --ntasks-per-node=1 --container-workdir /etc/workspace/home \\
             --container-name="{container_name}" --container-mounts="{mounts_str}" --container-image={image} \\
             --export=NCCL_LAUNCH_MODE=PARALLEL,PROFILING_DIR={profiling_dir} -A {account} --jobid={jobid}\\
             numactl --interleave=all ./build/bin/huge_ctr --train {config_file}
    '''.format(mounts_str=mounts_str, nodes_num=nodes_num, image=image, container_name=container_name,
               profiling_dir=profiling_dir, config_file=config_file, account=account, jobid=jobid)

    #os.system(cmd)

    '''
    5.b If you are already in the hugectr container, you have to run the trainning with env PROFILING_DIR, like:

    export NCCL_LAUNCH_MODE=PARALLEL
    export PROFILING_DIR={profiling_dir in the container}
    numactl --interleave=all ./build/bin/huge_ctr --train {config_file}
    '''

    '''
    6. Result

    The hugectr will exit after the profiling is completed. The raw result will appear in `profiling_dir` as
    `${host_name}.prof.json`. If you use multiple nodes, there will be several jsons appear. The result json is not
    human readable, so please use function below to parse it.
    '''
    # result = parse_result(os.path.join(project_root, profiling_dir), dlrm_interesed_events)
    #with open(os.path.join(project_root, profiling_dir, 'test' + '.json'), 'w') as f:
    #    json.dump(result, f, indent=2)
    #print(json.dumps(result, indent=2))

