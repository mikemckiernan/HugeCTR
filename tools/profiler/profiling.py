import sys
import os
from datetime import datetime
from shutil import copy

project_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.append(project_root)

from tools.profiler import generate_schedule, parse_result, print_result

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
config_file = os.path.join(project_root, 'mlperf', 'configs', '55296_8gpus.json')
working_dir = os.path.abspath(os.path.dirname(__file__))
profiling_dir = os.path.join(working_dir, 'test')

def gen_schedule():
    os.makedirs(profiling_dir, exist_ok=True)
    copy(config_file, profiling_dir)
    # generate the schedule
    generate_schedule(dlrm_perf_schedule, profiling_dir)

if __name__ == '__main__':
    gen_schedule()
    #result = parse_result(profiling_dir)
    #print_result(result)