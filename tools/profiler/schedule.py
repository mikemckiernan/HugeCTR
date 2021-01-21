import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from tools.profiler import generate_schedule

dlrm_perf_schedule = {
    # interested event name
    'BottomMLP.fc1': [
        'fused_fully_connected.fprop',
        'fused_fully_connected.fprop.cublasGemmEx',
#        'fused_fully_connected.fprop.add_bias_and_re_kernel',
        'fused_fully_connected.bprop',
#        'fused_fully_connected.bprop.initialize_array',
        'fused_fully_connected.bprop.reverse_add_bias_and_re_kernel',
        'fused_fully_connected.bprop.convert_array',
        'fused_fully_connected.bprop.cublasGemmEx_1',
        'fused_fully_connected.bprop.cublasGemmEx_2'
    ],
    'BottomMLP.fc2': [
        'fused_fully_connected.fprop',
        'fused_fully_connected.fprop.cublasGemmEx',
        'fused_fully_connected.fprop.add_bias_and_re_kernel',
        'fused_fully_connected.bprop',
        'fused_fully_connected.bprop.initialize_array',
#        'fused_fully_connected.bprop.reverse_add_bias_and_re_kernel',
        'fused_fully_connected.bprop.convert_array',
        'fused_fully_connected.bprop.cublasGemmEx_1',
#        'fused_fully_connected.bprop.cublasGemmEx_2'
    ],
    'BottomMLP.fc3': [
        'fused_fully_connected.fprop',
        'fused_fully_connected.fprop.cublasGemmEx',
        'fused_fully_connected.fprop.add_bias_and_re_kernel',
#        'fused_fully_connected.bprop',
        'fused_fully_connected.bprop.initialize_array',
        'fused_fully_connected.bprop.reverse_add_bias_and_re_kernel',
#        'fused_fully_connected.bprop.convert_array',
        'fused_fully_connected.bprop.cublasGemmEx_1',
        'fused_fully_connected.bprop.cublasGemmEx_2'
    ],
    'sparse_embedding1': [
        'localized_slot_sparse_embedding_one_hot.forward.mapping_and_fuse'
    ],
    'interaction1': [

    ],
    'TopMLP.fc4': [
        'fused_fully_connected.fprop',
        'fused_fully_connected.fprop.cublasGemmEx',
#        'fused_fully_connected.fprop.add_bias_and_re_kernel',
        'fused_fully_connected.bprop',
        'fused_fully_connected.bprop.initialize_array',
        'fused_fully_connected.bprop.reverse_add_bias_and_re_kernel',
#        'fused_fully_connected.bprop.convert_array',
        'fused_fully_connected.bprop.cublasGemmEx_1',
        'fused_fully_connected.bprop.cublasGemmEx_2'
    ],
    'TopMLP.fc5': [
        'fused_fully_connected.fprop',
#        'fused_fully_connected.fprop.cublasGemmEx',
        'fused_fully_connected.fprop.add_bias_and_re_kernel',
#        'fused_fully_connected.bprop',
        'fused_fully_connected.bprop.initialize_array',
#        'fused_fully_connected.bprop.reverse_add_bias_and_re_kernel',
        'fused_fully_connected.bprop.convert_array',
#        'fused_fully_connected.bprop.cublasGemmEx_1',
        'fused_fully_connected.bprop.cublasGemmEx_2'
    ],
    'TopMLP.fc6': [
        'fused_fully_connected.fprop',
        'fused_fully_connected.fprop.cublasGemmEx',
        'fused_fully_connected.fprop.add_bias_and_re_kernel',
        'fused_fully_connected.bprop',
        'fused_fully_connected.bprop.initialize_array',
        'fused_fully_connected.bprop.reverse_add_bias_and_re_kernel',
        'fused_fully_connected.bprop.convert_array',
        'fused_fully_connected.bprop.cublasGemmEx_1',
        'fused_fully_connected.bprop.cublasGemmEx_2'
    ],
    'TopMLP.fc7': [
    ]
}

def main():
    generate_schedule(dlrm_perf_schedule, repeat_for_each_event=50)

if __name__ == '__main__':
    main()