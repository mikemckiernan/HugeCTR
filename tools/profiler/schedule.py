import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from tools.profiler import generate_schedule

dlrm_schedule = {
    # interested event name
    'BottomMLP.fc1': [
        'fused_fully_connected',
        'fused_fully_connected.cublasGemm',
        'fused_fully_connected.add_bias_and_re_kernel',
    ],
    'BottomMLP.fc2': [
        'fused_fully_connected',
        'fused_fully_connected.cublasGemm',
    ],
    'BottomMLP.fc3': [
        'fused_fully_connected.cublasGemm',
        'fused_fully_connected.add_bias_and_re_kernel',
    ],
    'sparse_embedding1': [],
    'interaction1': [],
    'TopMLP.fc4': [
        'fused_fully_connected',
        'fused_fully_connected.cublasGemm',
        'fused_fully_connected.add_bias_and_re_kernel',
    ],
    'TopMLP.fc5': [
        'fused_fully_connected',
    ],
    'TopMLP.fc6': [
        'fused_fully_connected',
        'fused_fully_connected.add_bias_and_re_kernel',
    ],
    'TopMLP.fc7': [
    ]
}

def main():
    generate_schedule(dlrm_schedule)

if __name__ == '__main__':
    main()