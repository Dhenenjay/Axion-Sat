import os
os.add_dll_directory(r'C:\Users\Dhenenjay\Axion-Sat\.venv\Lib\site-packages\torch\lib')

import torch
print('='*60)
print('PyTorch Installation Test')
print('='*60)
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
else:
    print('CUDA Version: N/A')
    print('GPU Device: N/A')
print('='*60)
