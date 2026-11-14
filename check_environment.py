import platform
import sys
try:
    import torch
except ImportError:
    print('PyTorch is not installed. Follow README.md setup instructions.')
    raise SystemExit(1)
print('Python:', sys.version.split()[0])
print('Platform:', platform.platform())
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA runtime bundled with PyTorch:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('Running on CPU. If you expected GPU, reinstall a CUDA-enabled PyTorch build.')
