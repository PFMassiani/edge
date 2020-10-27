import torch

cuda_available = torch.cuda.is_available()

cpu = torch.device('cpu')
cuda = torch.device('cuda') if cuda_available else None

# Change here to force the use of GPU or CPU where it is applicable
device = cuda if cuda_available else cpu