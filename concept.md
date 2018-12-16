```
import torch.nn.functional as F

input = torch.Tensor([[[1,2,3,4,5,6,7]]])
F.avg_pool1d(input, kernel_size=3, stride=2, ceil_mode=True)

dim = (input.shape[-1] - kernel_size)/stride + 1
ceil_mode = True
out_dim = np.ceil(dim) if ceil_mode else np.floor(dim)

```
