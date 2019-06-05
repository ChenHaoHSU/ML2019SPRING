import sys
import torch
original_path = sys.argv[1]
quantized_path = sys.argv[2]
state_dict = torch.load(original_path)
for k, v in state_dict.items():
    state_dict[k] = v.type(torch.cuda.HalfTensor)
torch.save(state_dict, quantized_path)