import sys
import torch
original_path = sys.argv[1]
quantized_path = sys.argv[2]
model_dict = torch.load(original_path)
for key, value in model_dict.items():
    model_dict[key] = value.type(torch.cuda.HalfTensor)
torch.save(model_dict, quantized_path)