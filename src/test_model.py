import torch
from model import ECGCNN


model = ECGCNN()

x = torch.randn(32 , 187)

output = model(x)

print("Output shape: " , output.shape) # Output shape:  torch.Size([32, 5])