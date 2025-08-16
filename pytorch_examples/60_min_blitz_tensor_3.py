# from https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
import torch

tensor = torch.ones(4,4)

print(f"Original tensor:\n{tensor}\n")

tensor[:,1] = 0

print(f"Modified tensor:\n{tensor}\n")


tensor_cat = torch.cat((tensor, tensor, tensor), dim=1)

print(f"Concatenated tensor:\n{tensor_cat}\n shape: {tensor_cat.shape}")