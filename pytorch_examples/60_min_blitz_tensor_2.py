# from https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
import torch

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)


##################################################

tensor = torch.rand(3, 4)

device = torch.accelerator.current_accelerator()

tensor = tensor.to(device)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


