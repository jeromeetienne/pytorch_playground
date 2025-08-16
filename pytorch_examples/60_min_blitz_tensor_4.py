# from https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
import torch

A = torch.tensor([1, 2, 3])
B = torch.tensor([4, 5, 6])

# Element-wise multiplication
result = A * B  # [4, 10, 18]

print(f"Result of element-wise multiplication: {result}\n")

result.add_(10)  # Adding 10 to each element

print(f"Result of element-wise multiplication after adding 10: {result}\n")

# matrix multiplication
A2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
B2 = torch.tensor([[7, 8], [9, 10], [11, 12]])

result2 = A2 @ B2  # Matrix multiplication

print(f"Result of matrix multiplication:\n{result2}\n")