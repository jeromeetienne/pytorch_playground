# from https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# add [5,6] vector
# x_data = torch.cat((x_data, torch.tensor([[5, 6]])), dim=0)


############################################################################

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

############################################################################

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

breakpoint()  # Set a breakpoint here to inspect x_data

pass