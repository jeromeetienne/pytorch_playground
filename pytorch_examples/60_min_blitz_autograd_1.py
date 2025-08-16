import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

input_data = torch.rand(1, 3, 64, 64)
input_label = torch.rand(1, 1000)

predicted_label = model(input_data) # forward pass

loss = (predicted_label - input_label).sum()
loss.backward() # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optim.step() #gradient descent