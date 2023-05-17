import torch
x = torch.rand(5,3)
print(x)
print("cuda available ", torch.cuda.is_available())