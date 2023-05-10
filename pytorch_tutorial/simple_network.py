#%%
import torch
import torch.nn as nn # neuronal layers, function objects
import torch.nn.functional as F # contains activation functions
from torch.autograd import Variable # wrapper for tensor
import torch.optim as opt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")
#%%
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # dummy parameter to store device
        self.device = device

        # define layers (layers are functions)
        self.lin1 = nn.Linear(10, 10) # 10 features in, 10 features out
        self.lin2 = nn.Linear(10, 10) # input and output numbers need to match!

    # forward pass data x through network
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.hardtanh(self.lin1(x))
        return x
    

    # calculate how many features are contained in a single data item tensor (num of elements)
    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i

        return num

#%%
net = SimpleNet()
net.to(device)
print(net)
#%%
for i in range (100):
    # main training loop / pipeline
    #data = torch.randn(10,10) # 10 data points (directly create batch of 10)
    # define mockup data
    data_x = [1,0,0,0,1,0,0,0,1,1]
    data_y = [0,1,1,1,0,1,1,1,0,0]
    input = Variable(torch.Tensor([data_x for _ in range(10)])).to(net.device) # create as batch (duplicate for demonstration)
    target = Variable(torch.Tensor([data_y for _ in range(10)])).to(net.device) # s.o.
    
    # forward pass
    out = net(input)
    loss = F.mse_loss(out, target)
    net.zero_grad() # net accumulates errors and needs to be reset manually each step

    # backpropagation
    loss.backward()
    learning_rate = 0.1
    optimizer = opt.SGD(net.parameters(), lr=learning_rate) # default stochastic gradient descent
    optimizer.step()
    
    print(loss)
#%%
# example result of trained net and conversion to numpy
x_numpy = np.array([1,0,0,0,1,0,0,0,1,1])
x = torch.Tensor(x_numpy).to(net.device)
y = net(x)
y_numpy = y.detach().cpu().numpy()
print("in  ", x_numpy.astype(np.int8))
print("out ", np.abs(np.rint(y_numpy)).astype(np.int8)) # as nearest (positive) int
# %%
