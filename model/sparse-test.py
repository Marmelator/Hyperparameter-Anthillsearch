import numpy as np
from sparse import SparseLinear
import torch
from torch import optim
from torch import nn

x = torch.as_tensor([[1, 2, 3]]).type(torch.FloatTensor)
y = torch.as_tensor(5).type(torch.FloatTensor)
connectivity_matrix = torch.as_tensor([[1, 1, 1]])

sparse = SparseLinear(3, 1, connectivity_matrix)
l2_loss = nn.MSELoss()
optimizer = optim.SGD(sparse.parameters(), lr=0.01, momentum=0.0)

for i in range(100):
    optimizer.zero_grad()

    output = sparse.forward(x)
    loss = l2_loss(output, y)
    loss.backward()
    print(loss)
    optimizer.step()
