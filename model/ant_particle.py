import numpy as np
import torch

from sparse import SparseLinear
from torch import nn
from torch import optim


class AntParticle():
    def __init__(self, in_features_avg, hidden_node_avg, connection_percentage, in_features_max, hidden_node_max
                 , in_features_pheromone, connection_pheromone, hidden_node_pheromone, exploration, lr=0.1):
        super().__init__()
        self.loss = nn.MSELoss().float()

        in_features = np.max((1, np.random.binomial(in_features_max, in_features_avg/in_features_max)))
        in_features_strength = exploration * (np.random.rand(in_features_max)) + (1 - exploration) * in_features_pheromone
        self.in_features_idx = np.argpartition(in_features_strength, -in_features)[-in_features:]

        hidden_nodes = np.max((1, np.random.binomial(hidden_node_max, hidden_node_avg/hidden_node_max)))
        hidden_node_strength = exploration * (np.random.rand(hidden_node_max)) + (1 - exploration) * hidden_node_pheromone
        self.hidden_nodes_idx = np.argpartition(hidden_node_strength, -hidden_nodes)[-hidden_nodes:]

        max_connections = in_features * hidden_nodes
        connections = np.random.binomial(max_connections, connection_percentage)
        connections_strength = (exploration * (np.random.rand(*connection_pheromone.shape)) + (1 - exploration) * connection_pheromone)[self.hidden_nodes_idx][:, self.in_features_idx]
        self.connection_matrix = np.ones((hidden_nodes, in_features))

        self.net = Net(in_features, hidden_nodes, self.connection_matrix)

        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.0)

    def train_and_evaluate(self, x, y, iterations=500):
        x = x[self.in_features_idx].t().type(torch.FloatTensor)
        for i in range(iterations):
            self.optimizer.zero_grad()
            output = self.net.forward(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        y_hat = self.net.forward(x)
        accuracy = np.sum(torch.round(y_hat).detach().numpy() == y.detach().numpy()) / len(y)
        if accuracy < 0.1:
            print('wtf')
        return accuracy


class Net(nn.Module):
    def __init__(self, in_features, hidden_nodes, connectivity_matrix):
        super(Net, self).__init__()
        self.sparse = SparseLinear(in_features, hidden_nodes,
                                   torch.as_tensor(connectivity_matrix).type(torch.FloatTensor))
        self.hidden = nn.Linear(hidden_nodes, 1, bias=False)
        self.non_linear = nn.Sigmoid()
        if torch.isnan(self.sparse.weight)[0][0]:
            print('wtf')

    def forward(self, x):
        x = self.sparse(x)
        x = self.non_linear(x)
        x = self.hidden(x)
        return self.non_linear(x)
