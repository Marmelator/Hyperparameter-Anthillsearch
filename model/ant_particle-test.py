import numpy as np
import torch
from ant_particle import AntParticle
from Data.dataset import MushroomData
import matplotlib.pyplot as plt

hidden_node_max = 5
pheromone_degradation = 0.9

mushroom_data = MushroomData()
x = torch.as_tensor(mushroom_data.x)
y = torch.as_tensor(mushroom_data.y).float()

in_feature_pheromone = np.zeros(len(x))
hidden_node_pheromone = np.zeros(hidden_node_max)
connection_pheromone = np.zeros((hidden_node_max, hidden_node_max * len(x)))

for j in range(15):
    new_in_feature_pheromone = np.zeros(len(x))
    new_hidden_node_pheromone = np.zeros(hidden_node_max)
    new_connection_pheromone = np.zeros((hidden_node_max, hidden_node_max * len(x)))
    best_score = 0.0
    exploration = np.max((0.0, 1 - 0.2*j))
    print("Exploration Factor is now {}".format(exploration))
    for i in range(15):
        particle = AntParticle(15, 3, 1, len(x), hidden_node_max, in_feature_pheromone,
                               connection_pheromone, hidden_node_pheromone, exploration, lr=0.3)
        accuracy = particle.train_and_evaluate(x, y)
        score = np.abs(accuracy - 0.5) * 2

        new_in_feature_pheromone[particle.in_features_idx] += score * torch.sum(torch.abs(particle.net.sparse.weight), dim=0) \
            .detach().numpy()
        new_hidden_node_pheromone[particle.hidden_nodes_idx] += score * (torch.sum(torch.abs(particle.net.sparse.weight), dim=1)
            + torch.sum(torch.abs(particle.net.hidden.weight), dim=0)) \
            .detach().numpy()
        new_connection_pheromone[particle.hidden_nodes_idx][:, particle.in_features_idx] += \
            score * torch.abs(particle.net.sparse.weight).detach().numpy()

        print("{}: {} in features and {} hidden Nodes achieved accuracy {}"
              .format(i, particle.connection_matrix.shape[1], particle.connection_matrix.shape[0], accuracy))

        if accuracy == 0:
            continue

        if score > best_score:
            best_score = score
            best_particle = particle
            best_accuracy = accuracy

    in_feature_pheromone *= pheromone_degradation
    hidden_node_pheromone *= pheromone_degradation
    connection_pheromone *= pheromone_degradation

    in_feature_pheromone += new_in_feature_pheromone
    hidden_node_pheromone += new_hidden_node_pheromone
    connection_pheromone += new_connection_pheromone

    in_feature_pheromone /= np.max(in_feature_pheromone)
    hidden_node_pheromone /= np.max(hidden_node_pheromone)
    connection_pheromone /= np.max(connection_pheromone)

    print("The best net had {} features and {} hidden Nodes with an accuracy of {}"
          .format(best_particle.connection_matrix.shape[1], best_particle.connection_matrix.shape[0],
                  best_accuracy))

    plt.subplot(211)
    plt.bar(np.arange(len(in_feature_pheromone)), in_feature_pheromone)
    plt.title("feature pheromone")
    plt.subplot(212)
    plt.bar(np.arange(len(hidden_node_pheromone)), hidden_node_pheromone)
    plt.title("hidden node pheromone")
    plt.show()

