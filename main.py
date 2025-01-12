from Fire_Map import Generate_Fire_Map
from K_Clusters import K_Clusters
import numpy as np
import matplotlib.pyplot as plt
from random import choices
from time import time

# Generate a fire map
fireMap = Generate_Fire_Map(seed=0, precision=200)

clusterModel = K_Clusters(
    trainingMatrix=fireMap,
    kGroups=5,
    distanceStrategy="euclidean",
    centroidStrategy="mean",
    maxEpochs=100
)

# Train the model
startTime = time()
clusterModel.Train()
endTime = time()
print(f"Training time: {endTime - startTime} seconds")

# Plot the clusters
figure, axis = plt.subplots()
colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "black"]

for index, cluster in enumerate(clusterModel.clusters):
    for vector in choices(cluster.vectors, k=100): # Randomly sample 100 vectors from the cluster
        axis.scatter(vector[0], vector[1], c=colors[index])

plt.show()