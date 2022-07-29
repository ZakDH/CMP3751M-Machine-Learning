import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_euclidean_distance(vec_1, vec_2):
    #calculates the euclidean distance between the centroids
    distances = ((vec_1 - vec_2) ** 2).sum()**0.5
    return distances

def initialize_centroids(data, k):
    #Initializes clusters randomly across the datapoints in the dataset
    centroids = {}
    for i in range(k):
        centroids[i] = data[i]
    return centroids

def kmeans(data, k):
    #assigns 'X' with the first and second columns in the dataset
    X = data.iloc[:, [0, 1]].values
    #assigns 'X2' with the first the third columns in the dataset
    X2 = data.iloc[:, [0, 2]].values  
    datasets = [X,X2]
    for dataset in datasets:
        #calls the centroid function to randomly assign the centroids
        centroids = initialize_centroids(dataset,k)
        for i in range(k):
            clusters_assigned = {}
            for i in range(k):
                    #assigns the number of clusters that corresponds with 'i' - maximum 3
                    clusters_assigned[i] = []
            for features in dataset:
                #calls the euclidean distance function to compute the distance between the features and centroids
                distances = []
                for centroid in centroids:
                    distances.append(compute_euclidean_distance(features,centroids[centroid]))
                #assigns clusters to closest centroids
                clusters = distances.index(min(distances))
                clusters_assigned[clusters].append(features)
            for clusters in clusters_assigned: 
                #loops through centroids in each cluster assignment and computes the average distance to replace centroids
                centroids[clusters] = np.average(clusters_assigned[clusters],axis=0)
        for centroid in centroids:
            #scatters the centroids across a graph and assigns a colour
            plt.scatter(centroids[centroid][0], centroids[centroid][1],marker="x", color="r", linewidths=5)
        for clusters in clusters_assigned:
            color = colors[clusters]
            #loops through the datapoints in each cluster 
            for features in clusters_assigned[clusters]:        
                plt.scatter(features[0], features[1], color=color, alpha=0.5)
        plt.show()
        distance_list =[]
        #resets the totaldistance variable for each iteration
        totaldistance = 0  
        #loops through each feature in each cluster
        for features in clusters_assigned[clusters]:
            #Euclidean distance between the cluster features and the centroid is computed
            totaldistance += compute_euclidean_distance(features,centroids[centroid])
            #computed distance is appended to a list
            distance_list.append(totaldistance)
        #elements in the list are averaged
        finaldistance = np.average(distance_list)
    #averaged elements are appended to the error function list
    error_function.append(finaldistance)
    return centroids,clusters_assigned

data = pd.read_csv('Task2 - dataset - dog_breeds.csv')
#np.random.shuffle(data) #shuffles rows
colors = ['k', 'b', 'g','k', 'b', 'g']
error_function = []
for k in range(1,6):
    centroids,clusters_assigned = kmeans(data,k)
#average distances of clusters to centroids are plotted
plt.plot(error_function, color='magenta', marker='o')
#x-axis is set to the range of clusters
plt.xticks(range(k))
plt.ylabel("objective function value")
plt.xlabel("iteration step")
plt.title('Objective error function')
plt.show()