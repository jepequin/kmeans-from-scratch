import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#return square of euclidean distance between two points
def euclidean(list1,list2):
	return sum([(list1[i]-list2[i])**2 for i in range(len(list1))])

#assign each row in the dataframe to nearest centroid
def assign_cluster(centroids,dataframe):
	clusters = [[] for i in range(len(centroids))]
	for index, row in dataframe.iterrows():
		distances = [euclidean(row,centroid) for centroid in centroids]
		nearest = np.argmin(distances)
		clusters[nearest].append(row.to_numpy())
	return clusters

#calculate centroids from the means of points in clusters
def update_centroids(clusters):
	centroids = [sum(cluster)/len(cluster) for cluster in clusters]
	return centroids

def kmeans(k,csvfile):
	df = pd.read_csv(csvfile)
	#choose k random centroids
	centroids = df.sample(n=k).values.tolist()
	clusters = assign_cluster(centroids,df)
	new_centroids = update_centroids(clusters)
	#measure sum of distances between old and new centroids 
	error = sum([euclidean(centroids[i],new_centroids[i]) for i in range(k)])
	#update centroids until they converge
	while error > 0:
		centroids = new_centroids
		clusters = assign_cluster(centroids,df)
		new_centroids = update_centroids(clusters)
		error = sum([euclidean(centroids[i],new_centroids[i]) for i in range(k)])
	return clusters, centroids

#input number of clusters and calculate clusters and centroids
while True:
    try:
        k = int(input('Set a value for the number of clusters : '))
        break
    except ValueError:
        print("The input must be an integer")
clusters, centroids = kmeans(k,'cloud.txt')

#Plot the clusters with respective centroids
colors = cm.rainbow(np.linspace(0, 1, k))
for i in range(k):
	plt.scatter(*zip(*clusters[i]), color = colors[i])
plt.scatter(*zip(*centroids))
plt.show()