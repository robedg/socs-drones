from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

with open("data/p22.txt", "r") as f:
    lines = f.readlines()

customers = []
centers = []
minimum_index = 0
max_clusters = 10

for line in lines[1:]:
    coord = line.strip().split(" ")
    customers.append((int(coord[0]),int(coord[1])))

distance_sums = [0]*max_clusters

for i in range(1,max_clusters+1):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(customers)
    for c, l in enumerate(kmeans.labels_):
        dist = math.dist(customers[c], kmeans.cluster_centers_[l])
        if dist <= 150:
            distance_sums[i-1] += dist
        else:
            distance_sums[i-1] += 999999
    centers.append(kmeans.cluster_centers_) 

for index in range(len(distance_sums)):
    if distance_sums[index] + (index + 1)*2500 < distance_sums[minimum_index] + (minimum_index + 1)*2500:
        minimum_index = index

print("Optimum number of depots: " + str(minimum_index+1))
print("The location of depots:")
for c in centers[minimum_index]:
    print(c)