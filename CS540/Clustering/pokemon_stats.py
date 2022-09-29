import csv
import random
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance
import numpy as np
import sys
import matplotlib.pyplot as plt

def load_data(filepath):
    with open(filepath, "r") as file:
        reader = csv.DictReader(file)
        pokemonList = [{k: v for k, v in row.items()} for row in reader]
        for pokemon in pokemonList:
            del pokemon['Generation']
            del pokemon['Legendary']
            pokemon['#'] = int(pokemon['#'])
            pokemon['Total'] = int(pokemon['Total'])
            pokemon['HP'] = int(pokemon['HP'])
            pokemon['Attack'] = int(pokemon['Attack'])
            pokemon['Defense'] = int(pokemon['Defense'])
            pokemon['Sp. Atk'] = int(pokemon['Sp. Atk'])
            pokemon['Sp. Def'] = int(pokemon['Sp. Def'])
            pokemon['Speed'] = int(pokemon['Speed'])
    pokemonList = pokemonList[0:20]
    return pokemonList

def calculate_x_y(stats):
    offensive_strength = stats['Attack'] + stats['Sp. Atk'] + stats['Speed']
    defensive_strength = stats['Defense'] + stats['Sp. Def'] + stats['HP']
    return (offensive_strength,defensive_strength)

def euclidean_distance(node1, node2):
    return distance.euclidean(node1, node2)

def hac(dataset):
    final_output = [[0] * 4 for i in range(len(dataset) - 1)]
    clusterList = []
    for i in range(len(dataset)):
        # [x, y, orig_cluster_num]
        clusterList.append({'index': i, "components": [dataset[i]], "clustered": False, "size": 1})

    for row in final_output:
        closest = sys.maxsize
        firstCluster = None
        secondCluster = None
        for cluster in clusterList:
            if cluster["clustered"] is False:
                # go through all component clusters of current cluster
                for component in cluster["components"]:
                    # go through all other clusters to compare to current cluster
                    for comparison_cluster in clusterList:
                        # only check unclustered clusters
                        if comparison_cluster["clustered"] is False:
                            # go through components of all other unclustered clusters
                            for comparison_component in comparison_cluster["components"]:
                                distance = euclidean_distance(component, comparison_component)
                                # make sure you are not comparing the same clusters
                                if (cluster["index"] != comparison_cluster["index"]) and distance < closest:
                                    closest = distance
                                    row[2] = distance
                                    row[3] = cluster['size'] + comparison_cluster['size']
                                    # decide which cluster comes first in output row
                                    if cluster["index"] < comparison_cluster["index"]:
                                        firstCluster = cluster
                                        secondCluster = comparison_cluster
                                        row[0] = cluster["index"]
                                        row[1] = comparison_cluster["index"]
                                    else:
                                        firstCluster = comparison_cluster
                                        secondCluster = cluster
                                        row[0] = comparison_cluster["index"]
                                        row[1] = cluster["index"]
        # At this point we know the two closest unclustered clusters
        firstCluster['clustered'] = True #check if this reference changes original cluster's value
        secondCluster['clustered'] = True
        # Add new cluster to cluster list and concatenate component clusters
        clusterList.append({'index': len(clusterList),
                            'components': firstCluster["components"] + secondCluster["components"],
                            'clustered': False,
                            'size': firstCluster['size'] + secondCluster['size']})

    return np.array(final_output)


def random_x_y(m):
    observation_vectors = []
    for i in range(m):
        x = random.randrange(0,360)
        y = random.randrange(0,360)
        vectTup = (x,y)
        observation_vectors.append(vectTup)
    return observation_vectors

def imshow_hac(dataset):
    # Make scatter plot with initial points
    plt.ion() # make plot interactive
    figure, axes = plt.subplots()
    x_values = [point[0] for point in dataset]
    y_values = [point[1] for point in dataset]
    scatterPlot = axes.scatter(x_values,y_values)


    final_output = [[0] * 4 for i in range(len(dataset) - 1)]
    clusterList = []
    for i in range(len(dataset)):
        # [x, y, orig_cluster_num]
        clusterList.append({'index': i, "components": [dataset[i]], "clustered": False, "size": 1})

    for row in final_output:
        closest = sys.maxsize
        firstCluster = None
        secondCluster = None
        for cluster in clusterList:
            if cluster["clustered"] is False:
                # go through all component clusters of current cluster
                for component in cluster["components"]:
                    # go through all other clusters to compare to current cluster
                    for comparison_cluster in clusterList:
                        # only check unclustered clusters
                        if comparison_cluster["clustered"] is False:
                            # go through components of all other unclustered clusters
                            for comparison_component in comparison_cluster["components"]:
                                distance = euclidean_distance(component, comparison_component)
                                # make sure you are not comparing the same clusters
                                if (cluster["index"] != comparison_cluster["index"]) and distance < closest:
                                    closest = distance
                                    row[2] = distance
                                    row[3] = cluster['size'] + comparison_cluster['size']
                                    # decide which cluster comes first in output row
                                    if cluster["index"] < comparison_cluster["index"]:
                                        firstCluster = cluster
                                        secondCluster = comparison_cluster
                                        row[0] = cluster["index"]
                                        row[1] = comparison_cluster["index"]
                                    else:
                                        firstCluster = comparison_cluster
                                        secondCluster = cluster
                                        row[0] = comparison_cluster["index"]
                                        row[1] = cluster["index"]
        # At this point we know the two closest unclustered clusters
        firstCluster['clustered'] = True #check if this reference changes original cluster's value
        secondCluster['clustered'] = True
        # Add new cluster to cluster list and concatenate component clusters
        clusterList.append({'index': len(clusterList),
                            'components': firstCluster["components"] + secondCluster["components"],
                            'clustered': False,
                            'size': firstCluster['size'] + secondCluster['size']})

        # Plot the line between the two clusters
        xFirst, xSecond = firstCluster["components"][0][0], secondCluster["components"][0][0]
        yFirst, ySecond = firstCluster["components"][0][1], secondCluster["components"][0][1]
        x = [xFirst, xSecond]
        y = [yFirst, ySecond]
        plt.plot(x,y)
        plt.pause(0.1) # wait 0.1 seconds before plotting next line

    return np.array(final_output)


"""
def test_script():
    data = load_data("./Pokemon.csv")
    forest = []
    for point in data:
        forest.append(calculate_x_y(point))
    forest = random_x_y(20)
    #verified = linkage([(1,0), (2,0), (4,0), (5,0), (7.25,0)])
    verified = linkage(forest)
    print(verified)
    print("---------------------------------")
    #mine = hac([(1,0), (2,0), (4,0), (5,0), (7.25,0)])
    mine = hac(forest)
    print(mine)
    print("---------------------------------")
    #mine_imshow = imshow_hac([(1,0), (2,0), (4,0), (5,0), (7.25,0)])
    mine_imshow = imshow_hac(forest)
    print(mine_imshow)
"""
