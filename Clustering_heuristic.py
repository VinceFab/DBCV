#test

"""
@author: Vincent Faber

This code implements a clustering heuristic based on the Density Based
Clusting Validation (DBCV)

Concepts are pulled from the following paper:
Moulavi, Davoud, et al. "Density-based clustering validation."
Proceedings of the 2014 SIAM International Conference on Data Mining.
Society for Industrial and Applied Mathematics, 2014.
"""

import matplotlib.pyplot as plt
import sklearn.datasets
import scipy as sp
import numpy as np
import pandas as pd 
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import scipy
import copy
import DBCV_computation

###############################################################################
#description --> generates and plots a random globular data set
#input --> N: positive integer (sample size)
#          C: positive integer (number of cluster for data set generation)
#          r: positive integer (random state)						  
#output --> X: N by 2 numpy array holding x1 and x2 for each data points
#           Y: N by None numpy array holding the true cluster for each points
def generate_data(N, C, r):
	
	X, Y = sklearn.datasets.make_blobs(n_samples=N, n_features=2, centers=C, random_state = r)
	
	print('Initial data set:')
	plt.figure(figsize=(7, 5))
	plt.scatter(X[:, 0], X[:, 1], marker='o',alpha=0.5)
	plt.show()
	
	return X, Y
###############################################################################

###############################################################################
#description --> computes core distance for all points
#input --> X: a N by 2 numpy array holding the x1, x2 for each points
#output --> core_distances: a list of length N, holding the core distance of each point
def get_core_distances(X, k):
	
	core_distances = []
	for i in range(0,X.shape[0]):
		dist = scipy.spatial.distance.cdist(X[i].reshape((1,2)),X)
		dist[0][i] = np.inf #remove X[i] itself from neighbor candidates
		neighbors = list(np.argsort(dist, axis=1)[:,0:k].reshape(k))
		core_dist = 0
		for pt in neighbors:
			core_dist += (1/sp.spatial.distance.euclidean(X[i],X[pt]))**2
		core_dist = (core_dist/(k))**(float(-1)/float(2))
		core_distances.append(core_dist)
	return core_distances	
###############################################################################

###############################################################################
#description --> computes the Mutual reachability (MR) distance from all point to all points
#input --> X: a N by 2 numpy array holding the x1, x2 for each points
#          core_distances: a list of length N, holding the core distance of each point    
#output --> MR_matrix: a N by N matrix holding the core distance from all point to all points
def get_MR_matrix(X, core_distances):
	
	MR_matrix = np.zeros(shape=(X.shape[0],X.shape[0]))

	for i in range(0,X.shape[0]):
		
		eucli_dist = scipy.spatial.distance.cdist(X[i].reshape((1,2)),X)[0]
		core_dist_i = [core_distances[i]]*X.shape[0]
		core_dist_j = core_distances
		
		Mj = list(map(max, eucli_dist, core_dist_i, core_dist_j))
		
		MR_matrix[i] = Mj
		
	return MR_matrix
###############################################################################

###############################################################################
#description --> computes the minimum spanning tree (MST) of the complete MR-graph of the data
#input --> MR_matrix: a N by N matrix holding the core distance from all point to all points  
#output --> mst: a N by N symmetric matrix representing the MST of the complete MR-graph of the data
def get_mst(mrmatrix):
		
	mst = minimum_spanning_tree(csr_matrix(mrmatrix))
	mst = mst.toarray() # The MST matrix should be symmetric but scipy only keeps one value
	
	def make_symmetric(a):
	    return a+a.T-np.diag(a.diagonal())
	mst = make_symmetric(mst)

	return mst
###############################################################################

###############################################################################
#description --> retrieves the current clusters from a the fully connected subgraphs of 
#                of graph, from its matrix represention
#input --> A: a matrix
#output --> all_clusters: a list of lists. Each sub-list corresponds to a cluster
#                     	  and contains the indices of points within that cluster
def get_clusters(A):
	
	G = nx.from_numpy_matrix(A)
	Subgraphs = list(nx.connected_component_subgraphs(G))
	all_clusters = []
	
	for graph in Subgraphs:
		all_clusters.append(list(graph.nodes()))
	
	return all_clusters
###############################################################################
	
###############################################################################
#description --> assign each point to their respective cluster
#input --> N: an positive integer reprsenting the sample size
#          all_clusters: a list of lists. Each sub-list corresponds to a cluster
#                     	 and contains the indices of points within that cluster
#output --> assigned_cluster: a list of length N holding the cluster for each point 
def report_clusters(N, all_clusters):
	cluster_number = 0
	assigned_cluster = [0]*N
	for cluster in all_clusters:
		for point in cluster:
			assigned_cluster[point] = cluster_number 
		cluster_number += 1
	return assigned_cluster
###############################################################################

###############################################################################
#description --> finds the (i,j) coordinates of the heaviest edge in the current mst
#                while making sure not to create a cluster with <=2 points (such cluster
#                does not have an internal node, and the  DBVC cannot be computed)
#input --> A: a matrix representing a graph
#output --> (i,j): a pair of coordinates for the heaviest edge in A
def get_heaviest_edge(A):
	
	flat = list(set(A.flatten()))
	flat.sort(reverse=True)
	
	condition = False
	m = 0
	while condition == False: #while the cut create a cluster with <=2 points, keep searching
		
		B = copy.deepcopy(A)		
		i, j = np.where(B==flat[m])[0][0], np.where(B==flat[m])[0][1]

		B[i,j], B[j,i] = 0, 0 #delete heaviest edge in curent mst 
		all_clusters = get_clusters(B) #extract clusters

		if all(len(c) > 2 for c in all_clusters):
			condition = True		
		m += 1
			
	return i, j
###############################################################################

###############################################################################
#description --> plots the data in R2 with a color code corresponding to the
# 				   current clustering
#input --> dataframe: pandas dataframe with at least cols x1, x2 and cluster
#output --> none
def plot_clustering(dataframe):
	
	temp = dataframe.copy(deep=True)
	def color(c):
		return chr(int(c+97))
	temp['color'] = list(map(color, temp['cluster']))
		
	plt.figure(figsize=(7, 5))
	plt.scatter(temp['x1'], temp['x2'],c=temp['cluster'], marker='o',alpha=0.5)
	plt.show()

	return
###############################################################################

###############################################################################
#description --> generates random data set and runs clustering heuristic
#input --> N: positive integer (sample size)
#          C: positive integer (number of cluster for data set generation)
#          L: positive integer (maximum number of iterations) 
#          r: positive integer (random state)
#output --> final_df: data frame holding coordinates of points as well as their final cluster assignment
#           final_dbcv: DBCV value of final clustering
def main(N, C, L, r):
	
	X,Y = generate_data(N,C,r) #generates data set
	df = pd.DataFrame() #creates pandas df 
	df['x1'], df['x2'] = zip(*X.tolist()) #stores coordinates 
	df['core'] = get_core_distances(X, k) #stores initial core distances
	MR_matrix = get_MR_matrix(X, np.array(df['core'])) #computes MR matrix
	mst = get_mst(MR_matrix) #computes MST
	all_clusters = get_clusters(mst) #extract clusters
	df['cluster'] = report_clusters(N, all_clusters) #reports cluster assignment to main df
	
	#heuristic
	DBCV_list = []
	p=0
	end=0
	while (p<L and end==0):
		
		print('\nIteration {} -- removing heaviest edge from global MST...'.format(p+1))
		old_df = copy.deepcopy(df) #store current df 
		max_i, max_j = get_heaviest_edge(mst) #get heaviest edge in current mst
		mst[max_i][max_j], mst[max_j][max_i] = 0, 0 #delete heaviest edge in curent mst 
		all_clusters = get_clusters(mst) #extract clusters
		df['cluster'] = report_clusters(X, all_clusters) #update cluster assignment to main df
		
		print('Iteration {} -- Computing updated clustering\'s DBCV...'.format(p+1))
		DBCV_value = DBCV_computation.DBCV(df) #compute dbcv of current clustering
		DBCV_list.append(DBCV_value) #store in list
	
		print('After {} iteration || Number of clusters is: {} || DBCV value is {}'.format(p+1, len(set(list(df['cluster']))), DBCV_value))
		print('Graph:')
		plot_clustering(df) #plot
		
		if p > 0 and DBCV_list[len(DBCV_list)-2] > DBCV_list[len(DBCV_list)-1]:
			end = 1
			print('STOPPING condition --> backtrack to iteration {}'.format(p))
			#if stopping condition is reached, back-track to previous df and dbvi
			final_df = old_df.copy(deep=True)
			final_dbcv = DBCV_list[len(DBCV_list)-2]
		
		p+=1
	
	return final_df, final_dbcv

#run
N=200 #sample size to generate synthetic data
r = 48 #random state to generate synthetic data
C=3 #number of centers to generate synthetic data
k=10 #choose k<<n
L=15 #max number of iterations

final_df, final_dbcv = main(N,C,L,r)
###############################################################################
############################################################################### 
