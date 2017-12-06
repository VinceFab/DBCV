# -*- coding: utf-8 -*-
"""
@author: Faber Vincent

This script takes a pandas dataframe as input (indicating points coordinates and 
cluster assignments) and returns the corresponding Density Based Clustering 
Validation value.

The code follows this paper:
Moulavi, Davoud, et al. "Density-based clustering validation."
Proceedings of the 2014 SIAM International Conference on Data Mining.
Society for Industrial and Applied Mathematics, 2014.
"""

import scipy as sp
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import scipy

###############################################################################
#description --> computes core distance for all points
#input --> X: a N by 2 numpy array holding the x1, x2 for each points
#output --> core_distances: a list of length N, holding the core distance of each point
def get_core_distances(X):
	
	core_distances = []
	for i in range(0, X.shape[0]):
		core_dist = 0
		for j in range(0, X.shape[0]):
			if j != i:			
				core_dist += (1/sp.spatial.distance.euclidean(X[i],X[j]))**2

		core_dist = (core_dist/(X.shape[0]-1))**(float(-1)/float(2))		
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
#description --> gets density sparseness of cluster, using its matrix representation
#                Note that we only consider internal nodes (in matrix representation,
#                internal nodes have rows with >=2 non-zero entries) 
#input --> A: a N by N matrix 
#output --> np.max(A): the value of the heaviest entry in A
def get_density_sparseness(A):
	
	maximum = -np.inf
	for i in range(0, A.shape[0]):
		if np.count_nonzero(A[i]) <= 1: #i.e. if A[i] is a leaf 
			continue
		else:
			maximum = max(maximum, A[i].max())
	
	return np.max(A)
###############################################################################

###############################################################################
#description --> gets density separation of a pair of clusters
#input --> DICO: dictionary holding information about each cluster
#          cluster_a: a positive integer representing a cluster number
#          cluster_b: a positive integer representing a cluster number
#output --> minimum: a float; the smallest MR distance between internal nodes of cluster_a and cluster_b
def DSPC(DICO, cluster_a, cluster_b):
	
	X_a = np.array(DICO['cluster{}'.format(cluster_a)]['X'].reset_index(drop=True)[['x1', 'x2']])
	X_b = np.array(DICO['cluster{}'.format(cluster_b)]['X'].reset_index(drop=True)[['x1', 'x2']])

	core_a = np.array(DICO['cluster{}'.format(cluster_a)]['X'].reset_index(drop=True)[['core']])
	core_b = np.array(DICO['cluster{}'.format(cluster_b)]['X'].reset_index(drop=True)[['core']])

	mst_a = DICO['cluster{}'.format(cluster_a)]['mst']
	mst_b = DICO['cluster{}'.format(cluster_b)]['mst']

	
	minimum = np.inf
	for i in range(0, X_a.shape[0]):
		if np.count_nonzero(mst_a[i]) <= 1: #i.e. if X_a[i] is a leaf 
			continue
		else:
			for j in range(0, X_b.shape[0]):
				if np.count_nonzero(mst_b[j]) <= 1: #i.e. if X_b[j] is a leaf 
					continue
				else:
					eucli_dist = scipy.spatial.distance.cdist(X_a[i].reshape((1,2)), X_b[j].reshape((1,2)))[0]		
					core_dist_a = core_a[i]
					core_dist_b = core_b[j]
					mr_dist = max(eucli_dist, core_dist_a,core_dist_b)
					
					minimum = min(minimum, mr_dist)
					
	return minimum
###############################################################################

###############################################################################
#description --> gets validity index of a cluster
#input --> DICO: dictionary holding information about each cluster
#          cluster: a positive integer representing a cluster number
#output --> num/den: a float; validity index of a cluster
def VC(DICO, cluster):
	num = min(DICO['cluster{}'.format(cluster)]['dspcs']) - DICO['cluster{}'.format(cluster)]['dsc']
	den = max(min(DICO['cluster{}'.format(cluster)]['dspcs']), DICO['cluster{}'.format(cluster)]['dsc'])
	return num/den
###############################################################################

###############################################################################
#description --> computes the DBCV value of the current clustering
#input --> df: pandas dataframe with at least cols x1, x2 and cluster
#output --> DBCV: a float; DBCV of current clustering
def DBCV(df):
		
	nb_clusters = len(set(list(df['cluster']))) #number of distinct clusters
	DICO = {}
	
	for c in range(0, nb_clusters):
		
		cluster_df = df.loc[df['cluster']==c]
		X = np.array(cluster_df[['x1', 'x2']])
		core_distances = get_core_distances(X)
		mr_matrix = get_MR_matrix(X, core_distances)
		mst = get_mst(mr_matrix)
    	#interior_mst = get_interior_mst(mst)
		dsc = get_density_sparseness(mst)
		
		sub_dico = {}
		sub_dico['X'] = cluster_df
		sub_dico['core_distances'] = core_distances
		sub_dico['mr_matrix'] = mr_matrix
		sub_dico['mst'] = mst
		#sub_dico['interior_mst'] = interior_mst
		sub_dico['dsc'] = dsc
		
		DICO['cluster{}'.format(c)] = sub_dico
	
	for cluster_a in range(0, nb_clusters):
		dspcs = []
		for cluster_b in range(0, nb_clusters):
			if cluster_a != cluster_b:
				dspcs.append(DSPC(DICO, cluster_a, cluster_b))
		DICO['cluster{}'.format(cluster_a)]['dspcs'] = dspcs
		
	for c in range(0, nb_clusters):
		DICO['cluster{}'.format(c)]['vc'] = VC(DICO, c)
	
	
	DBCV = 0
	for c in range(0, nb_clusters):
		DBCV += DICO['cluster{}'.format(c)]['vc'] * (float(DICO['cluster{}'.format(c)]['X'].shape[0]) / float(df.shape[0]))

	return DBCV
###############################################################################
	