# Non-Parametric Density Based Clustering Heuristic

## Introduction 

The Density-Based Clustering Validation (DBCV), as developed by Davoud Moulavi Et al(1) in this [paper](http://epubs.siam.org/doi/pdf/10.1137/1.9781611973440.96), provides a good way of measuring the quality of a clustering, based on within-cluster density and between-cluster density connectedness. Through a newly defined kernel density function, such index efficiently assesses the quality of a given clustering.

Though the DBCV allows to accurately rate a given clustering solution, it does not provide an algorithm allowing to find such a clustering. In this short [research project](https://github.com/VinceFab/Non-Parametric-Clustering-Heuristic/blob/master/Non%20parametric%20Density%20Based%20Clustering%20Heuristic.pdf), we present a simple clustering heuristic, inspired from the DBCV methodology, that relies on the density sparseness inside a given cluster as well as the separation between distinct clusters, in the mutual reachability space.

Interestingly, this heuristic is basically non-parametric and does not require making any of the usual clustering assumption about the data (e.g. number of clusters). Empirically, when the data is not too complex, the clustering obtained at the very iteration of the algorithm at which the DBCV is maximized often matches the ground truth clustering used to generate the data.
We therefore use this fact to design a stopping condition for our heuristic, that indicates that the optimal clustering has been reached. Such results call for further theoretical research regarding the convexity of the DBCV values as the algorithm iterates. In some cases however, such stopping rule causes the algorithm to terminate too early or too late, and to return a sub-optimal clustering, and further work is needed to design a better stopping order.

## Files

* [Non parametric Density Based Clustering Heuristic.pdf](https://github.com/VinceFab/DBCV/blob/master/Non%20parametric%20Density%20Based%20Clustering%20Heuristic.pdf): describes our approach and walk through the heuristic
* [Clustering_heuristic.py](Clustering_heuristic.py): python implementation of the heuristic on 2-dimensional synthetic data
* [DBCV_computation.py](DBCV_computation.py): python script to compute the DBCV of a given clustering

## Citation
* (1) Davoud Moulavi, Pablo A. Jaskowiak, Ricardo J. G. B. Campello, Arthur Zimekz, Jorg Sander.
"Density-Based Clustering Validation" 
textit{SIAM} - 2014 | [paper](http://epubs.siam.org/doi/pdf/10.1137/1.9781611973440.96)
