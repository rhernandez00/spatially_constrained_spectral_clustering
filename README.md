# spatially_constrained_spectral_clustering

Spatially Constrained Spectral Clustering

Repository for the paper: [Full paper title]
DOI: [paper.doi]

Overview

This repository contains the full implementation of the spatially constrained spectral clustering algorithm introduced in the accompanying paper. The method extends classical spectral clustering by enforcing spatial contiguity, producing clusters that are both functionally/feature-homogeneous and spatially connected.

The workflow combines:

- a feature-based similarity matrix,

- a spatial adjacency/contiguity constraint,

- a joint graph Laplacian embedding, and

- k-means clustering in spectral space.

This implementation corresponds to the approach described in the paper and includes scripts for running clustering, preprocessing data, and evaluating cluster quality.

What the Code Does:
1. Data Loading & Preprocessing

Loads feature matrices (e.g., fMRI timeseries, tabular features, etc.).

Loads spatial coordinates and/or adjacency matrices.

Computes the feature similarity graph (RBF kernel, k-NN graph, etc.).

Builds a spatial adjacency kernel based on neighbourhoods or distances.

2. Spatial Constraint Kernel

Constructs a spatial contiguity matrix from adjacency information.

Allows tuning of neighbourhood radius / adjacency threshold.

Optional binarisation or weighting strategies.

3. Combined Affinity / Laplacian

Merges the feature similarity and spatial kernels (e.g., via element-wise multiplication).

Constructs graph Laplacians (L, normalized Laplacian, etc.).

Computes the first k eigenvectors for the spectral embedding.

4. Clustering

Runs k-means on the spectral embedding.

Produces spatially contiguous cluster assignments.

Includes optional post-processing to ensure connectivity.

5. Evaluation

Computes homogeneity / within-cluster similarity.

Computes contiguity measures.

Outputs cluster label maps, logs, and parameter summaries.
