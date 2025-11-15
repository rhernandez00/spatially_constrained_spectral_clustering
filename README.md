# Spatially Constrained Spectral Clustering

Repository for the paper: [Full paper title]
DOI: [paper.doi]

Overview

This repository contains the code used to generate the multi-scale functional dog brain atlas described in the accompanying paper. The workflow applies spatially constrained spectral clustering (Craddock et al., 2012) to resting-state fMRI data to create parcellations ranging from 20 to 300 clusters, ensuring both functional homogeneity and spatial contiguity.

The repository is primarily notebook-driven, and each notebook corresponds to one part of the analysis in the manuscript:

atlas generation

reproducibility and cross-dataset validation

homogeneity evaluation

figure generation

# In this code:

All steps follow exactly the pipeline described in the paper and can be reproduced end-to-end using the notebooks.

1. Preprocessing & Input Handling

The notebooks assume that the user provides:

a 4D fMRI time-series already preprocessed and registered to a common template

a voxel adjacency matrix or 3D grid neighbourhood structure

optional masks, metadata, or external parcellations for comparison

Preprocessing follows the steps described in the manuscript (motion correction, slice timing, alignment to the template, masking), but is not performed directly in this repository.

2. Spatially Constrained Spectral Clustering

The clustering implementation follows the algorithm by Craddock et al. (2012):

Compute voxel-wise time-series correlations to create a functional similarity graph.

Construct a spatial adjacency matrix enforcing contiguity through a nearest-neighbour or grid-based neighbourhood.

Combine functional and spatial constraints using the method described in the paper.

Build the graph Laplacian, compute its eigendecomposition, and embed voxels in spectral space.

Apply k-means to obtain parcellations at resolutions from N = 20 to N = 300.

This entire process is implemented directly inside calculate_segmentation.ipynb, which reproduces the atlas used in the manuscript.

3. Evaluation Metrics

Two notebooks reproduce the manuscript’s results:

Reproducibility & Cross-Dataset Replication

Figures_Reproducibility.ipynb computes:

Dice coefficient

Adjusted Rand Index (ARI)

cross-dataset reproducibility using the external dataset

within-participant reproducibility using the coil-comparison dataset

The notebook replicates Figures 1B–1E from the paper.

Functional Homogeneity

Figures_Homogeneity.ipynb computes:

parcel-wise homogeneity

average homogeneity across resolutions

homogeneity maps for visualization

These analyses correspond to Figure 2 of the paper.

4. Anatomical Concordance (Optional)

If atlas files are included, a comparison with the Johnson et al. anatomical atlas can be reproduced.
Otherwise, users may insert their own reference atlas.

5. Figure Generation

All figure panels in the manuscript are reproduced directly in the notebooks, using:

Nilearn

Matplotlib

Nibabel

Scikit-learn

Every figure in the paper has a corresponding code cell inside the notebooks.

Citation

If you use this repository or atlas, please cite the manuscript:

[Full citation here]
DOI: [paper.doi]

📂 Data Availability

The atlas and data used in the paper are available at:

Zenodo: []
External dataset: Beckmann et al., 2021 (OpenNeuro ds003830)