# Clustering Analysis Assignment

This repository contains code and analysis for a machine learning assignment on clustering algorithms. The assignment explores various clustering methods and their performance on different types of datasets.

## Assignment Overview

The assignment involves:

1. Exploring the Scikit-learn clustering comparison example
2. Generating a combined dataset using multiple data generation methods
3. Applying various clustering algorithms to the dataset
4. Evaluating and comparing the performance of different algorithms
5. Selecting and justifying the most effective algorithm

## Files in this Repository

- `clustering_assignment.ipynb`: Jupyter notebook with detailed analysis and visualizations
- `clustering_assignment.py`: Python script version of the same analysis
- `README.md`: This file, providing an overview of the assignment
- `output/`: Directory containing all generated visualization images

## Requirements

To run the code, you need the following Python packages:

- numpy
- matplotlib
- scikit-learn

You can install these packages using pip:

```
pip install numpy matplotlib scikit-learn
```

## How to Run the Code

### Using the Jupyter Notebook

1. Make sure you have Jupyter installed:

```
pip install jupyter
```

2. Start Jupyter:

```
jupyter notebook
```

3. Open `clustering_assignment.ipynb` and run the cells in order.

### Using the Python Script

Simply run the Python script:

```
python clustering_assignment.py
```

The script will create an `output` directory (if it doesn't exist) and save all visualization images there:

- `combined_dataset.png`: Visualization of the combined dataset
- `individual_datasets.png`: Visualizations of the individual dataset components
- `clustering_results.png`: Results from applying different clustering algorithms
- `best_spectral_clustering.png`: Best result from parameter tuning of Spectral Clustering

## Assignment Components

### 1. Introduction to Clustering

The notebook includes a comprehensive introduction to clustering and its importance in machine learning, covering:

- Definition and purpose of clustering
- Types of clustering algorithms
- Applications of clustering in real-world scenarios

### 2. Summary of Clustering Methods

The analysis covers several clustering algorithms:

- K-Means
- DBSCAN
- Agglomerative Clustering
- Mean Shift
- Spectral Clustering
- And others

For each algorithm, the strengths, limitations, and specific use cases are discussed.

### 3. Dataset Generation and Visualization

The code generates a combined dataset using:

- `make_blobs`: Creates spherical clusters
- `make_moons`: Creates moon-shaped clusters
- `make_circles`: Creates circular clusters

These datasets are combined to create a complex dataset with different types of structures.

### 4. Clustering Analysis and Results

The analysis includes:

- Application of multiple clustering algorithms
- Evaluation using metrics such as silhouette score, Calinski-Harabasz index, and Davies-Bouldin index
- Visual comparison of clustering results
- Parameter tuning for the selected algorithm

### 5. Algorithm Selection and Justification

The notebook provides a detailed justification for selecting the most effective clustering algorithm based on:

- Performance metrics
- Visual inspection of clusters
- Ability to handle different types of cluster shapes

## Conclusion

The assignment demonstrates the importance of understanding both the characteristics of the data and the strengths/limitations of different clustering algorithms. By combining multiple data generation methods and evaluating various clustering approaches, we gain insights into their applicability to different types of datasets.
