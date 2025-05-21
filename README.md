# Clustering Algorithm Comparison

![image](https://github.com/user-attachments/assets/7bab2d99-0d49-482f-a6d6-d95fd7335e17)

This repository contains a comprehensive analysis and comparison of various clustering algorithms across multiple synthetic datasets. The project evaluates how different clustering techniques perform when faced with diverse data structures and geometries.

## 📊 Project Overview

This research compares the performance of **9 popular clustering algorithms** across **6 different synthetic datasets** using **3 evaluation metrics**. The goal is to identify which algorithms excel under specific data conditions and provide insights into their strengths and limitations.

### Algorithms Analyzed:
- KMeans
- MiniBatchKMeans
- AffinityPropagation
- MeanShift
- SpectralClustering
- Agglomerative Clustering (Ward and Average linkage)
- DBSCAN
- HDBSCAN
- OPTICS
- Birch
- GaussianMixture

### Datasets Used:
- **Blobs**: Simple Gaussian clusters with equal variance
- **Moons**: Two interleaving half-circles
- **Circles**: Concentric circular clusters
- **Aniso**: Anisotropic blobs with directional elongation
- **Varied**: Clusters with different variances and densities
- **No Structure**: Random points with no inherent clustering
- **Custom Combined**: Hybrid of Moons and Circles datasets

### Evaluation Metrics:
- **Silhouette Score**: Measures cluster cohesion and separation
- **Calinski-Harabasz Index**: Evaluates ratio of between-cluster to within-cluster dispersion
- **Davies-Bouldin Score**: Assesses average similarity between clusters

## 🖼️ Key Visualizations

### Dataset Structures
![all_datasets_grid](https://github.com/user-attachments/assets/20f2416d-6be8-473e-958c-562c17adeee7)

### Algorithm Performance on Moons Dataset
![moons_all_algorithms](https://github.com/user-attachments/assets/82e843b1-313f-4b74-ae92-365b01dc3c00)

### Algorithm Performance on Circles Dataset
![circles_all_algorithms](https://github.com/user-attachments/assets/b6a2a356-b114-4ee3-a34f-0fe22c691825)

### Algorithm Performance on Combined Dataset
![combined_moons_circles_all_algorithms](https://github.com/user-attachments/assets/db2d7ed4-3e6e-4429-95b8-0ff418966680)

### Performance Comparison Across All Datasets
![best_algorithm_comparison](https://github.com/user-attachments/assets/fe157c8e-11a6-4f01-99e9-f956393f42ee)

![metrics_comparison_circles](https://github.com/user-attachments/assets/0ce496ad-0ce8-4c8d-881a-a87c4003e0a7)

![metrics_comparison_combined_moons_circles](https://github.com/user-attachments/assets/01fe3531-3cb4-4a46-95d7-4f09562c7403)

![metrics_comparison_moons](https://github.com/user-attachments/assets/e820244f-874f-480e-89d9-ba2f2a9c9cfe)

## 📈 Key Findings

| Dataset | Best Algorithm(s) | Notes |
|---------|-------------------|-------|
| Moons | KMeans (0.67), Spectral (0.56) | Clean, compact clusters |
| Circles | MiniBatchKMeans (0.67), AffinityPropagation | Well-separated circular structures |
| Combined | KMeans (0.67), Spectral (0.64), Agglomerative (0.65) | Strong performance on complex shapes |

- **KMeans** proves to be the most consistent performer across all datasets
- **Spectral Clustering** and **Agglomerative Clustering** excel at handling complex, non-convex shapes
- **DBSCAN** performs well on moon-like data but struggles with parameter sensitivity
- **HDBSCAN** and **OPTICS** generally underperform across most datasets

## 🛠️ Requirements

To run the code, you need the following Python packages:
```
numpy
matplotlib
scikit-learn
pandas
seaborn
```

You can install these packages using pip:
```bash
pip install numpy matplotlib scikit-learn pandas seaborn
```

## 📂 Repository Structure

```
.
├── clustering_comparison.ipynb   # Jupyter notebook with detailed analysis
├── clustering_comparison.py      # Python script version of the analysis
├── output/                       # Directory containing visualization images
│   ├── datasets.png              # Visualizations of all datasets
│   ├── moons_results.png         # Results on Moons dataset
│   ├── circles_results.png       # Results on Circles dataset
│   └── combined_results.png      # Results on Combined dataset
├── data/                         # Optional data files
└── README.md                     # This file
```

## 🚀 How to Run

### Using the Jupyter Notebook
1. Ensure you have Jupyter installed:
```bash
pip install jupyter
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Open `clustering_comparison.ipynb` and run the cells in order.

### Using the Python Script
Simply run the Python script:
```bash
python clustering_comparison.py
```
