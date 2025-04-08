#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clustering Analysis Assignment - Comparing Different Clustering Algorithms

This script demonstrates various clustering algorithms on synthetic datasets
and evaluates their performance using multiple metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering,
    MeanShift, SpectralClustering, AffinityPropagation, Birch, OPTICS
)
from sklearn.mixture import GaussianMixture
import hdbscan  # Ensure you have installed the hdbscan package
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time
import os
import csv
import pandas as pd
import seaborn as sns

def generate_datasets(n_samples=1500, random_state=42):
    """Generate synthetic datasets."""
    # Generate individual datasets
    X_moons, y_moons = make_moons(n_samples=n_samples//2, noise=0.05, random_state=random_state)
    X_circles, y_circles = make_circles(n_samples=n_samples//2, factor=0.5, noise=0.05, random_state=random_state)
    
    # Create combined moons and circles dataset
    X_combined = np.vstack([X_moons, X_circles])
    y_combined = np.hstack([y_moons, y_circles + 2])  # Add 2 to circles labels to distinguish from moons
    
    return {
        'moons': (X_moons, y_moons),
        'circles': (X_circles, y_circles),
        'combined_moons_circles': (X_combined, y_combined)
    }

def get_clustering_algorithms():
    """Define clustering algorithms to be tested."""
    return {
        'KMeans': KMeans(n_clusters=3, random_state=42),
        'MiniBatchKMeans': MiniBatchKMeans(n_clusters=3, random_state=42),
        'AffinityPropagation': AffinityPropagation(random_state=42),
        'MeanShift': MeanShift(bandwidth=2),
        'SpectralClustering': SpectralClustering(n_clusters=3, random_state=42),
        'Ward': AgglomerativeClustering(n_clusters=3, linkage='ward'),
        'AgglomerativeClustering': AgglomerativeClustering(n_clusters=3),
        'DBSCAN': DBSCAN(eps=0.3, min_samples=5),
        'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=15),
        'OPTICS': OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1),
        'Birch': Birch(n_clusters=3),
        'GaussianMixture': GaussianMixture(n_components=3, random_state=42)
    }

def evaluate_clustering(X, labels):
    """Evaluate clustering results using multiple metrics."""
    if len(np.unique(labels)) < 2:
        return None, None, None
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)
    return silhouette, calinski, davies

def plot_dataset(X, y, title, ax):
    """Plot a single dataset."""
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y if y is not None else 'blue',
                         s=50, alpha=0.7, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle='--', alpha=0.7)
    return scatter

def save_all_datasets_grid(datasets, output_dir):
    """Save a grid image showing all datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()
    for ax, (name, (X, y)) in zip(axes, datasets.items()):
        ax.scatter(X[:, 0], X[:, 1], c=y if y is not None else 'blue', s=40, alpha=0.7, cmap='viridis')
        ax.set_title(name.capitalize(), fontsize=14)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_datasets_grid.png"))
    plt.close()

def plot_clustering_results(dataset_name, X, all_labels, output_dir):
    """Plot all clustering results for a single dataset in one figure."""
    n_algorithms = len(all_labels)
    n_cols = 3  # Number of columns in the grid
    n_rows = (n_algorithms + n_cols) // n_cols  # Calculate number of rows needed
    
    # Create figure with adjusted size
    plt.figure(figsize=(20, 6 * n_rows))
    
    # Plot original dataset
    plt.subplot(n_rows, n_cols, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c='blue', s=50, alpha=0.7)
    plt.title(f"Original {dataset_name.capitalize()} Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot results for each algorithm
    for idx, (algo_name, result) in enumerate(all_labels.items(), start=2):
        if idx <= n_rows * n_cols:  # Ensure we don't exceed the grid size
            plt.subplot(n_rows, n_cols, idx)
            scatter = plt.scatter(X[:, 0], X[:, 1], 
                                c=result['labels'], 
                                cmap='viridis', 
                                s=50, 
                                alpha=0.7)
            plt.colorbar(scatter)
            plt.title(f"{algo_name}\nClusters: {result['n_clusters']}\n"
                     f"Silhouette: {result['silhouette']:.2f}")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_all_algorithms.png"))
    plt.close()

def save_results_to_csv(all_results, output_dir):
    """Save all clustering results to a CSV file."""
    csv_path = os.path.join(output_dir, "clustering_results_summary.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Dataset', 'Algorithm', 'Time (s)', 'Clusters', 'Silhouette', 'Calinski', 'Davies']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for dataset_name, dataset_results in all_results.items():
            for algo_name, result in dataset_results.items():
                writer.writerow({
                    'Dataset': dataset_name,
                    'Algorithm': algo_name,
                    'Time (s)': f"{result['time']:.4f}",
                    'Clusters': result['n_clusters'],
                    'Silhouette': f"{result['silhouette']:.4f}" if result['silhouette'] is not None else "N/A",
                    'Calinski': f"{result['calinski']:.2f}" if result['calinski'] is not None else "N/A",
                    'Davies': f"{result['davies']:.2f}" if result['davies'] is not None else "N/A"
                })
    print(f"\nResults saved to {csv_path}")

def plot_metrics_comparison(data, dataset_name, output_dir):
    """Create a comparison plot for a specific dataset."""
    # Filter data for the specific dataset
    dataset_df = data[data['Dataset'] == dataset_name]
    
    # Create figure with 3 subplots for each metric
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot Silhouette scores (higher is better)
    sns.barplot(x='Algorithm', y='Silhouette', data=dataset_df, ax=ax1, color='skyblue')
    ax1.set_title(f'Silhouette Score Comparison - {dataset_name} (Higher is better)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot Calinski-Harabasz scores (higher is better)
    sns.barplot(x='Algorithm', y='Calinski', data=dataset_df, ax=ax2, color='lightgreen')
    ax2.set_title(f'Calinski-Harabasz Score Comparison - {dataset_name} (Higher is better)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot Davies-Bouldin scores (lower is better)
    sns.barplot(x='Algorithm', y='Davies', data=dataset_df, ax=ax3, color='salmon')
    ax3.set_title(f'Davies-Bouldin Score Comparison - {dataset_name} (Lower is better)')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{dataset_name}.png'))
    plt.close()

def main():
    output_dir = "clustering_outputs"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # Generate datasets
    print("Generating datasets...")
    datasets = generate_datasets()

    # Save all datasets in a single grid image
    save_all_datasets_grid(datasets, output_dir)
    print("Saved all datasets grid image.")

    # Get clustering algorithms
    algorithms = get_clustering_algorithms()
    all_results = {}

    # Process each dataset
    for dataset_name, (X, y_true) in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        dataset_results = {}

        for algo_name, algorithm in algorithms.items():
            print(f"Running {algo_name}...")
            start_time = time.time()
            if algo_name == 'MeanShift':
                algorithm.fit(X_scaled)
                labels = algorithm.labels_
            else:
                labels = algorithm.fit_predict(X_scaled)
            end_time = time.time()
            execution_time = end_time - start_time
            silhouette, calinski, davies = evaluate_clustering(X_scaled, labels)

            dataset_results[algo_name] = {
                'labels': labels,
                'time': execution_time,
                'silhouette': silhouette if silhouette is not None else 0,
                'calinski': calinski if calinski is not None else 0,
                'davies': davies if davies is not None else 0,
                'n_clusters': len(np.unique(labels))
            }

        # Create combined visualization for this dataset
        plot_clustering_results(dataset_name, X, dataset_results, output_dir)

        all_results[dataset_name] = dataset_results

        # Display evaluation metrics
        print(f"\nEvaluation Metrics for {dataset_name}:")
        print("-" * 80)
        print(f"{'Algorithm':<20} {'Time (s)':<10} {'Clusters':<10} {'Silhouette':<12} {'Calinski':<12} {'Davies':<10}")
        print("-" * 80)
        for algo_name, result in dataset_results.items():
            silhouette = f"{result['silhouette']:.4f}" if result['silhouette'] is not None else "N/A"
            calinski = f"{result['calinski']:.2f}" if result['calinski'] is not None else "N/A"
            davies = f"{result['davies']:.2f}" if result['davies'] is not None else "N/A"
            print(f"{algo_name:<20} {result['time']:.4f}     {result['n_clusters']:<10} {silhouette:<12} {calinski:<12} {davies:<10}")
        print("-" * 80)

    # Save results to CSV
    save_results_to_csv(all_results, output_dir)

    print("\nGenerating comparison plots...")
    
    # Read the CSV file with all results
    results_df = pd.read_csv(os.path.join(output_dir, 'clustering_results_summary.csv'))
    
    # Create comparison plots for each dataset
    for dataset_name in datasets.keys():
        plot_metrics_comparison(results_df, dataset_name, output_dir)
    
    print("Analysis complete. Results and visualizations saved to the 'clustering_outputs' directory.")
    print("Note: Higher Silhouette and Calinski scores are better. Lower Davies score is better.")

if __name__ == "__main__":
    main()
