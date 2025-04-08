import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('clustering_outputs/clustering_results_summary.csv')

def plot_metrics_comparison(data, dataset_name):
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
    plt.savefig(f'clustering_outputs/comparison_{dataset_name}.png')
    plt.close()

# Create comparison plots for each dataset
datasets = ['moons', 'circles', 'combined_moons_circles']
for dataset in datasets:
    plot_metrics_comparison(df, dataset)