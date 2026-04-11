import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def compare_results(cpu_csv, gpu_csv):
    # Load the datasets
    cpu_df = pd.read_csv(cpu_csv).rename(columns={'ClusterID': 'CPU_Cluster'})
    gpu_df = pd.read_csv(gpu_csv).rename(columns={'ClusterID': 'GPU_Cluster'})
    
    # Merge on ImageID to ensure we are comparing the same images
    comparison_df = pd.merge(cpu_df, gpu_df, on='ImageID')
    
    # Calculation of Accuracy (Match Rate)
    matches = (comparison_df['CPU_Cluster'] == comparison_df['GPU_Cluster']).sum()
    total = len(comparison_df)
    accuracy = (matches / total) * 100
    
    print(f"--- Comparison: {cpu_csv} vs {gpu_csv} ---")
    print(f"Total Points Compared: {total}")
    print(f"Exact Matches:         {matches}")
    print(f"Match Rate (Accuracy): {accuracy:.2f}%")

    # Confusion Matrix (Where did the GPU diverge?)
    # This helps see if certain clusters are being swapped or merged
    plt.figure(figsize=(10, 8))
    confusion_matrix = pd.crosstab(comparison_df['CPU_Cluster'], 
                                   comparison_df['GPU_Cluster'])
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f"Cluster Assignment Match Matrix\nAccuracy: {accuracy:.2f}%")
    plt.xlabel("GPU Cluster ID")
    plt.ylabel("CPU Cluster ID")
    
    output_name = f"comparison_{accuracy:.0f}pct.png"
    plt.savefig(output_name)
    print(f"Visual matrix saved as: {output_name}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare.py <cpu_results.csv> <gpu_results.csv>")
    else:
        compare_results(sys.argv[1], sys.argv[2])