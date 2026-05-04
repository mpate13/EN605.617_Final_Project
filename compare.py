import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from scipy.optimize import linear_sum_assignment

def compare_results(cpu_csv, gpu_csv):
    cpu_df = pd.read_csv(cpu_csv).rename(columns={'ClusterID': 'CPU_Cluster'})
    gpu_df = pd.read_csv(gpu_csv).rename(columns={'ClusterID': 'GPU_Cluster'})
    
    # 1. Create the Contingency Matrix (the data used for your heatmap)
    # Rows = CPU, Cols = GPU
    contingency = pd.crosstab(cpu_df['CPU_Cluster'], gpu_df['GPU_Cluster'])
    
    # 2. Use Hungarian Algorithm to find the optimal re-labeling
    # We want to maximize matches, so we negate the matrix
    row_ind, col_ind = linear_sum_assignment(-contingency.values)
    
    # 3. Create a mapping dictionary (GPU ID -> CPU ID)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    
    # 4. Apply mapping to GPU labels
    gpu_df['GPU_Cluster_Remapped'] = gpu_df['GPU_Cluster'].map(mapping)
    
    # 5. Calculate "True" Accuracy
    matches = (cpu_df['CPU_Cluster'] == gpu_df['GPU_Cluster_Remapped']).sum()
    total = len(cpu_df)
    
    print(f"--- Corrected Accuracy (Permutation Invariant) ---")
    print(f"Match Rate: {(matches / total) * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare.py <cpu_results.csv> <gpu_results.csv>")
    else:
        compare_results(sys.argv[1], sys.argv[2])