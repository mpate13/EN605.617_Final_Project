import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# Load binary file (n, dim, labels, features)
# ============================================================
def load_bin(path):
    with open(path, "rb") as f:
        # metadata
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]

        # labels
        labels = np.fromfile(f, dtype=np.int32, count=n)

        # features
        X = np.fromfile(f, dtype=np.float32, count=n * dim)
        X = X.reshape(n, dim)

    return X, labels


# ============================================================
# Load CPU and GPU results
# ============================================================
print("Loading data...")

X_cpu, labels_cpu = load_bin("cpu_features.bin")
X_gpu, labels_gpu = load_bin("gpu_features.bin")

print(f"CPU data shape: {X_cpu.shape}")
print(f"GPU data shape: {X_gpu.shape}")


# ============================================================
# PCA (fit once for consistent comparison)
# ============================================================
print("Running PCA...")

pca = PCA(n_components=2)

X_cpu_2d = pca.fit_transform(X_cpu)
X_gpu_2d = pca.fit_transform(X_gpu)


# ============================================================
# Plot results
# ============================================================
print("Plotting results...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))


# ---------------- CPU plot ----------------
axes[0].scatter(
    X_cpu_2d[:, 0],
    X_cpu_2d[:, 1],
    c=labels_cpu,
    cmap="tab10",
    s=5,
    alpha=0.7
)
axes[0].set_title("CPU K-Means (PCA Projection)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")


# ---------------- GPU plot ----------------
axes[1].scatter(
    X_gpu_2d[:, 0],
    X_gpu_2d[:, 1],
    c=labels_gpu,
    cmap="tab10",
    s=5,
    alpha=0.7
)
axes[1].set_title("GPU K-Means (PCA Projection)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")


plt.tight_layout()
plt.show()