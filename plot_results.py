import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Phase 1: Block Size Scaling
# -------------------------
df1 = pd.read_csv("phase1_block_scaling.csv")

plt.figure()
plt.plot(df1["block_size"], df1["speedup"], marker='o')
plt.xlabel("Block Size")
plt.ylabel("Speedup (x)")
plt.title("Phase 1: Speedup vs Block Size (N=10000)")
plt.grid()
plt.savefig("phase1_speedup.png")


# -------------------------
# Phase 2: Grid Scaling
# -------------------------
df2 = pd.read_csv("phase2_grid_scaling.csv")

plt.figure()
plt.plot(df2["n"], df2["speedup"], marker='o')
plt.xlabel("Dataset Size (N)")
plt.ylabel("Speedup (x)")
plt.title("Phase 2: Speedup vs Dataset Size")
plt.grid()
plt.savefig("phase2_speedup.png")


# -------------------------
# Phase 3: Mini-Batch Scaling
# -------------------------
df3 = pd.read_csv("phase3_minibatch.csv")

plt.figure()
plt.plot(df3["n"], df3["speedup"], marker='o', color='red')
plt.xlabel("Dataset Size (N)")
plt.ylabel("Speedup (x)")
plt.title("Phase 3: Mini-Batch Speedup")
plt.grid()
plt.savefig("phase3_speedup.png")


# -------------------------
# Phase 2 vs Phase 3 (Best Graph)
# -------------------------
plt.figure()
plt.plot(df2["n"], df2["speedup"], marker='o', label="Standard")
plt.plot(df3["n"], df3["speedup"], marker='o', label="Mini-batch")
plt.xlabel("Dataset Size (N)")
plt.ylabel("Speedup (x)")
plt.title("Standard vs Mini-Batch Speedup")
plt.legend()
plt.grid()
plt.savefig("comparison_speedup.png")


# -------------------------
# Phase 4: Stress Test
# -------------------------
df4 = pd.read_csv("phase4_stress_test.csv")

plt.figure()
plt.plot(df4["block_size"], df4["speedup"], marker='o', color='orange')
plt.xlabel("Block Size")
plt.ylabel("Speedup (x)")
plt.title("Phase 4: Speedup vs Block Size (N=1,000,000)")
plt.grid()
plt.savefig("phase4_speedup.png")


print("Speedup plots generated.")