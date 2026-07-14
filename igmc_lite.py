import numpy as np
import matplotlib
# Force Matplotlib to use a headless background generator so it runs perfectly in the cloud
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

print("🧬 Step 1: Simulating Single-Cell RNA-seq Data...")
np.random.seed(42)
# 300 cells total: 2 common types (140 each), 1 rare type (20 cells)
X_raw, y_true = make_blobs(n_samples=[140, 140, 20], n_features=2, center_box=(0, 20), cluster_std=1.5)
X_counts = np.abs(np.round(X_raw)).astype(int) # Convert to integer gene counts

print("📏 Step 2: Applying the Smart Ruler (Information-Geometric Intuition)...")
def smart_distance(cell1, cell2):
    # Variance increases with the mean in biological count data
    mean_expr = (cell1 + cell2) / 2.0 + 1e-5
    variance = mean_expr + 0.1 * (mean_expr ** 2) 
    # Scale differences by noise variance to prioritize quiet, rare signals
    return np.sum(((cell1 - cell2) ** 2) / variance)

# Build a customized similarity matrix
n_cells = len(X_counts)
W = np.zeros((n_cells, n_cells))
for i in range(n_cells):
    for j in range(i, n_cells):
        dist = smart_distance(X_counts[i], X_counts[j])
        sim = np.exp(-dist / 10.0) # Convert distance to similarity
        W[i, j] = sim
        W[j, i] = sim

print("💧 Step 3: Running the Ink Drop Test (Markov Stability Transition)...")
# Normalize rows to create transition probabilities for a random walker
row_sums = W.sum(axis=1)
P = W / row_sums[:, np.newaxis]

# Simulate ink spreading at timescale t=2
P_t = np.linalg.matrix_power(P, 2)
predicted_clusters = np.argmax(P_t, axis=1)
unique_clusters, indexed_clusters = np.unique(predicted_clusters, return_inverse=True)

print("🌤️ Step 4: Computing Conformal Prediction (Honest Confidence Forecasts)...")
train_idx = np.arange(0, n_cells, 2)
cal_idx = np.arange(1, n_cells, 2)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_counts[train_idx], y_true[train_idx])

# Calibrate confidence thresholds
probs_cal = knn.predict_proba(X_counts[cal_idx])
true_labels_cal = y_true[cal_idx]
scores_cal = 1.0 - probs_cal[np.arange(len(cal_idx)), true_labels_cal]

alpha = 0.1 # 90% confidence target
q_level = np.ceil((len(cal_idx) + 1) * (1 - alpha)) / len(cal_idx)
q_level = min(max(q_level, 0.0), 1.0)
qhat = np.quantile(scores_cal, q_level)

# Flag ambiguous transition cells
all_probs = knn.predict_proba(X_counts)
prediction_sets = []
ambiguous_count = 0

for prob in all_probs:
    scores = 1.0 - prob
    valid_classes = np.where(scores <= qhat)[0]
    prediction_sets.append(valid_classes)
    if len(valid_classes) > 1:
        ambiguous_count += 1

print(f"📊 Dashboard Check: {ambiguous_count}/{n_cells} cells flagged as 'Ambiguous/Transitioning'.")

# Generate the high-quality dashboard plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: True Biology
axes[0].scatter(X_counts[:, 0], X_counts[:, 1], c=y_true, cmap='Set2', s=40, edgecolors='k', alpha=0.8)
axes[0].set_title("True Biology (Ground Truth)\nIncludes Rare Cell Type (Small Cluster)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Gene 1 Expression")
axes[0].set_ylabel("Gene 2 Expression")

# Plot 2: IGMC Lite Discovery
scatter = axes[1].scatter(X_counts[:, 0], X_counts[:, 1], c=indexed_clusters, cmap='jet', s=40, edgecolors='k', alpha=0.8)
axes[1].set_title("IGMC-Lite Discoveries\nSmart Ruler + Ink-Drop Trapped Clusters", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Gene 1 Expression")

# Highlight transitioning cells
for i, p_set in enumerate(prediction_sets):
    if len(p_set) > 1:
        axes[1].plot(X_counts[i, 0], X_counts[i, 1], 'ro', markersize=8, fillstyle='none', alpha=0.6)

axes[1].plot([], [], 'ro', markersize=8, fillstyle='none', label='Ambiguous/Transitioning Cell')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('igmc_results.png', dpi=300)
print("🎉 Success! Results cleanly saved as 'igmc_results.png' without display issues!")
